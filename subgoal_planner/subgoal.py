import torch
import torch.nn as nn
import numpy as np
import math
from tdm.td3 import TD3
from copy import copy
from gym.envs.robotics.fetch.push_labyrinth2 import goal_distance
from envs import make_env
from custom_start import get_args_and_initialize
import time
from utils.image_util import store_image_array_at

def np_to_pytorch(ob_np, goal_np, taus_np, num_subprobs, batch_size=1):
    ob_np = np.tile(ob_np, (batch_size, 1, 1))
    goal_np = np.tile(goal_np, (batch_size, 1, 1))
    taus_np = np.tile(taus_np.reshape((1, num_subprobs, 1)), (batch_size, 1, 1))

    ob = torch.from_numpy(ob_np)
    goal = torch.from_numpy(goal_np)
    taus = torch.from_numpy(taus_np)

    return ob, goal, taus

class SubgoalPlanner(nn.Module):
    def __init__(self, vae, act_cr_model, args):
        super(SubgoalPlanner, self).__init__()
        self.vae = vae
        self.act_cr_model = act_cr_model
        self.args = args
        self.lower_bounds = np.repeat(-2, self.args.latent_dim)
        self.upper_bounds = np.repeat(2, self.args.latent_dim)
        self.cem_optimizer_kwargs = {'batch_size': 1000, 'frac_top_chosen': 0.05,
                                     'num_iters': 15, 'use_init_subgoals': True}
        self.realistic_hard_constraint_threshold = 1.75
        self.realistic_subgoal_weight = 0.001
        true_prior_mu = torch.from_numpy(np.zeros(self.args.latent_dim, dtype=float))
        true_prior_std = torch.from_numpy(np.ones(self.args.latent_dim, dtype=float))
        self.true_prior_distr = torch.distributions.Normal(true_prior_mu, true_prior_std)

    def get_intervals(self, maxT, max_tau):
        self.maxT = maxT
        self.max_tau = max_tau
        self.num_subprobs = int(math.ceil((maxT) / (max_tau)))
        self.num_subgoals = self.num_subprobs - 1

        taus_np = np.ones(self.num_subprobs) * ((maxT) // (self.num_subprobs))
        extra_time = int(np.sum(taus_np) - maxT)
        i = 0
        while extra_time != 0:
            if extra_time > 0:
                taus_np[i] -= 1
                extra_time -=1
            else:
                taus_np[i] += 1
                extra_time +=1
            i = (i+1) % self.num_subprobs
        self.taus = taus_np
        return copy(taus_np)

    def get_subgoal(self, interval_index, ob, goal, use_reprojected=True):
        if interval_index == len(self.taus)-1:#the last goal is the real one
            return goal
        self._update_subgoals(interval_index, ob, goal)
        if use_reprojected:
            return self.subgoals_reproj[interval_index]
        else:
            return self.subgoals[interval_index]

    def _get_vae_prior(self):
        mu_np = np.zeros(self.args.latent_dim)
        std_np = np.ones(self.args.latent_dim)

        return mu_np, std_np

    def _update_subgoals(self, interval_index, ob, goal):
        if interval_index == 0:#first call
            lower_bounds = np.tile(self.lower_bounds, self.num_subgoals)
            upper_bounds = np.tile(self.upper_bounds, self.num_subgoals)
            initial_x = np.random.uniform(lower_bounds, upper_bounds)
        else:
            initial_x = self.subgoals[interval_index:]
        initial_x = initial_x.flatten()

        opt_sample, opt_sample_reproj = self._optimize_cem(interval_index, init_subgoals_np=initial_x,
                                      ob_np=ob,goal_np=goal,taus_np=self.taus[interval_index:])

        if interval_index == 0:
            self.subgoals = opt_sample.cpu().data.numpy()
            self.subgoals_reproj = opt_sample_reproj.cpu().data.numpy()
        else:
            self.subgoals[interval_index:] = opt_sample.cpu().data.numpy()
            self.subgoals_reproj[interval_index:]  = opt_sample_reproj.cpu().data.numpy()

    def _optimize_cem(self, interval_index, init_subgoals_np, ob_np, goal_np, taus_np):
        optimizer_kwargs = self.cem_optimizer_kwargs
        num_iters = optimizer_kwargs['num_iters']
        batch_size = optimizer_kwargs['batch_size']
        # if the batch sizes are variable over iterations
        if hasattr(batch_size, "__len__"):
            batch_sizes = [batch_size[0]]*(num_iters//2) + [batch_size[1]]*(num_iters-(num_iters//2))
        else:
            batch_sizes = [batch_size]*num_iters

        frac_top_chosen = optimizer_kwargs['frac_top_chosen']
        # if the top chosen is variable over iterations
        if hasattr(frac_top_chosen, "__len__"):
            frac_top_chosens = np.array(
                [frac_top_chosen[0]]*(num_iters//2) + [frac_top_chosen[1]]*(num_iters-(num_iters//2))
            )
        else:
            frac_top_chosens = np.ones(num_iters) * frac_top_chosen

        use_init_subgoals = optimizer_kwargs.get('use_init_subgoals', False)
        obs, goals, taus = np_to_pytorch(ob_np, goal_np, taus_np, self.num_subprobs - interval_index,
                                         batch_size=batch_sizes[0])

        vae_mu, vae_std = self._get_vae_prior()
        mu_np = np.tile(vae_mu, (self.num_subgoals - interval_index, 1)).flatten()
        std_np = np.tile(vae_std * 1.5, (self.num_subgoals - interval_index, 1)).flatten()

        if use_init_subgoals and interval_index != 0:
            mu_np = init_subgoals_np

        mu = torch.from_numpy(mu_np).float()
        std = torch.from_numpy(std_np).float()

        for i in range(num_iters):
            # if batch sized changed from prev iteration
            if i > 0 and batch_sizes[i] != batch_sizes[i-1]:
                obs, goals, taus = np_to_pytorch(ob_np,goal_np,taus_np,self.num_subprobs - interval_index,
                                                 batch_size=batch_sizes[i])

            samples = torch.distributions.Normal(mu, std).sample_n(batch_sizes[i])
            losses = self._loss(interval_index, samples, obs, goals, taus)

            sorted_losses, sorted_indices = torch.sort(losses)
            num_top_chosen = int(frac_top_chosens[i] * batch_sizes[i])
            elite_indices = sorted_indices[:num_top_chosen]
            elites = samples[elite_indices]
            mu = torch.mean(elites, dim=0)
            std = torch.std(elites, dim=0)

        opt_sample = elites[0].reshape(-1, self.args.latent_dim)
        opt_sample_reproj, _ = self.vae.encode(self.vae.decode(opt_sample))

        return opt_sample, opt_sample_reproj

    def _loss(self, interval_index, subgoals, obs, goals, taus, info=False):#becom exmplary the n samples of z v per subgoal flttende
        subgoals = subgoals.float()
        obs = obs.float()
        taus = taus.float()
        batch_size = int(obs.numel() / self.args.latent_dim)
        subgoals = subgoals.view(batch_size, self.num_subgoals - interval_index, self.args.latent_dim)#reshaped to[n, latent_dim, subgolas]
        original_subgoals = subgoals
        #if self.reproject_encoding:#TODO
        subgoals, _ = self.vae.encode(self.vae.decode(subgoals))
        path = torch.cat((obs, subgoals, goals), dim=1)
        s = path[:, :-1].contiguous().view(-1, self.args.latent_dim).float()
        g = path[:, 1:].contiguous().view(-1, self.args.latent_dim).float()
        taus = taus.view(-1, 1)
        n = s.shape[0]
        v_vals = None
        bs = 100000
        for i in range(0, n, bs):
            '''if self.infinite_horizon:
                states_and_goals = torch.cat((s[i:i + bs], g[i:i + bs]), dim=1)
                a = self.mf_policy(states_and_goals).detach()
                batch_v_vals = self.qf(states_and_goals, a).detach() / self.reward_scale
            else:'''
            a = self.act_cr_model.get_action(s[i:i + bs].cuda(), g[i:i + bs].cuda(), taus[i:i + bs].cuda())
            a = torch.from_numpy(a).cuda().float()
            batch_v_vals = self.act_cr_model.get_Q_val(s[i:i + bs].cuda(), a.cuda(), g[i:i + bs].cuda(), taus[i:i + bs].cuda())
            batch_v_vals = torch.from_numpy(batch_v_vals).float()
            if v_vals is None:
                v_vals = batch_v_vals
            else:
                v_vals = torch.cat((v_vals, batch_v_vals), dim=0)

        if v_vals.size()[1] > 1:
            v_vals = -torch.norm(v_vals, p=2, dim=-1)#todo
        v_vals = v_vals.view(batch_size, self.num_subprobs - interval_index)

        min_v_val, _ = torch.min(v_vals, dim=-1)
        v_val = min_v_val

        realistic_subgoal_rew = self._realistic_subgoal_reward(original_subgoals)
        #is_outside_threshold = torch.abs(original_subgoals) > self.realistic_hard_constraint_threshold
        #is_outside_threshold = is_outside_threshold.float()
        #realistic_hard_constraint_subgoal_rew = - 1e6 * is_outside_threshold

        realistic_subgoal_rew = realistic_subgoal_rew.view(batch_size, self.num_subgoals - interval_index)
        realistic_subgoal_rew = torch.sum(realistic_subgoal_rew, dim=-1)

        '''realistic_hard_constraint_subgoal_rew = realistic_hard_constraint_subgoal_rew.view(
            batch_size, self.num_subgoals * self.arg.latent_dim)
        realistic_hard_constraint_subgoal_rew = torch.sum(realistic_hard_constraint_subgoal_rew, dim=-1)

        if self.use_realistic_hard_constraint:
            loss = - (self.realistic_subgoal_weight * realistic_subgoal_rew
                      + realistic_hard_constraint_subgoal_rew
                      + v_val).squeeze(0)
        else:'''
        loss = -(self.realistic_subgoal_weight * realistic_subgoal_rew + v_val).squeeze(0)
        return loss

    def _realistic_subgoal_reward(self, subgoals):
        if type(subgoals) is np.ndarray:
            subgoals = torch.from_numpy(subgoals)
        if hasattr(self, "true_prior_distr"):
            log_prob = self.true_prior_distr.log_prob(subgoals)
            log_prob = torch.sum(log_prob, dim=-1)
            return log_prob
        else:
            return torch.from_numpy(np.zeros(subgoals.shape[:-1]))

def episode_loop_with_subgoal_planner(epoch, episode, planner, env, act_cr_model, args):
    assert isinstance(planner, SubgoalPlanner)
    assert isinstance(act_cr_model, TD3)
    obs = env.reset()
    first_obs = copy(obs)
    goal = obs['goal_latent']
    episode_reward = 0

    intervals = planner.get_intervals(args.tdm_env_steps, args.max_tau)
    for interval_i in range(len(intervals)):
        sub_goal = planner.get_subgoal(interval_i, obs['state_latent'], goal)
        with torch.no_grad():
            sub_as_image = env._env.vae_model.decode(sub_goal)
            sub_as_image = env._env.torch_to_np_image(sub_as_image)
            store_image_array_at(sub_as_image, args.dirpath + 'images_for_video/',
                                 'subgoal_epoch_{}_episode_{}_interval_{}'.format(epoch, episode,interval_i))
        for step in range(int(intervals[interval_i])):
            store_image_array_at(obs['state_image'], args.dirpath + 'images_for_video/',
                                 'frame_epoch_{}_episode_{}_interval_{}_step_{}'.format(epoch, episode,
                                                                                        interval_i, step))
            rem_steps = args.max_tau - step
            state = obs['state_latent']
            action = act_cr_model.get_action(state, sub_goal, rem_steps)[0]
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            '''if done or step == self.args.tdm_env_steps - 1:
            self.replay_buffer.store_trajectory(trajectory)'''
            obs = next_obs

        store_image_array_at(obs['state_image'], args.dirpath + 'images_for_video/',
                             'frame_epoch_{}_episode_{}_interval_{}_step_{}'.format(epoch, episode,
                                                                                    interval_i, step))
        latent_dist = np.linalg.norm(goal-sub_goal, ord=0)
        subgoal_last_dist = goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
        args.csv_logger.add_log('latent_distant_sub', latent_dist)
        args.csv_logger.add_log('distance_sub', subgoal_last_dist)
        args.csv_logger._finish_it_log('sub_goal_loop', interval_i)

    args.csv_logger.add_log('episode_reward', episode_reward)
    args.csv_logger.add_log('episode_steps', 100)#todo must be calculated in other way
    args.csv_logger.add_log('success', float(done))
    distance = goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
    args.csv_logger.add_log('distance_to_goal', distance)
    #
    args.csv_logger.finish_episode_log(episode)

def setup_subgoal(args, td3_recover_filename):
    env = make_env(args)
    sample_obs = env.reset()
    latent_dim = sample_obs['state_latent'].shape[0]
    action_dim = env.action_space.shape[0]
    td3_actor_critic = TD3(state_dim=latent_dim, action_dim=action_dim, goal_dim=latent_dim,
                           rem_steps_dim=1, max_action=env.action_space.high, args=args)
    td3_actor_critic.load(filename=td3_recover_filename)
    td3_actor_critic.actor.eval()
    td3_actor_critic.critic.eval()
    planner = SubgoalPlanner(env._env.vae_model, td3_actor_critic, args)
    return env, planner, td3_actor_critic


if __name__ == '__main__':
    args = get_args_and_initialize()
    env, planner, td3_actor_critic = setup_subgoal(args, td3_recover_filename='td3_tr_last')
    for epoch in range(1):
        start = time.time()
        for episode in range(4):
            episode_loop_with_subgoal_planner(epoch, episode, planner, env, td3_actor_critic, args)
        end = time.time()
        args.csv_logger.add_log('time', end-start)
        args.csv_logger.finish_epoch_log(epoch)
