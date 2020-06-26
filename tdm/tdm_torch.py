from custom_start import get_args_and_initialize
import time
from tdm.tdm_storage_structures import TDM_Trajectory, ReplayBuffer_TDM
from envs import make_env
from tdm.ou_noise import OUNoise
from tdm.td3 import TD3
from temp_func import conc_latent
from gym.envs.robotics.fetch.push_labyrinth2 import goal_distance
from copy import copy

#more than a model this learns a q function based on goal_conditioned
class TDM:
    def __init__(self, act_cr_model, replay_buffer, env, ou_noiser, args):
        self.act_cr_model = act_cr_model
        self.replay_buffer = replay_buffer
        self.ou_noiser = ou_noiser
        self.env = env
        self.args = args

    def training_loop(self, start_epoch):
        for epoch in range(start_epoch, self.args.tdm_training_epochs):
            start_time = time.time()
            for episode in range(self.args.tdm_training_episodes):
                obs = self.env.reset()
                first_obs = copy(obs)
                if self.args.just_latent:
                    goal = obs['goal_latent']
                else:
                    goal = conc_latent(obs['desired_goal'], obs['goal_latent'])
                self.ou_noiser.reset()
                episode_reward = 0
                trajectory = TDM_Trajectory()
                cr_tr_loss, ac_tr_loss, cr_ev_loss, ac_ev_loss = None, None, None, None

                for step in range(self.args.tdm_env_steps):
                    rem_steps = self.args.max_tau - (step % (self.args.max_tau+1))
                    if self.args.just_latent:
                        state = obs['state_latent']
                    else:
                        state = conc_latent(obs['observation'], obs['state_latent'])
                    action = self.act_cr_model.get_action(state, goal, rem_steps)
                    action = self.ou_noiser.get_action(action, step)
                    next_obs, reward, done, _ = self.env.step(action)
                    if self.args.just_latent:
                        next_state = next_obs['state_latent']
                    else:
                        next_state = conc_latent(next_obs['observation'], next_obs['state_latent'])
                    episode_reward += reward

                    trajectory.store_step(state, goal, rem_steps, action, next_state, reward, done,
                                          _goal=first_obs['desired_goal'], _state=obs['achieved_goal'],
                                          _next_state=next_obs['achieved_goal'])
                    if done or step == self.args.tdm_env_steps - 1:
                        self.replay_buffer.store_trajectory(trajectory)

                    #TRAINING OF MODEL
                    if self.replay_buffer.steps_counter >= args.min_replay_size:
                        for training_step in range(args.training_steps):
                            batch = self.replay_buffer.sample_batch(self.args.batch_size, transform_to_tensor=True)
                            critic_loss, actor_loss = self.act_cr_model.train(batch)
                            if critic_loss is not None:
                                cr_tr_loss = cr_tr_loss + critic_loss if cr_tr_loss is not None else critic_loss
                            if actor_loss is not None:
                                ac_tr_loss = ac_tr_loss + actor_loss if ac_tr_loss is not None else actor_loss

                    if done:
                        break
                    obs = next_obs

                self.args.csv_logger.add_log('critic_tr_loss', cr_tr_loss)
                self.args.csv_logger.add_log('actor_tr_loss', ac_tr_loss)
                self.args.csv_logger.add_log('episode_reward', episode_reward)
                self.args.csv_logger.add_log('episode_steps', step+1)
                self.args.csv_logger.add_log('success', float(done))
                distance = goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
                self.args.csv_logger.add_log('distance_to_goal', distance)
                #EVALUATION
                if self.replay_buffer.steps_counter >= self.args.min_replay_size:
                    for training_step in range(self.args.evaluation_steps):
                        batch = self.replay_buffer.sample_batch(self.args.batch_size, transform_to_tensor=True)
                        critic_loss, actor_loss = self.act_cr_model.evaluate(batch)
                        if critic_loss is not None:
                            cr_ev_loss = cr_ev_loss + critic_loss if cr_ev_loss is not None else critic_loss
                        if actor_loss is not None:
                            ac_ev_loss = ac_ev_loss + actor_loss if ac_ev_loss is not None else actor_loss

                self.args.csv_logger.add_log('critic_eval_loss', cr_ev_loss)
                self.args.csv_logger.add_log('actor_eval_loss', ac_ev_loss)
                #
                self.args.csv_logger.finish_episode_log(episode)

            time_needed = time.time() - start_time
            self.args.csv_logger.add_log('time', time_needed)
            if epoch % self.args.checkpoint_interval == 0:
                self.act_cr_model.save_train_checkpoint('td3_tr', epoch)
            #
            print('finished epoch ', epoch)
            print('last actor loss ', cr_tr_loss)
            print('last ep success ', done)
            print('time needed ', time_needed)
            print('______________________________')
            self.args.csv_logger.finish_epoch_log(epoch)
        self.act_cr_model.save('td3_tr_last')

def setup_tdm(args, recover_filename=None):
    env = make_env(args)
    sample_obs = env.reset()
    latent_dim = sample_obs['state_latent'].shape[0]
    obs_dim = sample_obs['observation'].shape[0]
    desired_dim = sample_obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    '''td3_actor_critic = TD3(state_dim=obs_dim+latent_dim, action_dim=action_dim, goal_dim=desired_dim+latent_dim,
                           rem_steps_dim=1, max_action=env.action_space.high, args=args)'''
    td3_actor_critic = TD3(state_dim=latent_dim, action_dim=action_dim, goal_dim=latent_dim,
                           rem_steps_dim=1, max_action=env.action_space.high, args=args)
    if recover_filename is not None:
        td3_actor_critic.load(filename=recover_filename)
    replay_buffer = ReplayBuffer_TDM(args, env)
    ou_noiser = OUNoise(env.action_space)
    tdm = TDM(td3_actor_critic,replay_buffer, env, ou_noiser, args)
    return tdm, td3_actor_critic, ou_noiser, replay_buffer, env


if __name__ == "__main__":
    args = get_args_and_initialize()
    tdm, _, _, _, _ = setup_tdm(args)
    tdm.training_loop(0)