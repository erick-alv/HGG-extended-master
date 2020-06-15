from custom_start import get_args_and_initialize
import time
from tdm_storage_structures import TDM_Trajectory, ReplayBuffer_TDM
from envs import make_env
from ou_noise import OUNoise
from td3 import TD3
import numpy as np

#more than a model this learns a q function based on goal_conditioned
class TDM:
    def __init__(self, act_cr_model, replay_buffer, env, ou_noiser, args):
        self.act_cr_model = act_cr_model
        self.replay_buffer = replay_buffer
        self.ou_noiser = ou_noiser
        self.env = env
        self.args = args

    def training_loop(self, start_epoch):
        for epoch in range(start_epoch, 12):#args.tdm_training_epochs):
            self._epoch_start_time = time.time()
            obs = self.env.reset()
            goal = np.concatenate((obs['desired_goal'], obs['goal_latent']))
            self.ou_noiser.reset()
            episode_reward = 0
            trajectory = TDM_Trajectory()
            for step in range(self.args.tdm_env_steps):#todo the max number of steps and how many steps are take differs in leap, but actually does not make sense what they do and steps_left is for them the same because then they randomsample them
                rem_steps = self.args.tdm_env_steps - step
                state = np.concatenate((obs['observation'], obs['state_latent']))
                action = self.act_cr_model.get_action(state, goal, rem_steps)
                action = self.ou_noiser.get_action(action, step)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = np.concatenate((next_obs['observation'], next_obs['state_latent']))
                episode_reward += reward

                trajectory.store_step(state, goal, rem_steps, action, next_state, reward, done)
                if done or step == self.args.tdm_env_steps - 1:
                    self.replay_buffer.store_trajectory(trajectory)

                if self.replay_buffer.steps_counter >= 200:#args.min_replay_size:
                    for training_step in range(args.training_steps):
                        batch = self.replay_buffer.sample_batch(self.args.batch_size, transform_to_tensor=True)
                        #todo transform batch into torch
                        self.act_cr_model.train(batch)
                    #todo DO evaluation
                if done:
                    break
                obs = next_obs
                '''logger.log("Epoch Duration: {0}".format(
                            time.time() - self._epoch_start_time
                        ))
                        logger.log("Started Training: {0}".format(self._can_train()))
                        logger.pop_prefix()'''
            if epoch % self.args.checkpoint_interval == 0:
                self.act_cr_model.save_train_checkpoint('td3_tr', epoch)
            print('Epoch reward: ', episode_reward)
        self.act_cr_model.save('td3_tr_last')

def setup_tdm(args, recover_filename=None):
    env = make_env(args)
    sample_obs = env.reset()
    latent_dim = sample_obs['state_latent'].shape[0]
    obs_dim = sample_obs['observation'].shape[0]
    desired_dim = sample_obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    td3_actor_critic = TD3(state_dim=obs_dim+latent_dim, action_dim=action_dim, goal_dim=desired_dim+latent_dim,
                           rem_steps_dim=1, max_action=env.action_space.high, args=args)
    if recover_filename is not None:
        td3_actor_critic.load(filename=recover_filename)
    replay_buffer = ReplayBuffer_TDM(args, env)
    ou_noiser = OUNoise(env.action_space)
    tdm = TDM(td3_actor_critic,replay_buffer, env, ou_noiser, args)


    return tdm, td3_actor_critic, ou_noiser, replay_buffer, env



if __name__ == "__main__":
    args = get_args_and_initialize()    #start_training(args)
    tdm, _, _, _, _ = setup_tdm(args)
    tdm.training_loop(0)