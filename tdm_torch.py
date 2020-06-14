from custom_start import get_args_and_initialize
import time
from tdm_storage_structures import TDM_Trajectory, ReplayBuffer_TDM
from envs import make_env
from ou_noise import OUNoise
from td3 import TD3

#more than a model this learns a q function based on goal_conditioned
class TDM:
    def __init__(self, act_cr_model, replay_buffer, env, ou_noiser, args):
        self.act_cr_model = act_cr_model
        self.replay_buffer = replay_buffer
        self.ou_noiser = ou_noiser
        self.env = env
        self.args = args

    def training_loop(self, start_epoch):
        for epoch in range(start_epoch, args.epochs):# apparently 500
            self._epoch_start_time = time.time()
            obs = self.env.reset()
            goal = self.env.get_goal()
            self.ou_noiser.reset()
            episode_reward = 0
            trajectory = TDM_Trajectory()
            for step in range(self.args.num_env_steps_per_epoch):
                rem_steps = self.args.max_rem_steps - step
                action = self.act_cr_model.get_action(obs, goal, rem_steps)
                action = self.ou_noiser.get_action(action, step)
                next_obs, reward, done, _ = self.env.step(action)
                self._n_env_steps_total += 1 #todo this is important
                episode_reward += reward

                trajectory.store_step(obs, goal, rem_steps, action, next_obs, reward, done)
                if done or step == args.num:
                    self.replay_buffer.store_trajectory(trajectory)

                if self.replay_buffer.size() > args.min_replay_size_for_training:#30000:
                    for training_step in range(args.training_steps):
                        batch = self.replay_buffer.sample_batch(self.args.batch_size)
                        #todo transform batch into torch
                        self.act_cr_model.train(batch)
                        self._n_training_steps_total += 1
                    #todo DO evaluation
                if done:
                    break
                obs = next_obs
                '''logger.log("Epoch Duration: {0}".format(
                            time.time() - self._epoch_start_time
                        ))
                        logger.log("Started Training: {0}".format(self._can_train()))
                        logger.pop_prefix()'''

def setup_tdm(args, recover_filename=None):
    env = make_env(args)
    replay_buffer =  ReplayBuffer_TDM(args, env)
    ou_noiser = OUNoise(env.action_space)
    latent_dim = 1  # TODO
    action_dim = env.action_space.sample.flatten()
    td3_actor_critic = TD3(latent_dim, action_dim, latent_dim, 1, env.action_space.max, args)
    tdm = TDM(td3_actor_critic,replay_buffer, env, ou_noiser, args)
    return tdm, td3_actor_critic, ou_noiser, replay_buffer, env

from gym.wrappers.pixel_observation import PixelObservationWrapper
def setup_tdm_with_vae(args, recover_filename=None):
    env = make_env(args)
    t = env.reset()
    t2 = env.get_obs()
    print(t)
    print(t)





if __name__ == "__main__":
    args = get_args_and_initialize()    #start_training(args)
    '''tdm, _, _, _, _ = setup_tdm(args)
    tdm.training_loop(0)'''
    setup_tdm_with_vae(args)