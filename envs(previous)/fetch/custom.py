import numpy as np
import gym
from utils.os_utils import remove_color
from vae.vae_wrapper import VAEWrapper
from vae.goal_pixel_wrapper import PixelAndGoalWrapper
import copy


class CustomGoalEnv():
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env)
        if hasattr(args, 'img_dim'):
            self.env.env.set_goal_img_size(args.img_dim)
        if hasattr(args, 'wrap_with_vae') and args.wrap_with_vae:
            self._env = VAEWrapper(args, env=self.env, pixels_only=False,
                                   render_kwargs={'state_image':args.vae_wrap_kwargs},
                                   pixel_keys=('state_image', ))
        elif hasattr(args, 'wrap_with_pixel') and args.wrap_with_pixel:
            self._env = PixelAndGoalWrapper(args, env=self.env, pixels_only=False,
                                            render_kwargs={'state_image':args.pixel_wrap_kwargs},
                                            pixel_keys=('state_image', ))
            self.env.env.set_goal_img_size(args.img_dim)

        self.np_random = self.env.env.np_random
        self.distance_threshold = self.env.env.distance_threshold
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_episode_steps = self.env._max_episode_steps
        self.fixed_obj = False
        self.has_object = self.env.env.has_object
        self.obj_range = self.env.env.obj_range
        # self.target_range = self.env.env.target_range
        self.target_offset = self.env.env.target_offset
        self.target_in_the_air = self.env.env.target_in_the_air
        if self.has_object: self.height_offset = self.env.env.height_offset

        self.render = self.env.render
        if (hasattr(args, 'wrap_with_vae') and args.wrap_with_vae) \
                or (hasattr(args, 'wrap_with_pixel') and args.wrap_with_pixel):
            self.get_obs = self._env._get_obs
        else:
            self.get_obs = self.env.env._get_obs
        self.reset_sim = self.env.env._reset_sim

        self.reset_ep()
        self.env_info = {
            'Rewards': self.process_info_rewards,  # episode cumulative rewards
            'Distance': self.process_info_distance,  # distance in the last step
            'Success@green': self.process_info_success  # is_success in the last step
        }
        self.env.reset()
        self.fixed_obj = True
        if False and hasattr(self.args, 'type_experiment') and self.args.type_experiment == 'dyn_stack':
            # stack the new information dynamic object
            self.stack_counter = 0
            self.stack_max = 4

    def compute_reward(self, achieved, goal):
        # achieved is a tuple of two goals
        return self.env.env.compute_reward(achieved[0], goal, None)
        # Original
        # dis = goal_distance(achieved[0], goal)
        # return -1.0 if dis > self.distance_threshold else 0.0

    def compute_distance(self, achieved, goal):
        return np.sqrt(np.sum(np.square(achieved - goal)))

    def process_info_rewards(self, obs, reward, info):
        self.rewards += reward
        return self.rewards

    def process_info_distance(self, obs, reward, info):
        return self.compute_distance(obs['achieved_goal'], obs['desired_goal'])

    def process_info_success(self, obs, reward, info):
        return info['is_success']

    def process_info(self, obs, reward, info):
        return {
            remove_color(key): value_func(obs, reward, info)
            for key, value_func in self.env_info.items()
        }

    def step(self, action):
        action = action.copy()
        if False and hasattr(self.args, 'type_experiment') and self.args.type_experiment == 'dyn_stack':
            # block movement up and down
            action = np.array([action[0], action[1], 0.0, action[2]])

        obs, reward, done, info = self.env.step(action)
        info = self.process_info(obs, reward, info)
        reward = self.compute_reward((obs['achieved_goal'], self.last_obs['achieved_goal']),
                                     obs['desired_goal'])  # TODO: why second argument if it is then ignored??

        obs = self.get_obs()
        if False and hasattr(self.args, 'type_experiment') and self.args.type_experiment == 'dyn_stack':
            # stack the new information dynamic object
            last_stacked = self.last_obs['stack']
            stacked = last_stacked.reshape((self.stack_max, 4))#todo not always for
            new_stacked = np.concatenate([np.array([obs['dyn_pos']]), stacked[:-1, :]])
            obs['stack'] = new_stacked.flatten()

        self.last_obs = copy.deepcopy(obs)
        # imaginary infinity horizon (without done signal)
        #return obs, reward, False, info
        return obs, reward, not reward==-1.0, info #TODO ask why; and notice that the last one is always set to true/might be becuase one wrapper sets it to true when enden

    def reset_ep(self):
        self.rewards = 0.0


    @property
    def sim(self):
        return self.env.env.sim

    @sim.setter
    def sim(self, new_sim):
        self.env.env.sim = new_sim

    @property
    def initial_state(self):
        return self.env.env.initial_state

    @property
    def initial_gripper_xpos(self):
        return self.env.env.initial_gripper_xpos.copy()

    @property
    def goal(self):
        return self.env.env.goal.copy()

    @goal.setter
    def goal(self, value):
        self.env.env.goal = value.copy()

    def generate_goal(self):
        """
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.target_offset
            if self.args.env == 'FetchSlide-v1':
                goal[0] += self.target_range * 0.5
                goal[1] += np.random.uniform(-self.target_range, self.target_range) * 0.5
            else:
                goal[0] += np.random.uniform(-self.target_range, self.target_range)
                goal[1] += self.target_range
            # goal[1] += np.random.uniform(-self.target_range, self.target_range) # TODO: changed
            goal[2] = self.height_offset + int(self.target_in_the_air) * 0.45
        else:
            goal = self.initial_gripper_xpos[:3] + np.array(
                [np.random.uniform(-self.target_range, self.target_range), self.target_range, self.target_range])
        return goal.copy()
        """
        return self.env.env._sample_goal()

    def reset(self, **kwargs):
        self.reset_ep()
        self.env.env._reset_sim(**kwargs)
        """
        self.sim.set_state(self.initial_state)

        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2].copy()
            random_offset = np.random.uniform(-1, 1) * self.obj_range * self.args.init_offset
            if self.args.env == 'FetchSlide-v1':
                object_xpos -= np.array([self.obj_range * 0.5, random_offset])
            else:
                object_xpos -= np.array([random_offset, self.obj_range])
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        """
        self.goal = self.generate_goal()
        self.env.reset()
        obs = self.get_obs()
        if False and hasattr(self.args, 'type_experiment') and self.args.type_experiment == 'dyn_stack':
            # stack the new information dynamic object
            pad = np.tile(np.zeros_like(obs['dyn_pos']), self.stack_max-1)
            obs['stack'] = np.concatenate([obs['dyn_pos'], pad])
        self.last_obs = copy.deepcopy(obs)
        return copy.deepcopy(obs)

    def generate_goal(self):
        return self.env.env._sample_goal()



