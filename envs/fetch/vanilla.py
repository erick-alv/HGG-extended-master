import gym
import numpy as np
from envs.utils import goal_distance, goal_distance_obs
import copy
from utils.os_utils import remove_color
from vae_env_inter import take_goal_image, take_obstacle_image, take_image_objects, goal_latent_from_images, \
	obstacle_latent_from_images, latents_from_images
from PIL import Image


class VanillaGoalEnv():
	def __init__(self, args):
		self.args = args
		self.env = gym.make(args.env)
		self.np_random = self.env.env.np_random

		self.distance_threshold = self.env.env.distance_threshold

		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.max_episode_steps = self.env._max_episode_steps

		self.fixed_obj = False
		self.has_object = self.env.env.has_object
		self.target_in_the_air = self.env.env.target_in_the_air
		if self.has_object: self.height_offset = self.env.env.height_offset
		if not hasattr(self.env.env, 'target_goal_center'):
			self.target_goal_center = None
		if not hasattr(self.env.env, 'object_center'):
			self.object_center = None

		self.render = self.env.render
		self.reset_sim = self.env.env._reset_sim

		self.reset_ep()
		self.env_info = {
			'Rewards': self.process_info_rewards, # episode cumulative rewards
			'Distance': self.process_info_distance, # distance in the last step
			'Success@green': self.process_info_success # is_success in the last step
		}

	def compute_reward(self, achieved, goal):
		dis = goal_distance(achieved[0], goal)
		return -1.0 if dis>self.distance_threshold else 0.0

	def compute_distance(self, achieved, goal):
		return np.sqrt(np.sum(np.square(achieved-goal)))

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

	def get_obs(self):
		if hasattr(self.args, 'vae_dist_help') and self.args.vae_dist_help:
			obs = self.env.env._get_obs()
			obs['desired_goal_latent'] = self.desired_goal_latent.copy()
			obs['achieved_goal_latent'] = self.achieved_goal_latent.copy()
			obs['achieved_goal_size_latent'] = self.achieved_goal_size_latent.copy()
			obs['desired_goal_size_latent'] = self.desired_goal_size_latent.copy()
			#obs['achieved_goal_image'] = self.achieved_goal_image.copy()
			obs['obstacle_latent'] = self.obstacle_latent.copy()
			obs['obstacle_size_latent'] = self.obstacle_size_latent.copy()
			return obs
		else:
			return self.env.env._get_obs()

	def step(self, action):
		# imaginary infinity horizon (without done signal)
		obs, reward, done, info = self.env.step(action)
		if hasattr(self.args, 'vae_dist_help') and self.args.vae_dist_help:
			if self.args.vae_type == 'monet' or self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
				achieved_image = take_image_objects(self, self.args.img_size)
				if self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
					lg, lg_s, lo, lo_s = latents_from_images(np.array([achieved_image]), self.args)
					self.achieved_goal_size_latent = lg_s[0].copy()
				else:
					lg, lo, lo_s = latents_from_images(np.array([achieved_image]), self.args)
				self.achieved_goal_latent = lg[0].copy()
				self.obstacle_latent = lo[0].copy()
				self.obstacle_size_latent = lo_s[0].copy()
			else:
				achieved_goal_image = take_goal_image(self, self.args.img_size)
				latents_goal = goal_latent_from_images(np.array([achieved_goal_image]), self.args)
				self.achieved_goal_latent = latents_goal[0].copy()

				obstacle_image = take_obstacle_image(self, self.args.img_size)
				latents_obstacle, latents_o_size = obstacle_latent_from_images(np.array([obstacle_image]), self.args)
				self.obstacle_latent = latents_obstacle[0].copy()
				self.obstacle_size_latent = latents_o_size[0].copy()
				#self.achieved_goal_image = achieved_goal_image.copy()


			info = self.process_info(obs, reward, info)

		else:
			info = self.process_info(obs, reward, info)
			reward = self.compute_reward((obs['achieved_goal'], self.last_obs['achieved_goal']), obs['desired_goal'])
		obs = self.get_obs()
		self.last_obs = obs.copy()
		return obs.copy(), reward, False, info

	def reset_ep(self):
		if hasattr(self.args, 'vae_dist_help') and self.args.vae_dist_help:
			obs = self.env.env._get_obs()
			if self.args.vae_type == 'monet' or self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
				self.env.env._move_object(position=obs['desired_goal'].copy())
				desired_goal_image = take_image_objects(self, self.args.img_size)
				self.env.env._move_object(position=obs['achieved_goal'].copy())
				achieved_goal_image = take_image_objects(self, self.args.img_size)
				if self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
					lg, lg_s, lo, lo_s = latents_from_images(np.array([desired_goal_image, achieved_goal_image]), self.args)
					self.desired_goal_latent = lg[0].copy()
					self.desired_goal_size_latent = lg_s[0].copy()
					self.achieved_goal_size_latent = lg_s[1].copy()
					self.achieved_goal_latent = lg[1].copy()
					self.obstacle_latent = lo[1].copy()
					self.obstacle_size_latent = lo_s[1].copy()
				else:
					lg, lo, lo_s = latents_from_images(np.array([desired_goal_image, achieved_goal_image]), self.args)
					self.desired_goal_latent = lg[0].copy()
					self.achieved_goal_latent = lg[1].copy()
					self.obstacle_latent = lo[1].copy()
					self.obstacle_size_latent = lo_s[1].copy()

			else:
				self.env.env._move_object(position=obs['desired_goal'].copy())
				desired_goal_image = take_goal_image(self, self.args.img_size)
				self.env.env._move_object(position=obs['achieved_goal'].copy())
				achieved_goal_image = take_goal_image(self, self.args.img_size)
				latents = goal_latent_from_images(np.array([desired_goal_image, achieved_goal_image]), self.args)
				self.desired_goal_latent = latents[0].copy()
				self.achieved_goal_latent = latents[1].copy()
				#self.achieved_goal_image = achieved_goal_image.copy()

				obstacle_image = take_obstacle_image(self, self.args.img_size)
				latents_obstacle, latents_o_size = obstacle_latent_from_images(np.array([obstacle_image]), self.args)
				self.obstacle_latent = latents_obstacle[0].copy()
				self.obstacle_size_latent = latents_o_size[0].copy()

		self.rewards = 0.0

	def reset(self):
		self.env.reset()
		self.reset_ep()
		self.last_obs = self.get_obs().copy()
		return self.last_obs.copy()

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
		if hasattr(self.args, 'vae_dist_help') and self.args.vae_dist_help:
			obs = self.env.env._get_obs()
			if self.args.vae_type == 'monet' or self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
				self.env.env._move_object(position=value.copy())
				desired_goal_image = take_image_objects(self, self.args.img_size)

				if self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
					lg, lg_s, lo, lo_s = latents_from_images(np.array([desired_goal_image]), self.args)
					'''try:
						assert lg[0][0] != 100.
					except:
						im = Image.fromarray(desired_goal_image.copy().astype(np.uint8))
						im.save('failed_on_this.png')
						print('failed!!!')
						print("these are the coordinates: {}".format(value))
						object_qpos = self.env.env.sim.data.get_joint_qpos('object0:joint')
						assert object_qpos.shape == (7,)
						print('q_pos: {}'.format(object_qpos))
						v_rgba = self.env.env._get_visibility_rgba('object0')
						print('visibility rgba is: {}'.format(v_rgba))

						im = Image.fromarray(take_image_objects(self, self.args.img_size).copy().astype(np.uint8))
						im.save('failed_on_this_take2.png')'''
					self.desired_goal_size_latent = lg_s[0].copy()
				else:
					lg, lo, lo_s = latents_from_images(np.array([desired_goal_image]), self.args)
				#reset agent to orginal position
				self.env.env._move_object(position=obs['achieved_goal'].copy())
				#store latent in variable
				self.desired_goal_latent = lg[0].copy()
			else:
				self.env.env._move_object(position=value.copy())
				desired_goal_image = take_goal_image(self, self.args.img_size)
				self.env.env._move_object(position=obs['achieved_goal'].copy())
				latents = goal_latent_from_images(np.array([desired_goal_image]), self.args)
				self.desired_goal_latent = latents[0].copy()

	@property
	def target_goal_center(self):
		v = self.env.env.target_goal_center
		if v is None:
			return None
		else:
			return v.copy()

	@target_goal_center.setter
	def target_goal_center(self, value):
		if value is not None:
			self.env.env.target_goal_center = value.copy()
		else:
			self.env.env.target_goal_center = None

	@property
	def object_center(self):
		v = self.env.env.object_center
		if v is None:
			return None
		else:
			return v.copy()


	@object_center.setter
	def object_center(self, value):
		if value is not None:
			self.env.env.object_center = value.copy()
		else:
			self.env.env.object_center = None

	@property
	def obj_range(self):
		return copy.deepcopy(self.env.env.obj_range)
	@obj_range.setter
	def obj_range(self, value):
		self.env.env.obj_range = copy.deepcopy(value)

	@property
	def target_range(self):
		return copy.deepcopy(self.env.env.target_range)

	@target_range.setter
	def target_range(self, value):
		self.env.env.target_range = copy.deepcopy(value)

	@property
	def target_offset(self):
		return copy.deepcopy(self.env.env.target_offset)

	@target_offset.setter
	def target_offset(self, value):
		self.env.env.target_offset = copy.deepcopy(value)
