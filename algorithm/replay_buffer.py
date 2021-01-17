import numpy as np
import copy
from envs.utils import quaternion_to_euler_angle
import pickle
import hickle
import os
import sys
from memory_profiler import profile
import gc
import psutil


def goal_concat(obs, goal):#todo will this require that the observation is modified coorespondingly if goal is expanded
	return np.concatenate([obs, goal], axis=0)


def goal_based_process(obs):
	return goal_concat(obs['observation'], obs['desired_goal'])


class Trajectory:
	def __init__(self, init_obs):
		self.ep = {
			'obs': [copy.deepcopy(init_obs)],
			'rews': [],
			'acts': [],
			'done': []
		}
		self.length = 0

	def store_step(self, action, obs, reward, done):
		self.ep['acts'].append(copy.deepcopy(action))
		self.ep['obs'].append(copy.deepcopy(obs))
		self.ep['rews'].append(copy.deepcopy([reward]))
		self.ep['done'].append(copy.deepcopy([np.float32(done)]))
		self.length += 1

	def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
		# from "Energy-Based Hindsight Experience Prioritization"
		if env_id[:5]=='Fetch':
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['achieved_goal'])
			obj = np.array([obj])

			clip_energy = 0.5
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0#this could be prblematic if object falls, even not necessay if always ata table
			g, m, delta_t = 9.81, 1, 0.04
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)
		else:
			assert env_id[:4]=='Hand'
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['observation'][-7:])
			obj = np.array([obj])

			clip_energy = 2.5
			g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
			quaternion = obj[:,:,3:].copy()
			angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
			diff_angle = np.diff(angle, axis=1)
			angular_velocity = diff_angle / delta_t
			rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
			rotational_energy = np.sum(rotational_energy, axis=2)
			obj = obj[:,:,:3]
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy + w_rotational*rotational_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)


class ReplayBuffer_Episodic:
	def __init__(self, args):
		self.args = args
		if args.buffer_type=='energy':
			self.energy = True
			self.energy_sum = 0.0
			self.energy_offset = 0.0
			self.energy_max = 1.0
		else:
			self.energy = False
		self.buffer = {}
		self.steps = []
		self.length = 0
		self.counter = 0
		self.steps_counter = 0
		self.sample_methods = {
			'ddpg': self.sample_batch_ddpg,
			'sac': self.sample_batch_ddpg #use same sample mechanism
		}
		if hasattr(args, 'alg'):
			self.sample_batch = self.sample_methods[args.alg]
		else:
			self.sample_batch = self.sample_methods['ddpg']

	def load_checkpoint(self,filename):
		if filename.endswith('.pkl'):
			filename.replace('.pkl','')
		path = self.args.dirpath + 'replaybuf/' + filename
		with open('{}.pkl'.format(path), 'rb') as f:
			save_dict = pickle.load(f)
			self.buffer = save_dict['buffer']
			self.steps = save_dict['steps']
			self.length = save_dict['length']
			self.counter = save_dict['counter']
			self.steps_counter = save_dict['steps_counter']
			del save_dict

	def store_trajectory(self, trajectory):
		episode = trajectory.ep
		if self.energy:
			energy = trajectory.energy(self.args.env)
			self.energy_sum += energy
		if self.counter==0:
			for key in episode.keys():
				self.buffer[key] = []
			if self.energy:
				self.buffer_energy = []
				self.buffer_energy_sum = []
		if self.counter<self.args.buffer_size:
			for key in self.buffer.keys():
				self.buffer[key].append(episode[key])
			if self.energy:
				self.buffer_energy.append(copy.deepcopy(energy))
				self.buffer_energy_sum.append(copy.deepcopy(self.energy_sum))
			self.length += 1
			self.steps.append(trajectory.length)
		else:
			idx = self.counter%self.args.buffer_size
			for key in self.buffer.keys():
				self.buffer[key][idx] = episode[key]
			if self.energy:
				self.energy_offset = copy.deepcopy(self.buffer_energy_sum[idx])
				self.buffer_energy[idx] = copy.deepcopy(energy)
				self.buffer_energy_sum[idx] = copy.deepcopy(self.energy_sum)
			self.steps[idx] = trajectory.length
		self.counter += 1
		self.steps_counter += trajectory.length

	def energy_sample(self):
		t = self.energy_offset + np.random.uniform(0,1)*(self.energy_sum-self.energy_offset)
		if self.counter>self.args.buffer_size:
			if self.buffer_energy_sum[-1]>=t:
				return self.energy_search(t, self.counter%self.length, self.length-1)
			else:
				return self.energy_search(t, 0, self.counter%self.length-1)
		else:
			return self.energy_search(t, 0, self.length-1)

	def energy_search(self, t, l, r):
		if l==r: return l
		mid = (l+r)//2
		if self.buffer_energy_sum[mid]>=t:
			return self.energy_search(t, l, mid)
		else:
			return self.energy_search(t, mid+1, r)

	def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=False):
		assert int(normalizer) + int(plain) <= 1
		if batch_size == -1: batch_size = self.args.batch_size
		batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

		for i in range(batch_size):
			if self.energy:
				idx = self.energy_sample()
			else:
				idx = np.random.randint(self.length)
			step = np.random.randint(self.steps[idx])

			if self.args.goal_based:
				if plain:
					# no additional tricks
					goal = self.buffer['obs'][idx][step]['desired_goal']
					used_goal_key = 'desired_goal'
					used_goal_step = step
				elif normalizer:
					# uniform sampling for normalizer update
					goal = self.buffer['obs'][idx][step]['achieved_goal']
					used_goal_key = 'achieved_goal'
					used_goal_step = step
				else:
					# upsampling by HER trick
					if (self.args.her != 'none') and (np.random.uniform() <= self.args.her_ratio):
						if self.args.her == 'match':
							goal = self.args.goal_sampler.sample()
							goal_pool = np.array([obs['achieved_goal'] for obs in self.buffer['obs'][idx][step + 1:]])
							step_her = (step + 1) + np.argmin(np.sum(np.square(goal_pool - goal), axis=1))
							goal = self.buffer['obs'][idx][step_her]['achieved_goal']
							used_goal_key = 'achieved_goal'
							used_goal_step = step_her
						else:
							step_her = {
								'final': self.steps[idx],
								'future': np.random.randint(step + 1, self.steps[idx] + 1)
							}[self.args.her]
							goal = self.buffer['obs'][idx][step_her]['achieved_goal']
							used_goal_key = 'achieved_goal'
							used_goal_step = step_her
					else:
						goal = self.buffer['obs'][idx][step]['desired_goal']
						used_goal_key = 'desired_goal'
						used_goal_step = step

				achieved = self.buffer['obs'][idx][step + 1]['achieved_goal']
				achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
				obs = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
				obs_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], goal)
				act = self.buffer['acts'][idx][step]
				if plain:
					rew = self.buffer['rews'][idx][step][0]
				else:
					rew = self.args.compute_reward(self.buffer['obs'][idx][step + 1], self.buffer['obs'][idx][step], goal)
				done = self.buffer['done'][idx][step]#todo is this done correct?

				batch['obs'].append(copy.deepcopy(obs))
				batch['obs_next'].append(copy.deepcopy(obs_next))
				batch['acts'].append(copy.deepcopy(act))
				if isinstance(rew, np.ndarray) and len(rew.shape) > 0:
					batch['rews'].append(copy.deepcopy(rew))
				else:
					batch['rews'].append(copy.deepcopy([rew]))
				batch['done'].append(copy.deepcopy(done))
			else:
				for key in ['obs', 'acts', 'rews', 'done']:
					if key == 'obs':
						batch['obs'].append(copy.deepcopy(self.buffer[key][idx][step]))
						batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][step + 1]))
					else:
						batch[key].append(copy.deepcopy(self.buffer[key][idx][step]))

		return batch


class ReplayBuffer_Imaginary:
	def __init__(self, args, buffer_size):
		self.args = args
		self.buffer = {}
		self.steps = []
		self.length = 0
		self.counter = 0
		self.steps_counter = 0
		self.buffer_size = buffer_size


	def store_im_info(self, im_info, env):
		index, imaginary_info_dict = im_info
		obs_t0 = imaginary_info_dict['obs'][0]
		obs_t1 = imaginary_info_dict['obs'][1]


		if self.counter==0:
			for key in imaginary_info_dict.keys():
				self.buffer[key] = []

		'''if self.args.goal == 'intervalPAVRewMod':
			fname0 = self.args.logger.my_log_dir + 'original_{}_t0.png'.format(self.counter)
			fname1 = self.args.logger.my_log_dir + 'original_{}_t1.png'.format(self.counter)
			env.visualize(obs_t0, fname0)
			env.visualize(obs_t1, fname1)'''

		bbox_obstacle_t0 = obs_t0['obstacle_st_t'][index]
		bbox_obstacle_t1 = obs_t1['obstacle_st_t'][index]
		bbox_elem_t0 = obs_t0['goal_st_t']
		bbox_elem_t1 = obs_t1['goal_st_t']

		new_bboxes_t0, new_bboxes_t1, extra_info = create_new_interactions(
			bbox_obstacle_t0, bbox_obstacle_t1, bbox_elem_t0, bbox_elem_t1,
			n_per_type=self.args.im_n_per_type
		)
		episodes = []
		for i in range(len(new_bboxes_t0)):
			new_list_bboxes_t0 = obs_t0['obstacle_st_t'].copy()
			new_list_bboxes_t0[index] = new_bboxes_t0[i]
			new_list_bboxes_t1 = obs_t1['obstacle_st_t'].copy()
			new_list_bboxes_t1[index] = new_bboxes_t1[i]

			new_obst0 = env._modify_obs(obs_t0, new_list_bboxes_t0, extra_info[i], index)
			new_obst1 = env._modify_obs(obs_t1, new_list_bboxes_t1, extra_info[i], index)
			'''if self.args.goal == 'intervalPAVRewMod':
				fname0 = self.args.logger.my_log_dir + 'imaginary_{}_{}_t0.png'.format(self.counter, i)
				fname1 = self.args.logger.my_log_dir + 'imaginary_{}_{}_t1.png'.format(self.counter, i)
				env.visualize(new_obst0, fname0)
				env.visualize(new_obst1, fname1)'''
			im_ep = {'obs':[new_obst0, new_obst1]}
			for key in imaginary_info_dict:
				if key == 'obs':
					pass
				else:
					im_ep[key] = imaginary_info_dict[key]
			episodes.append(im_ep)
		episodes.append(imaginary_info_dict)

		for ep in episodes:
			if self.counter < self.buffer_size:
				for key in self.buffer.keys():
					self.buffer[key].append(ep[key])
				self.length += 1
			else:
				idx = self.counter % self.buffer_size
				for key in self.buffer.keys():
					self.buffer[key][idx] = ep[key]
			self.counter += 1


	def sample_batch(self, batch_size=-1, normalizer=False, plain=False):
		assert int(normalizer) + int(plain) <= 1
		if batch_size == -1: batch_size = max(int(self.args.batch_size / 100), 100)
		batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

		for i in range(batch_size):
			idx = np.random.randint(self.length)

			if self.args.goal_based:
				if plain:
					# no additional tricks
					goal = self.buffer['obs'][idx][1]['desired_goal']
				elif normalizer:
					# uniform sampling for normalizer update
					goal = self.buffer['obs'][idx][1]['achieved_goal']
				else:
					# upsampling by HER trick
					if (self.args.her != 'none') and (np.random.uniform() <= self.args.her_ratio):
						#with this we can just do future
						goal = self.buffer['obs'][idx][1]['achieved_goal']
					else:
						goal = self.buffer['obs'][idx][1]['desired_goal']

				obs = goal_concat(self.buffer['obs'][idx][0]['observation'], goal)
				obs_next = goal_concat(self.buffer['obs'][idx][1]['observation'], goal)
				act = self.buffer['acts'][idx][0]
				rew = self.args.compute_reward(self.buffer['obs'][idx][1], self.buffer['obs'][idx][0], goal)
				done = self.buffer['done'][idx][0]#todo is this done correct?

				batch['obs'].append(copy.deepcopy(obs))
				batch['obs_next'].append(copy.deepcopy(obs_next))
				batch['acts'].append(copy.deepcopy(act))
				if isinstance(rew, np.ndarray) and len(rew.shape) > 0:
					batch['rews'].append(copy.deepcopy(rew))
				else:
					batch['rews'].append(copy.deepcopy([rew]))
				batch['done'].append(copy.deepcopy(done))
			else:
				for key in ['obs', 'acts', 'rews', 'done']:
					if key == 'obs':
						batch['obs'].append(copy.deepcopy(self.buffer[key][idx][0]))
						batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][1]))
					else:
						batch[key].append(copy.deepcopy(self.buffer[key][idx][0]))

		return batch


def check_collisions(a_bbox, b_bboxes):
	# b_min_x - a_max_x
	d1x = (b_bboxes[:, 0] - b_bboxes[:, 2]) - (a_bbox[0] + a_bbox[2])
	d1y = (b_bboxes[:, 1] - b_bboxes[:, 3]) - (a_bbox[1] + a_bbox[3])
	d2x = (a_bbox[0] - a_bbox[2]) - (b_bboxes[:, 0] + b_bboxes[:, 2])
	d2y = (a_bbox[1] - a_bbox[3]) - (b_bboxes[:, 1] + b_bboxes[:, 3])
	d1_bools = np.logical_or(d1x > 0., d1y > 0.)
	d2_bools = np.logical_or(d2x > 0., d2y > 0.)
	d_bools = np.logical_or(d1_bools, d2_bools)
	return np.logical_not(d_bools)


def create_new_interactions(bbox_obstacle_t0, bbox_obstacle_t1, bbox_elem_t0, bbox_elem_t1, n_per_type):
	ox0, oy0, oxs0, oys0 = bbox_obstacle_t0
	ox1, oy1, oxs1, oys1 = bbox_obstacle_t1
	elx0, ely0, elxs0, elys0 = bbox_elem_t0
	elx1, ely1, elxs1, elys1 = bbox_elem_t1

	direction = (bbox_obstacle_t1[0:2] - bbox_obstacle_t0[0:2])
	new_bboxes_t0 = []
	new_bboxes_t1 = []
	extra_info = []
	if np.abs(direction[1]) > np.abs(direction[0]):
		dif = np.abs(bbox_obstacle_t1[1] - bbox_obstacle_t0[1])
		# place above, go down not reach
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			# size of obstacle not considered so it is colliding and when moving more chance to not collide any
			to_move = np.abs(direction[1])
			a_ny0 = ely1 + elys1 + extra_dist + elys1 + to_move
			a_new_bbox_t0 = np.array([ox0, a_ny0, oxs0, oys0])
			a_ny1 = a_ny0 - to_move
			a_new_bbox_t1 = np.array([ox1, a_ny1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place above, go down pass
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			#size of obstacle not considered so it is colliding and when moving more chance to not collide any
			a_ny0 = ely1 + elys1 + extra_dist - elys1
			a_new_bbox_t0 = np.array([ox0, a_ny0, oxs0, oys0])
			to_move = np.abs(direction[1])
			if np.random.randint(2) == 0:
				to_move += elys1 + extra_dist
			a_ny1 = a_ny0 - to_move
			a_new_bbox_t1 = np.array([ox1, a_ny1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place above, go up
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			a_ny0 = ely1 + elys1 + oys1 + extra_dist
			a_new_bbox_t0 = np.array([ox0, a_ny0, oxs0, oys0])
			if check_collisions(a_new_bbox_t0, np.array([bbox_elem_t0]))[0]:
				a_ny0 = ely0 + elys0 + oys0 + 0.01 + extra_dist
				a_new_bbox_t0 = np.array([ox0, a_ny0, oxs0, oys0])
			a_ny1 = a_ny0 + np.abs(direction[1])
			a_new_bbox_t1 = np.array([ox1, a_ny1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place below, go down
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			b_ny0 = ely1 - elys1 - oys1 - extra_dist
			b_new_bbox_t0 = np.array([ox0, b_ny0, oxs0, oys0])
			if check_collisions(b_new_bbox_t0, np.array([bbox_elem_t0]))[0]:
				b_ny0 = ely0 - elys0 - oys0 - 0.01 - extra_dist
				b_new_bbox_t0 = np.array([ox0, b_ny0, oxs0, oys0])
			b_ny1 = b_ny0 - np.abs(direction[1])
			b_new_bbox_t1 = np.array([ox1, b_ny1, oxs1, oys1])
			new_bboxes_t0.append(b_new_bbox_t0)
			new_bboxes_t1.append(b_new_bbox_t1)
			dir = np.array(b_new_bbox_t1[0:2] - b_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place below, go up not reach
		extra_dists = np.random.uniform(low=0., high=dif , size=n_per_type)
		for extra_dist in extra_dists:
			# size of obstacle not considered so it is colliding and when moving more chance to not collide any
			to_move = np.abs(direction[1])
			a_ny0 = ely1 - elys1 - extra_dist - elys1 - to_move
			a_new_bbox_t0 = np.array([ox0, a_ny0, oxs0, oys0])
			a_ny1 = a_ny0 + to_move
			a_new_bbox_t1 = np.array([ox1, a_ny1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place below, go up pass
		extra_dists = np.random.uniform(low=0., high=dif , size=n_per_type)
		for extra_dist in extra_dists:
			# size of obstacle not considered so it is colliding and when moving more chance to not collide any
			a_ny0 = ely1 - elys1 - extra_dist + elys1
			a_new_bbox_t0 = np.array([ox0, a_ny0, oxs0, oys0])
			to_move = np.abs(direction[1])
			if np.random.randint(2) == 0:
				to_move += elys1 + extra_dist
			a_ny1 = a_ny0 + to_move
			a_new_bbox_t1 = np.array([ox1, a_ny1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

	else:
		dif = np.abs(bbox_obstacle_t1[0] - bbox_obstacle_t0[0])
		# place right, go right
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			a_nx0 = elx1 + elxs1 + oxs1 + extra_dist
			a_new_bbox_t0 = np.array([a_nx0, oy0, oxs0, oys0])
			if check_collisions(a_new_bbox_t0, np.array([bbox_elem_t0]))[0]:
				a_nx0 = elx0 + elxs0 + oxs0 + 0.01 + extra_dist
				a_new_bbox_t0 = np.array([a_nx0, oy0, oxs0, oys0])
			a_nx1 = a_nx0 + np.abs(direction[0])
			a_new_bbox_t1 = np.array([a_nx1, oy1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place right, go left no reach
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			to_move = np.abs(direction[0])
			a_nx0 = elx1 + elxs1 + extra_dist + oxs1 + to_move
			a_new_bbox_t0 = np.array([a_nx0, oy0, oxs0, oys0])
			a_nx1 = a_nx0 - to_move
			a_new_bbox_t1 = np.array([a_nx1, oy1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place right, go left pass
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			a_nx0 = elx1 + elxs1 + extra_dist - oxs1
			a_new_bbox_t0 = np.array([a_nx0, oy0, oxs0, oys0])
			to_move = np.abs(direction[0])
			if np.random.randint(2) == 0:
				to_move += elxs1
			a_nx1 = a_nx0 - to_move
			a_new_bbox_t1 = np.array([a_nx1, oy1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place left, go left
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			b_nx0 = elx1 - elxs1 - oxs1 - extra_dist
			b_new_bbox_t0 = np.array([b_nx0, oy0, oxs0, oys0])
			if check_collisions(b_new_bbox_t0, np.array([bbox_elem_t0]))[0]:
				b_nx0 = elx0 - elxs0 - oxs0 - 0.01 - extra_dist
				b_new_bbox_t0 = np.array([b_nx0, oy0, oxs0, oys0])
			b_nx1 = b_nx0 - np.abs(direction[0])
			b_new_bbox_t1 = np.array([b_nx1, oy1, oxs1, oys1])
			new_bboxes_t0.append(b_new_bbox_t0)
			new_bboxes_t1.append(b_new_bbox_t1)
			dir = np.array(b_new_bbox_t1[0:2] - b_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place left, go right no reach
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			to_move = np.abs(direction[0])
			a_nx0 = elx1 - elxs1 - extra_dist - oxs1 - to_move
			a_new_bbox_t0 = np.array([a_nx0, oy0, oxs0, oys0])
			a_nx1 = a_nx0 + to_move
			a_new_bbox_t1 = np.array([a_nx1, oy1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})

		# place left, go right pass
		extra_dists = np.random.uniform(low=0., high=dif, size=n_per_type)
		for extra_dist in extra_dists:
			a_nx0 = elx1 - elxs1 - extra_dist + oxs1
			a_new_bbox_t0 = np.array([a_nx0, oy0, oxs0, oys0])
			to_move = np.abs(direction[0])
			if np.random.randint(2) == 0:
				to_move += elxs1
			a_nx1 = a_nx0 + to_move
			a_new_bbox_t1 = np.array([a_nx1, oy1, oxs1, oys1])
			new_bboxes_t0.append(a_new_bbox_t0)
			new_bboxes_t1.append(a_new_bbox_t1)
			dir = np.array(a_new_bbox_t1[0:2] - a_new_bbox_t0[0:2])
			extra_info.append({'dir_not_scaled': dir.copy()})


	return new_bboxes_t0, new_bboxes_t1, extra_info