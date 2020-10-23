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

def goal_concat(obs, goal):
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


	def save_checkpoint(self, filename, epoch):
		if not filename.endswith(str(epoch)):
			filename = filename + '_' + str(epoch)
		save_dict = {
			'buffer':self.buffer,
			'steps': self.steps,
			'length': self.length,
			'counter': self.counter,
			'steps_counter': self.steps_counter,
		}
		'''if self.energy=='energy':
			save_dict['energy_sum'] = self.energy_sum
			save_dict['energy_offset'] = self.energy_offset
			save_dict['energy_max'] = self.energy_max
			save_dict['buffer_energy'] = self.buffer_energy
			save_dict['buffer_energy_sum'] = self.buffer_energy_sum'''
		path = self.args.dirpath + 'replaybuf/' + filename
		with open('{}.pkl'.format(path), 'wb') as f:
			pickle.dump(save_dict, f)
	'''def make_folder_if_not(folderpath):
		try:
			os.makedirs(folderpath)
		except FileExistsError:
			pass
		except:
			print("Unexpected error:", sys.exc_info()[0])
			raise
	make_folder_if_not(path+'/steps')
	with open(path+'/steps/steps_file', 'wb') as f:
		pickle.dump(self.steps, f)
	make_folder_if_not(path + '/steps')
	for k in self.buffer.keys():
		pass
	path = path
	make_folder_if_not(path)
	make_folder_if_not('{}/buffer/'.format(path))
	d = {}
	for k in self.buffer.keys():
		d[k] = []
	with open('{}/buffer/{}.pkl'.format(path, 'buffer_struct'), 'wb') as f:
		pickle.dump(d, f)
	del d
	for k in self.buffer.keys():
		make_folder_if_not('{}/buffer/{}'.format(path, k))
		for i, el in enumerate(self.buffer[k]):
			with open('{}/buffer/{}/{}_el.pkl'.format(path, k, i), 'wb') as f:
				pickle.dump(el, f)'''

	#@profile(precision=4)
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
			'''if self.energy == 'energy':
				self.energy_sum = save_dict['energy_sum']
				self.energy_offset = save_dict['energy_offset']
				self.energy_max  = save_dict['energy_max']
				self.buffer_energy = save_dict['buffer_energy']
				self.buffer_energy_sum = save_dict['buffer_energy_sum']
		with open('{}/buffer/{}.pkl'.format(path, 'buffer_struct'), 'rb') as f:
			d = pickle.load(f)
			for k in d.keys():
				self.buffer[k] = []
			del d
		for k in self.buffer.keys():
			list_files = os.listdir('{}/buffer/{}/'.format(path, k))
			list_files.sort()
			for fname in enumerate(list_files):
				with open('{}/buffer/{}/{}'.format(path, k, fname), 'rb') as f:
					el = pickle.load(f)
					self.buffer[k].append(el)'''

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
					if self.args.transform_dense:
						if 'desired' in used_goal_key:
							goal_latent = self.buffer['obs'][idx][used_goal_step]['desired_goal_latent']
						else:
							goal_latent = self.buffer['obs'][idx][used_goal_step]['achieved_goal_latent']
						rew = -self.args.compute_reward_dense(
							self.buffer['obs'][idx][step + 1]['obstacle_latent'].copy(),
							self.buffer['obs'][idx][step + 1]['obstacle_size_latent'].copy(),
							self.buffer['obs'][idx][step + 1]['achieved_goal_latent'].copy(),
							goal_latent, range_x=[-1., 1.], range_y=[-1., 1.])
						if not np.isscalar(rew):
							rew = rew[0]
					else:
						rew = self.args.compute_reward((achieved, achieved_old), goal)
				done = self.buffer['done'][idx][step]

				batch['obs'].append(copy.deepcopy(obs))
				batch['obs_next'].append(copy.deepcopy(obs_next))
				batch['acts'].append(copy.deepcopy(act))
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

	'''def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=False,using_goal_as_field=False):
		assert int(normalizer) + int(plain) <= 1
		if batch_size==-1: batch_size = self.args.batch_size
		if using_goal_as_field:
			batch = dict(obs=[], obs_next=[], goals=[], acts=[], rews=[], done=[])
		else:
			batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])
		if self.args.goal_type == 'latent':
			goal_key = 'goal_latent'
			goal_her_key = 'state_latent'
		elif self.args.goal_type == 'goal_space':
			goal_key = 'desired_goal'
			goal_her_key = 'achieved_goal'
		elif self.args.goal_type == 'state':
			goal_key = 'goal_state'
			goal_her_key = 'observation'
		elif self.args.goal_type == 'concat':
			raise Exception('not implemented yet')
		else:
			raise Exception('goal obs type not valid')
		if self.args.observation_type == 'latent':
			state_key = 'state_latent'
		elif self.args.observation_type == 'real':
			state_key = 'observation'
		elif self.args.observation_type == 'concat':
			raise Exception('not implemented yet')
		else:
			raise Exception('observation type not valid')

		for i in range(batch_size):
			if self.energy:
				idx = self.energy_sample()
			else:
				idx = np.random.randint(self.length)
			step = np.random.randint(self.steps[idx])

			if self.args.goal_based:
				if plain:
					# no additional tricks
					goal = self.buffer['obs'][idx][step][goal_key]
					goal_gspace = self.buffer['obs'][idx][step]['desired_goal']
				elif normalizer:
					# uniform sampling for normalizer update
					goal = self.buffer['obs'][idx][step][goal_her_key]#her key is actually achieved state key
					goal_gspace = self.buffer['obs'][idx][step]['achieved_goal']
				else:
					# upsampling by HER trick
					if (self.args.her!='none') and (np.random.uniform()<=self.args.her_ratio):
						if self.args.her=='match':
							goal = self.args.goal_sampler.sample()
							goal_pool = np.array([obs['achieved_goal'] for obs in self.buffer['obs'][idx][step+1:]])
							step_her = (step+1) + np.argmin(np.sum(np.square(goal_pool-goal),axis=1))
							goal = self.buffer['obs'][idx][step_her][goal_her_key]
							goal_gspace = self.buffer['obs'][idx][step_her]['desired_goal']
						else:
							steps_her = {
								'final': [self.steps[idx]],
								'future': np.random.randint(step+1, self.steps[idx]+1, size=10)
							}[self.args.her]
							for i, step_her in enumerate(
									steps_her):  # verify that hindsight goal is visible and object is not below table
								hindsight_state_dict = self.buffer['obs'][idx][step_her]
								max_behind = min(4, step_her - step)
								prev_visibles = all([self.buffer['obs'][idx][step_her - t]['gripper_visible']
													 for t in range(max_behind)])
								if prev_visibles and not hindsight_state_dict['object_below_table']:
									goal = self.buffer['obs'][idx][step_her][goal_her_key].copy()
									goal_gspace = self.buffer['obs'][idx][step_her]['achieved_goal'].copy()
									break
								if i == len(steps_her) - 1:
									# not use hindsight goal, but real one
									goal = self.buffer['obs'][idx][step][goal_key]
									goal_gspace = self.buffer['obs'][idx][step]['desired_goal']
					else:
						goal = self.buffer['obs'][idx][step][goal_key]
						goal_gspace = self.buffer['obs'][idx][step]['desired_goal']

				achieved = self.buffer['obs'][idx][step+1]['achieved_goal']
				achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
				if using_goal_as_field:
					obs = self.buffer['obs'][idx][step][state_key]
					obs_next = self.buffer['obs'][idx][step + 1][state_key]
					batch['obs'].append(copy.deepcopy(obs))
					batch['obs_next'].append(copy.deepcopy(obs_next))
					batch['goals'].append(copy.deepcopy(goal))
				else:
					obs = goal_concat(self.buffer['obs'][idx][step][state_key], goal)
					obs_next = goal_concat(self.buffer['obs'][idx][step+1][state_key], goal)
					batch['obs'].append(copy.deepcopy(obs))
					batch['obs_next'].append(copy.deepcopy(obs_next))
				act = self.buffer['acts'][idx][step]
				rew = self.args.compute_reward((achieved, achieved_old), goal_gspace)
				#done = self.buffer['done'][idx][step]
				done = [not (rew == -1.0)]#todo see if affect hgg, probably not
				batch['acts'].append(copy.deepcopy(act))
				batch['rews'].append(copy.deepcopy([rew]))
				batch['done'].append(copy.deepcopy(done))
			else:
				for key in ['obs', 'acts', 'rews', 'done']:
					if key=='obs':
						batch['obs'].append(copy.deepcopy(self.buffer[key][idx][step]))
						batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][step+1]))
					else:
						batch[key].append(copy.deepcopy(self.buffer[key][idx][step]))

		return batch

	def sample_for_distance(self, last_eps, goal_key, goal_her_key, state_key, batch_size=-1):
		if batch_size==-1: batch_size = self.args.batch_size
		batch = dict(obs=[], obs_next=[], goals=[], acts=[], dis=[], done=[])
		possible_idx = [(self.counter-1-i) % self.args.buffer_size for i in range(last_eps)]

		for i in range(batch_size):
			idx = np.random.randint(len(possible_idx))
			idx = possible_idx[idx]
			step = np.random.randint(self.steps[idx])
			# upsampling by HER trick
			if (self.args.her!='none') and (np.random.uniform()<=self.args.her_ratio):
				steps_her = {
					'final': [self.steps[idx]],
					'future': np.random.randint(step+1, self.steps[idx]+1, size=10)
				}[self.args.her]
				for i, step_her in enumerate(steps_her):#verify that hindsight goal is visible and object is not below table
					hindsight_state_dict = self.buffer['obs'][idx][step_her]
					max_behind = min(4, step_her-step)
					prev_visibles = all([self.buffer['obs'][idx][step_her - t]['gripper_visible']
										 for t in range(max_behind)])
					if prev_visibles and not hindsight_state_dict['object_below_table']:
						goal = self.buffer['obs'][idx][step_her][goal_her_key].copy()
						goal_gspace = self.buffer['obs'][idx][step_her]['achieved_goal'].copy()
						break
					if i == len(steps_her)-1:
						#not use hindsight goal, but real one
						goal = self.buffer['obs'][idx][step][goal_key]
						goal_gspace = self.buffer['obs'][idx][step]['desired_goal']

				# todo delete this condition, just for one case
				#goal[1] = 0.0
				#if not (goal_gspace[0] > self.buffer['obs'][idx][step_her]['init_pos'] + 0.3 or
				#		goal_gspace[0] < self.buffer['obs'][idx][step_her]['init_pos'] - 0.3):
				#	goal = self.buffer['obs'][idx][step][goal_key]
				#	goal_gspace = self.buffer['obs'][idx][step]['desired_goal']
			else:
				goal = self.buffer['obs'][idx][step][goal_key]
				goal_gspace = self.buffer['obs'][idx][step]['desired_goal']

			achieved = self.buffer['obs'][idx][step+1]['achieved_goal']
			achieved_old = self.buffer['obs'][idx][step]['achieved_goal']


			obs = self.buffer['obs'][idx][step][state_key]
			obs_next = self.buffer['obs'][idx][step + 1][state_key]
			batch['obs'].append(copy.deepcopy(obs))
			batch['obs_next'].append(copy.deepcopy(obs_next))
			batch['goals'].append(copy.deepcopy(goal))
			act = self.buffer['acts'][idx][step]
			rew = self.args.compute_reward((achieved, achieved_old), goal_gspace)
			#done = self.buffer['done'][idx][step]
			done = [not (rew == -1.0)]
			dis = 1.0 if not done[0] else 0.0

			batch['acts'].append(copy.deepcopy(act))
			batch['dis'].append(copy.deepcopy([dis]))
			batch['done'].append(copy.deepcopy(done))

		return batch'''
