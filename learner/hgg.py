import copy
import numpy as np
from envs import make_env
from envs.utils import goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double
from envs.distance_graph import DistanceGraph
from vae_env_inter import take_env_image


class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

class TrajectoryPool_VAEs(TrajectoryPool):
	def __init__(self, args, pool_length):
		super(TrajectoryPool_VAEs, self).__init__(args, pool_length)
		self.latent_goals = []
		self.latent_goals_init = []
		self.latent_obstacles = []
		self.latent_obstacles_sizes = []


	def insert(self, trajectory, init_state, achieved_trajectory_goals_latents,
			   achieved_trajectory_init_goals_latents, achieved_trajectory_obstacle_latents,
			   achieved_trajectory_obstacle_latents_sizes):
		super(TrajectoryPool_VAEs, self).insert(trajectory, init_state)
		self.counter -= 1
		if self.counter<self.length:
			self.latent_goals.append(achieved_trajectory_goals_latents)
			self.latent_goals_init.append(achieved_trajectory_init_goals_latents)
			self.latent_obstacles.append(achieved_trajectory_obstacle_latents)
			self.latent_obstacles_sizes.append(achieved_trajectory_obstacle_latents_sizes)
		else:
			self.latent_goals[self.counter % self.length] = achieved_trajectory_goals_latents
			self.latent_goals_init[self.counter % self.length] = achieved_trajectory_init_goals_latents
			self.latent_obstacles[self.counter % self.length] = achieved_trajectory_obstacle_latents
			self.latent_obstacles_sizes[self.counter % self.length] = achieved_trajectory_obstacle_latents_sizes
		self.counter += 1

	def pad(self):
		copy_pool, copy_pool_init_state = super(TrajectoryPool_VAEs, self).pad()
		if self.counter>=self.length:
			return copy_pool, copy_pool_init_state, copy.deepcopy(self.latent_goals), \
				   copy.deepcopy(self.latent_goals_init), copy.deepcopy(self.latent_obstacles), copy.deepcopy(self.latent_obstacles_sizes)
		pool_latent_goals = copy.deepcopy(self.latent_goals)
		pool_latent_goals_init = copy.deepcopy(self.latent_goals_init)
		pool_latent_obstacles = copy.deepcopy(self.latent_obstacles)
		pool_latent_obstacles_sizes = copy.deepcopy(self.latent_obstacles_sizes)
		while len(pool_latent_goals) < self.length:
			pool_latent_goals += copy.deepcopy(self.latent_goals)
			pool_latent_goals_init += copy.deepcopy(self.latent_goals_init)
			pool_latent_obstacles += copy.deepcopy(self.latent_obstacles)
			pool_latent_obstacles_sizes += copy.deepcopy(self.latent_obstacles_sizes)
		return copy_pool, copy_pool_init_state, copy.deepcopy(pool_latent_goals[:self.length]), \
			   copy.deepcopy(pool_latent_goals_init[:self.length]), copy.deepcopy(pool_latent_obstacles[:self.length]), \
			   copy.deepcopy(pool_latent_obstacles_sizes[:self.length])

class MatchSampler:
	def __init__(self, args, achieved_trajectory_pool):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)
		self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
		self.delta = self.env.distance_threshold

		self.length = args.episodes
		init_goal = self.env.reset()['achieved_goal'].copy()
		self.pool = np.tile(init_goal[np.newaxis,:], [self.length,1])+np.random.normal(0,self.delta,size=(self.length,self.dim))
		self.init_state = self.env.reset()['observation'].copy()

		self.match_lib = gcc_load_lib('learner/cost_flow.c')
		self.achieved_trajectory_pool = achieved_trajectory_pool

		if self.args.graph:
			self.create_graph_distance()

		# estimating diameter
		self.max_dis = 0
		for i in range(1000):
			obs = self.env.reset()
			dis = self.evaluate_distance_start(obs)
			if dis>self.max_dis: self.max_dis = dis

	def evaluate_distance_start(self, obs):
		if self.args.dist_estimator_type is not None:
			if self.args.dist_estimator_type in ['noneTypeReal','realCoords','multipleReal', 'substReal']:
				g = obs['desired_goal']
				current = obs['achieved_goal']
			else:
				g = obs['desired_goal_latent'].copy()
				current = obs['achieved_goal_latent'].copy()

			d = self.args.dist_estimator.calculate_distance_batch(goal_pos=g,
																  current_pos_batch=np.array([current])
																  )[0]
			return d
		else:
			return self.get_graph_goal_distance(obs['achieved_goal'], obs['desired_goal'])


	# Pre-computation of graph-based distances
	def create_graph_distance(self):
		obstacles = list()
		field = self.env.env.env.adapt_dict["field"]
		obstacles = self.env.env.env.adapt_dict["obstacles"]
		num_vertices = self.args.num_vertices
		## just to make temporal prove delete then
		'''field_2D = [field[0], field[1], field[3], field[4]]  #
		num_vertices_2D = [self.args.num_vertices[0], self.args.num_vertices[1]]

		obstacles_2D = []
		for o in obstacles:
			obstacles_2D.append([o[0], o[1], o[3], o[4]])
			
		graph = DistanceGraph2D(args=self.args, field=field_2D, num_vertices=num_vertices_2D, obstacles=obstacles_2D)'''
		
		##
		graph = DistanceGraph(args=self.args, field=field, num_vertices=num_vertices, obstacles=obstacles)
		graph.compute_cs_graph()
		graph.compute_dist_matrix()
		self.graph = graph

	def get_graph_goal_distance(self, goal_a, goal_b):
		if self.args.graph:
			d, _ = self.graph.get_dist(goal_a, goal_b)
			if d == np.inf:
				d = 9999
			return d
		else:
			return 	np.linalg.norm(goal_a - goal_b, ord=2)

	def add_noise(self, pre_goal, noise_std=None):
		goal = pre_goal.copy()
		dim = 2 if self.args.env[:5]=='Fetch' else self.dim
		if noise_std is None: noise_std = self.delta
		goal[:dim] += np.random.normal(0, noise_std, size=dim)
		return goal.copy()

	def sample(self, idx):
		if self.args.env[:5]=='Fetch':
			return self.add_noise(self.pool[idx])
		else:
			return self.pool[idx].copy()

	def find(self, goal):
		res = np.sqrt(np.sum(np.square(self.pool-goal),axis=1))
		idx = np.argmin(res)
		if test_pool:
			self.args.logger.add_record('Distance/sampler', res[idx])
		return self.pool[idx].copy()

	def update(self, initial_goals, desired_goals, initial_goals_latents=None, desired_goals_latents=None):
		if self.achieved_trajectory_pool.counter==0:
			self.pool = copy.deepcopy(desired_goals)
			return
		if self.args.vae_dist_help:
			assert initial_goals_latents is not None and desired_goals_latents is not None

		if self.args.vae_dist_help:
			achieved_pool, achieved_pool_init_state, achieved_latent_goals, achieved_latent_goals_init, \
			achieved_latent_obstacles, achieved_latent_obstacles_sizes = self.achieved_trajectory_pool.pad()
		else:
			achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
		candidate_goals = []
		candidate_edges = []
		candidate_id = []

		agent = self.args.agent
		achieved_value = []
		for i in range(len(achieved_pool)):
			obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in range(achieved_pool[i].shape[0])]#obervation from first state of trajectoy to goal, which is the achieved state in the trajectory
			value = agent.get_q_pi(obs)
			achieved_value.append(value.copy())

		n = 0
		graph_id = {'achieved':[],'desired':[]}
		for i in range(len(achieved_pool)):
			n += 1
			graph_id['achieved'].append(n)
		for i in range(len(desired_goals)):
			n += 1
			graph_id['desired'].append(n)
		n += 1
		self.match_lib.clear(n)

		for i in range(len(achieved_pool)):
			self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
		for i in range(len(achieved_pool)):
			if self.args.vae_dist_help or self.args.dist_estimator_type is not None:
				i1 = achieved_pool[i][:, 0] > self.args.real_field_center[0] + self.args.real_field_size[0]
				i2 = achieved_pool[i][:, 0] < self.args.real_field_center[0] - self.args.real_field_size[0]
				i3 = achieved_pool[i][:, 1] > self.args.real_field_center[1] + self.args.real_field_size[1]
				i4 = achieved_pool[i][:, 1] < self.args.real_field_center[1] - self.args.real_field_size[1]
				indices_outside = np.logical_or(np.logical_or(i1, i2), np.logical_or(i3, i4))
			for j in range(len(desired_goals)):
				if self.args.graph:
					size = achieved_pool[i].shape[0]
					res_1 = np.zeros(size)
					for k in range(size):
						res_1[k] = self.get_graph_goal_distance(achieved_pool[i][k], desired_goals[j])
					res = res_1 - achieved_value[i]/(self.args.hgg_L/self.max_dis/(1-self.args.gamma))
				elif self.args.vae_dist_help:
					distances = np.zeros(shape=len(achieved_latent_goals[i]))
					indices_inside = np.logical_not(indices_outside)
					if self.args.dist_estimator_type is not None:
						latent_distances = \
						self.args.dist_estimator.calculate_distance_batch(goal_pos=desired_goals_latents[j].copy(),
																		  current_pos_batch=achieved_latent_goals[i][indices_inside].copy())

					distances[indices_inside] = latent_distances.copy()
					distances[indices_outside] = 9999.

					res = distances - achieved_value[i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
				elif self.args.dist_estimator_type in ['realCoords','multipleReal','substReal']:
					distances = np.zeros(shape=len(achieved_pool[i]))
					indices_inside = np.logical_not(indices_outside)
					ds = self.args.dist_estimator.calculate_distance_batch(goal_pos=desired_goals[j].copy(),
																		  current_pos_batch=achieved_pool[i][
																			  indices_inside].copy())


					distances[indices_inside] = ds.copy()
					distances[indices_outside] = 100.0

					res = distances - achieved_value[i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
				else:
					#(2.22) || g^ ^i - m(s_t^i) || - (1/L) V^(pi)(s_0^i || m(s_t^i))
					res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=1)) - achieved_value[i]/(self.args.hgg_L / self.max_dis / (1 - self.args.gamma))  # Todo: that was original

				#(2.22) c * || m(s^_0 ^i) - m(s_0 ^i) || + min_t(..)
				if self.args.vae_dist_help:
					if self.args.dist_estimator_type is not None:
						d_i = self.args.dist_estimator.calculate_distance_batch(
							goal_pos=desired_goals_latents[j].copy(),
							current_pos_batch=np.array([achieved_latent_goals[i][0].copy()])
						)[0]
					match_dis = np.min(res) + d_i * self.args.hgg_c
				elif self.args.dist_estimator_type in ['noneTypeReal','realCoords','multipleReal','substReal']:
						d_i = self.args.dist_estimator.calculate_distance_batch(
							goal_pos=desired_goals[j].copy(),
							current_pos_batch=np.array([achieved_pool[i][0].copy()])
						)[0]
						match_dis = np.min(res) + d_i * self.args.hgg_c
				else:
					match_dis = np.min(res)+goal_distance(achieved_pool[i][0], initial_goals[j])*self.args.hgg_c # TODO: distance of initial positions: take l2 norm_as before
				match_idx = np.argmin(res)

				edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
				candidate_goals.append(achieved_pool[i][match_idx])
				candidate_edges.append(edge)
				candidate_id.append(j)
		for i in range(len(desired_goals)):
			self.match_lib.add(graph_id['desired'][i], n, 1, 0)

		match_count = self.match_lib.cost_flow(0,n)#this library chooses for each j of our distributions the trajectory thath minimizes w
		assert match_count==self.length

		explore_goals = [0]*self.length
		for i in range(len(candidate_goals)):
			if self.match_lib.check_match(candidate_edges[i])==1:
				explore_goals[candidate_id[i]] = candidate_goals[i].copy()
		assert len(explore_goals)==self.length
		self.pool = np.array(explore_goals)


class HGGLearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)

		self.env_List = []
		self.goal_env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))
			self.goal_env_List.append(make_env(args))

		if args.vae_dist_help:
			self.achieved_trajectory_pool = TrajectoryPool_VAEs(args, args.hgg_pool_size)
		else:
			self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
		self.sampler = MatchSampler(args, self.achieved_trajectory_pool)

		self.stop_hgg_threshold = self.args.stop_hgg_threshold
		self.stop = False
		self.learn_calls = 0
		self.success_n = 0

	def learn(self, args, env, env_test, agent, buffer, write_goals=0, epoch=None, cycle=None):
		# Actual learning cycle takes place here!
		initial_goals = []
		desired_goals = []
		goal_list = []
		if args.vae_dist_help:
			initial_goals_latents = []
			desired_goals_latents = []

		# get initial position and goal from environment for each episode
		for i in range(args.episodes):
			obs = self.env_List[i].reset()
			goal_a = obs['achieved_goal'].copy()
			goal_d = obs['desired_goal'].copy()
			initial_goals.append(goal_a.copy())
			desired_goals.append(goal_d.copy())
			if args.vae_dist_help:
				initial_goals_latents.append(obs['achieved_goal_latent'].copy())
				desired_goals_latents.append(obs['desired_goal_latent'].copy())

		# if HGG has not been stopped yet, perform crucial HGG update step here
		# by updating the sampler, a set of intermediate goals is provided and stored in sampler
		# based on distance to target goal distribution, similarity of initial states and expected reward (see paper)
		# by bipartite matching
		if not self.stop:
			if args.vae_dist_help:
				if self.learn_calls > 0:
					self.sampler.update(initial_goals, desired_goals, initial_goals_latents, desired_goals_latents)
				else:
					self.sampler.update(initial_goals, desired_goals, None, None)
			else:
				self.sampler.update(initial_goals, desired_goals)

		achieved_trajectories = []
		achieved_init_states = []
		if args.vae_dist_help:
			achieved_trajectory_goals_latents = []
			achieved_trajectory_init_goals_latents = []
			achieved_trajectory_obstacle_latents = []
			achieved_trajectory_obstacle_latents_sizes = []

		explore_goals = []
		test_goals = []
		inside = []

		for i in range(args.episodes):
			obs = self.env_List[i].get_obs()
			init_state = obs['observation'].copy()

			# if HGG has not been stopped yet, sample from the goals provided by the update step
			# if it has been stopped, the goal to explore is simply the one generated by the environment
			if not self.stop:
				explore_goal = self.sampler.sample(i)
			else:
				explore_goal = desired_goals[i]

			# store goals in explore_goals list to check whether goals are within goal space later
			explore_goals.append(explore_goal)
			test_goal = self.env.generate_goal()
			test_goals.append(test_goal)

			# Perform HER training by interacting with the environment
			self.env_List[i].goal = explore_goal.copy()
			obs = self.env_List[i].get_obs()
			if write_goals != 0 and len(goal_list)<write_goals:
				goal_list.append(explore_goal.copy())

			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]
			if args.vae_dist_help:
				trajectory_goals_latents = [obs['achieved_goal_latent'].copy()]
				trajectory_obstacles_latents = [obs['obstacle_latent'].copy()]
				trajectory_obstacles_latents_sizes = [obs['obstacle_size_latent'].copy()]
			## just for video
			#tr_env_images = [take_env_image(self.env_List[i], args.img_size)]
			##
			for timestep in range(args.timesteps):
				# get action from the policy
				action = agent.step(obs, explore=True)
				#action = acs[timestep]
				obs, reward, done, info = self.env_List[i].step(action)
				trajectory.append(obs['achieved_goal'].copy())
				if args.vae_dist_help:
					trajectory_goals_latents.append(obs['achieved_goal_latent'].copy())
					trajectory_obstacles_latents.append(obs['obstacle_latent'].copy())
					trajectory_obstacles_latents_sizes.append(obs['obstacle_size_latent'].copy())
				## just for video
				#tr_env_images.append(take_env_image(self.env_List[i], args.img_size))
				##
				if timestep==args.timesteps-1: done = True#this makes that the last obs is as done
				current.store_step(action, obs, reward, done)
				stop_trajectory, im_info = check_conditions_after_step(obs, current, args)
				if args.imaginary_obstacle_transitions and im_info is not None:
					args.imaginary_buffer.store_im_info(im_info, env=self.env_List[i])
				if done or stop_trajectory: break
			achieved_trajectories.append(np.array(trajectory))
			achieved_init_states.append(init_state)
			if args.vae_dist_help:
				achieved_trajectory_goals_latents.append(np.array(trajectory_goals_latents))
				achieved_trajectory_init_goals_latents.append(trajectory_goals_latents[0].copy())
				achieved_trajectory_obstacle_latents.append(np.array(trajectory_obstacles_latents))
				achieved_trajectory_obstacle_latents_sizes.append(np.array(trajectory_obstacles_latents_sizes))

			# Trajectory is stored in replay buffer, replay buffer can be normal or EBP
			buffer.store_trajectory(current)
			#update normalizer
			norm_batch = buffer.sample_batch()
			if args.imaginary_obstacle_transitions and args.imaginary_buffer.counter > args.im_warmup:
				if args.im_norm_counter % args.im_norm_freq == 0:
					#use obs to get size
					len_b = len(norm_batch['obs'])
					im_norm_batch = args.imaginary_buffer.sample_batch()
					len_im = len(im_norm_batch['obs'])
					indices = np.random.choice(len_b, size=len_im, replace=False)
					for im_it, idx in enumerate(indices):
						for key in norm_batch.keys():
							norm_batch[key][idx] = im_norm_batch[key][im_it]
				args.im_norm_counter += 1
			agent.normalizer_update(norm_batch)

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					batch = buffer.sample_batch()
					if args.imaginary_obstacle_transitions and args.imaginary_buffer.counter > args.im_warmup:
						if args.im_train_counter % args.im_train_freq == 0:
							#replaces samples with samples of the imaginary buffer
							imaginary_batch = args.imaginary_buffer.sample_batch()
							len_b = len(batch['obs'])
							len_im = len(imaginary_batch['obs'])
							indices = np.random.choice(len_b, size=len_im, replace=False)
							for im_it, idx in enumerate(indices):
								for key in norm_batch.keys():
									batch[key][idx] = imaginary_batch[key][im_it]
						args.im_train_counter += 1
					# train with Hindsight Goals (HER step)
					info = agent.train(batch)
					args.logger.add_dict(info)
				# update target network
				agent.target_update()


		selection_trajectory_idx = {}
		for i in range(self.args.episodes):
			# only add trajectories with movement to the trajectory pool --> use default (L2) distance measure!
			if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1])>0.01:#todo?? use here distance as well?
				selection_trajectory_idx[i] = True
		for idx in selection_trajectory_idx.keys():
			if args.vae_dist_help:
				self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(),
													 achieved_init_states[idx].copy(),
													 achieved_trajectory_goals_latents[idx].copy(),
													 achieved_trajectory_init_goals_latents[idx].copy(),
													 achieved_trajectory_obstacle_latents[idx].copy(),
													 achieved_trajectory_obstacle_latents_sizes[idx].copy())
			else:
				self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(),
													 achieved_init_states[idx].copy())

		# unless in first call:
		# Check which of the explore goals are inside the target goal space
		# target goal space is represented by a sample of test_goals directly generated from the environemnt
		# an explore goal is considered inside the target goal space, if it is closer than the distance_threshold to one of the test goals
		# (i.e. would yield a non-negative reward if that test goal was to be achieved)
		if self.learn_calls > 0:
			assert len(explore_goals) == len(test_goals)
			for ex in explore_goals:
				is_inside = 0
				for te in test_goals:
					if goal_distance(ex, te) <= self.env.env.env.distance_threshold:
						is_inside = 1
				inside.append(is_inside)
			assert len(inside) == len(test_goals)
			inside_sum = 0
			for i in inside:
				inside_sum += i

			# If more than stop_hgg_threshold (e.g. 0.9) of the explore goals are inside the target goal space, stop HGG
			# and continue with normal HER.
			# By default, stop_hgg_threshold is disabled (set to a value > 1)
			average_inside = inside_sum / len(inside)
			self.args.logger.info("Average inside: {}".format(average_inside))
			if average_inside > self.stop_hgg_threshold:
				self.stop = True
				self.args.logger.info("Continue with normal HER")

		self.learn_calls += 1

		return goal_list if len(goal_list)>0 else None


def check_conditions_after_step(obs, trajectory, args):
	if args.imaginary_obstacle_transitions:
		assert 'coll' in obs.keys() and 'coll_bool_ar' in obs.keys()
		#there is a collision
		indices = np.nonzero(obs['coll_bool_ar'])[0]
		if len(indices) > 0:
			if len(trajectory.ep['acts']) > 1:
				#select just cases with moving obstacles
				diff = obs['obstacle_st_t'][:, 0:2] - trajectory.ep['obs'][-2]['obstacle_st_t'][:, 0:2]
				diff = diff[indices]
				diff = np.atleast_2d(diff)
				dist = np.sqrt(np.sum(np.square(diff), axis=1))
				#todo check which one is a good value
				indices_moving = np.nonzero(dist > 0.001)[0]
				if len(indices_moving) > 0:
					index = indices[np.random.choice(indices_moving)]
					imaginary_info_dict = {}
					for key in trajectory.ep.keys():
						imaginary_info_dict[key] = []
						imaginary_info_dict[key].append(trajectory.ep[key][-2])
						imaginary_info_dict[key].append(trajectory.ep[key][-1])
					imaginary_info = (index, imaginary_info_dict)
					if 'coll_stop' in obs.keys():
						return True, imaginary_info
					else:
						return False, imaginary_info
				else:
					if 'coll_stop' in obs.keys():
						return True, None
					else:
						return False, None
			else:
				if 'coll_stop' in obs.keys():
					return True, None
				else:
					return False, None
		else:
			return False, None
	else:
		if 'coll_stop' in obs.keys():
			if obs['coll_stop']:
				return True, None
			else:
				return False, None
		else:
			return False, None


class NormalLearner:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)

		self.env_List = []
		for i in range(args.episodes):
			self.env_List.append(make_env(args))

		self.learn_calls = 0
		self.success_n = 0

	def learn(self, args, env, env_test, agent, buffer, write_goals=0, epoch=None, cycle=None):
		# Actual learning cycle takes place here!
		goal_list = []

		# get initial position and goal from environment for each episode
		for i in range(args.episodes):
			obs = self.env_List[i].reset()

		for i in range(args.episodes):
			if write_goals != 0 and len(goal_list)<write_goals:
				goal_list.append(self.env_List[i].goal.copy())
			obs = self.env_List[i].get_obs()
			current = Trajectory(obs)
			## just for video
			tr_env_images = [take_env_image(self.env_List[i], args.img_size)]
			##
			for timestep in range(args.timesteps):
				# get action from the ddpg policy
				action = agent.step(obs, explore=True)
				obs, reward, done, info = self.env_List[i].step(action)
				## just for video
				#tr_env_images.append(take_env_image(self.env_List[i], args.img_size))
				##
				if timestep==args.timesteps-1: done = True#this makes that the last obs is as done
				current.store_step(action, obs, reward, done)
				if done: break

			# Trajectory is stored in replay buffer, replay buffer can be normal or EBP
			buffer.store_trajectory(current)
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					# train with Hindsight Goals (HER step)
					if args.her == 'none':
						info = agent.train(buffer.sample_batch(plain=True))
					else:
						info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
				# update target network
				agent.target_update()

		self.learn_calls += 1



		return goal_list if len(goal_list)>0 else None
