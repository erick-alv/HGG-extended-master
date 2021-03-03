import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir, LoggerExtra
from utils.image_util import create_rollout_video
from utils.hindsight_goals_visualizer import show_points
from vae_env_inter import take_env_image
import copy


class Tester:
	def __init__(self, args, test_rollouts=100, after_train_test=False):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)

		self.info = []
		self.calls = 0
		self.after_train_test = after_train_test
		if args.save_acc:
			make_dir('log/accs', clear=False)
			self.test_rollouts = test_rollouts

			self.env_List = []
			for _ in range(self.test_rollouts):
				self.env_List.append(make_env(args))

			self.acc_record = {}
			self.acc_record[self.args.goal] = []
			for key in self.acc_record.keys():
				self.info.append('Success'+'@blue')
				self.info.append('MaxDistance')
				self.info.append('MinDistance')

		self.coll_tol = 0 #this attribute is just used for tests after training


	def test_acc(self, key, env, agent):
		if self.args.goal in self.args.colls_test_check_envs:
			# +1 since when it becomes 0 then not considered as success
			# coll toll is modified externally for every iteraiton
			envs_collision = [self.coll_tol + 1 for _ in range(len(self.env_List))]
		if self.calls % 40 == 0 or self.calls in [0, 1, 2, 5, 8, 10] or self.after_train_test:
			if self.after_train_test:
				eps_idx = [i for i in range(0, self.test_rollouts-1, 20)] + [self.test_rollouts-1]
			else:
				eps_idx = [0, 5, 8, 10, 15, 20, self.test_rollouts-1]
			acc_sum, obs = 0.0, []
			prev_obs = []
			env_images = [[] for _ in eps_idx]
			if self.args.vae_dist_help:
				latent_points = [[] for _ in eps_idx]
			for i in range(self.test_rollouts):
				o = env[i].reset()
				obs.append(goal_based_process(o))
				prev_obs.append(o)
				if i in eps_idx:
					t = eps_idx.index(i)
					env_images[t].append(take_env_image(env[i], self.args.img_vid_size))
					if self.args.vae_dist_help:
						latent_points[t].append(o['achieved_goal_latent'])
			for timestep in range(self.args.timesteps):
				actions = agent.step_batch(obs)
				obs, infos = [], []
				for i in range(self.test_rollouts):
					ob, _, _, info = env[i].step(actions[i])
					obs.append(goal_based_process(ob))
					#this should be used just for testing after training
					if self.args.goal in self.args.colls_test_check_envs:
						if ob['collision_check']:
							envs_collision[i] -= 1
						if envs_collision[i] <= 0:
							info['Success'] = 0.0
					infos.append(info)


					if i in eps_idx:
						t = eps_idx.index(i)
						env_images[t].append(take_env_image(env[i], self.args.img_vid_size))
						if self.args.vae_dist_help:
							latent_points[t].append(ob['achieved_goal_latent'])
					prev_obs[i] = copy.deepcopy(ob)
			for i, t in enumerate(eps_idx):
				create_rollout_video(np.array(env_images[i]), args=self.args,
									 filename='rollout_it_{}_ep_{}_test'.format(self.calls, t))
				name = "{}rollout_latent_coords_it_{}_ep_{}_test".format(self.args.logger.my_log_dir, self.calls, t)
				if self.args.vae_dist_help:
					show_points(points_list=np.array(latent_points[i]), save_file=name, space_of='latent')
			
		else:
			acc_sum, obs = 0.0, []
			for i in range(self.test_rollouts):
				obs.append(goal_based_process(env[i].reset()))
			for timestep in range(self.args.timesteps):
				actions = agent.step_batch(obs)
				obs, infos = [], []
				for i in range(self.test_rollouts):
					ob, _, _, info = env[i].step(actions[i])
					obs.append(goal_based_process(ob))
					if self.args.goal in self.args.colls_test_check_envs:
						if ob['collision_check']:
							envs_collision[i] -= 1
						if envs_collision[i] <= 0:
							info['Success'] = 0.0
					infos.append(info)
		minDist = infos[0]['Distance']
		maxDist = infos[0]['Distance']
		for i in range(self.test_rollouts):
			acc_sum += infos[i]['Success']
			if infos[i]['Distance'] < minDist:
				minDist = infos[i]['Distance']
			if infos[i]['Distance'] > maxDist:
				maxDist = infos[i]['Distance']

		steps = self.args.buffer.counter
		acc = acc_sum/self.test_rollouts
		self.acc_record[key].append((steps, acc, minDist, maxDist))
		self.args.logger.add_record('Success', acc)
		self.args.logger.add_record('MaxDistance', maxDist)
		self.args.logger.add_record('MinDistance', minDist)
		self.calls += 1

	def cycle_summary(self):
		if self.args.save_acc:
			self.test_acc(self.args.goal, self.env_List, self.args.agent)

	def epoch_summary(self):
		if self.args.save_acc:
			for key, acc_info in self.acc_record.items():
				log_folder = 'accs'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)

	def final_summary(self):
		if self.args.save_acc:
			for key, acc_info in self.acc_record.items():
				log_folder = 'accs'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)