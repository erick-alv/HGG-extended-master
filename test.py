import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir, LoggerExtra
from utils.image_util import create_rollout_video
from j_vae.distance_estimation import calculate_distance, calculate_distance_real
from vae_env_inter import take_env_image
import copy


class Tester:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.env_test = make_env(args)

		self.info = []
		self.calls = 0
		if args.save_acc:
			make_dir('log/accs', clear=False)
			self.test_rollouts = 100

			self.env_List = []
			self.env_test_List = []
			for _ in range(self.test_rollouts):
				self.env_List.append(make_env(args))
				self.env_test_List.append(make_env(args))

			self.acc_record = {}
			self.acc_record[self.args.goal] = []
			for key in self.acc_record.keys():
				self.info.append('Success'+'@blue')
				self.info.append('MaxDistance')
				self.info.append('MinDistance')


	def test_acc(self, key, env, agent):
		if (self.args.vae_dist_help or self.args.transform_dense) and self.calls % 40 == 0:
			eps_idx = [0, 10, 20, self.test_rollouts-1]
			ex_logs = [LoggerExtra(self.args.logger.my_log_dir, 'results_it_{}_ep_{}_test'.format(self.calls, i))
					   for i in eps_idx]
			for i in range(len(eps_idx)):
				ex_logs[i].add_item('Step')
				ex_logs[i].add_item('Success')
				ex_logs[i].add_item('RealDirectDistance')
				ex_logs[i].add_item('RealPathDistance')
				ex_logs[i].add_item('RealDirectToPrevDistance')
				ex_logs[i].add_item('LatentDirectDistance')
				ex_logs[i].add_item('LatentPathDistance')
				ex_logs[i].add_item('LatentDirectToPrevDistance')


			acc_sum, obs = 0.0, []
			prev_obs = []
			env_images = [[] for _ in eps_idx]
			for i in range(self.test_rollouts):
				o = env[i].reset()
				obs.append(goal_based_process(o))
				prev_obs.append(o)
				if i in eps_idx:
					t = eps_idx.index(i)
					env_images[t].append(take_env_image(env[i], self.args.img_size))
			for timestep in range(self.args.timesteps):
				actions = agent.step_batch(obs)
				obs, infos = [], []
				for i in range(self.test_rollouts):
					ob, _, _, info = env[i].step(actions[i])
					obs.append(goal_based_process(ob))
					infos.append(info)
					if i in eps_idx:
						t = eps_idx.index(i)
						ex_logs[t].add_record('Step', timestep)
						ex_logs[t].add_record('Success', info['Success'])
						ex_logs[t].add_record('RealDirectDistance', info['Distance'])
						rpd = calculate_distance_real(np.array([1.3, 0.75]), obstacle_radius=np.array(0.13),
												 current_pos=ob['achieved_goal'][:2], goal_pos=ob['desired_goal'][:2],
												 range_x=None, range_y=None)
						ex_logs[t].add_record('RealPathDistance', rpd)
						rddpr = env[i].compute_distance(ob['achieved_goal'][:2], prev_obs[i]['achieved_goal'][:2])
						ex_logs[t].add_record('RealDirectToPrevDistance', rddpr)
						ldd = env[i].compute_distance(ob['achieved_goal_latent'], ob['desired_goal_latent'])
						ex_logs[t].add_record('LatentDirectDistance', ldd)
						lpd = calculate_distance(ob['obstacle_latent'], obstacle_radius=ob['obstacle_size_latent'],
												 current_pos=ob['achieved_goal_latent'],
												 goal_pos=ob['desired_goal_latent'],
												 range_x=[-1., 1.], range_y=[-1., 1.])
						ex_logs[t].add_record('LatentPathDistance', lpd)
						lddpr = env[i].compute_distance(ob['achieved_goal_latent'], prev_obs[i]['achieved_goal_latent'])
						ex_logs[t].add_record('LatentDirectToPrevDistance', lddpr)
						ex_logs[t].save_csv()

						env_images[t].append(take_env_image(env[i], self.args.img_size))
					prev_obs[i] = copy.deepcopy(ob)
			for i, t in enumerate(eps_idx):
				create_rollout_video(np.array(env_images[i]), args=self.args,
									 filename='rollout_it_{}_ep_{}_test'.format(self.calls, t))
			
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