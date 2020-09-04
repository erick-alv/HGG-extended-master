import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir, LoggerExtra

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

	def process_extra_log(self, prev_ob, ob, i, logger, env):
		pass



	def test_acc(self, key, env, agent):
		if False:
			ex_log = LoggerExtra(self.args.logger.my_log_dir, 'results_it_{}_test'.format(self.calls))
			ex_log.add_item('')

			acc_sum, obs = 0.0, []
			prev_obs = []
			for i in range(self.test_rollouts):
				o = env[i].reset()
				obs.append(goal_based_process(o))
				prev_obs.append(o)
			for timestep in range(self.args.timesteps):
				actions = agent.step_batch(obs)
				obs, infos = [], []
				for i in range(self.test_rollouts):
					ob, _, _, info = env[i].step(actions[i])
					obs.append(goal_based_process(ob))
					infos.append(info)
					if i == self.test_rollouts - 1:

						self.process_extra_log(prev_obs[i], ob, i, env)
					prev_obs[i] = ob
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
		self.args.logger.add_record('MaxDistance', minDist)
		self.args.logger.add_record('MinDistance', maxDist)
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