import numpy as np
import time
from common import get_args, experiment_setup
from utils.hindsight_goals_visualizer import show_points
from copy import deepcopy
import pickle
import torch
import tensorflow as tf

from gym.envs.registration import register

if __name__=='__main__':

	# Getting arguments from command line + defaults
	# Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
	args = get_args()
	env, env_test, agent, buffer, learner, tester = experiment_setup(args)
	args.logger.summary_init(agent.graph, agent.sess)


	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)')

	best_success = -1

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()
	counter= 0

	# Learning
	for epoch in range(args.epoches):
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()
			start_time = time.time()

			# Learn
			goal_list = learner.learn(args, env, env_test, agent, buffer, write_goals=args.show_goals,
									  epoch=epoch, cycle=cycle)


			# Log learning progresss
			tester.cycle_summary()
			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epoches))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			# Save learning progress to progress.csv file
			args.logger.save_csv()

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

			# Save latest policy
			policy_file = args.logger.my_log_dir + "saved_policy-latest"
			agent.save(policy_file)

			# Save policy if new best_success was reached
			if args.logger.values["Success"] > best_success:
				best_success = args.logger.values["Success"]
				policy_file = args.logger.my_log_dir + "saved_policy-best"
				agent.save(policy_file)
				args.logger.info("Saved as best policy to {}!".format(args.logger.my_log_dir))


		# Save periodic policy every epoch
		policy_file = args.logger.my_log_dir + "saved_policy"
		agent.save(policy_file, global_step=epoch)
		args.logger.info("Saved periodic policy to {}!".format(args.logger.my_log_dir))

		# Plot current goal distribution for visualization (G-HGG only)
		if args.learn == 'hgg' and goal_list and args.show_goals != 0 and cycle % 10 == 0:
			name = "{}goals_{}_{}".format(args.logger.my_log_dir, epoch, cycle)
			if args.vae_dist_help:
				# !!!!!!!!!! currently the goals are still in real space
				show_points(np.array(goal_list), name, 'real')
			else:
				show_points(np.array(goal_list), name, 'real')

		# Plot current goal distribution for visualization (G-HGG only)
		if args.learn == 'hgg' and goal_list and args.show_goals != 0:
			name = "{}goals_{}".format(args.logger.my_log_dir, epoch)
			'''if args.graph:
				learner.sampler.graph.plot_graph(goals=goal_list, save_path=name)'''
			with open('{}.pkl'.format(name), 'wb') as file:
					pickle.dump(goal_list, file)
			if args.vae_dist_help:
				#!!!!!!!!!! currently the goals are still in real space
				show_points(np.array(goal_list), name, 'real')
			else:
				show_points(np.array(goal_list), name, 'real')

		tester.epoch_summary()

	tester.final_summary()

