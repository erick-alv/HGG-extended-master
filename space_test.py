import numpy as np
import time
from common import get_args, experiment_setup, load_vaes, make_env, load_field_parameters, load_dist_estimator
from vae_env_inter import take_env_image, take_image_objects, latents_from_images
import copy
from vae.latent_space_transformations import interval_map_function
import matplotlib.pyplot as plt
from play import Player
from algorithm.replay_buffer import goal_based_process
from utils.image_util import create_rollout_video
from collections import namedtuple
from PIL import Image

#the distances to the edges seen in the image are 0.025
doing_f_r = False
if doing_f_r:
	map_coords_x = interval_map_function(0., 64., 1.025, 1.575)
	map_coords_y = interval_map_function(0., 64., 0.475, 1.025)
	map_size = interval_map_function(0., 64., 0., 1.025 - 0.475)
else:
	map_coords_x = interval_map_function(-1., 1., 1.025, 1.575)
	map_coords_y = interval_map_function(-1., 1., 0.475, 1.025)
	map_size = interval_map_function(0., 2., 0., 1.025 - 0.475)

def map_coords(coords):
	if doing_f_r:
		# here the y coordinates are flipped since images use other direction
		m_xs = map_coords_x(coords[:, 0:1])
		ny = 64. - coords[:, 1:2]
		m_ys = map_coords_y(ny)

	else:
		m_xs = map_coords_x(coords[:, 0:1])
		m_ys = map_coords_y(-1. * coords[:, 1:2])
	return np.concatenate([m_xs, m_ys], axis=1)


	
def map_sizes(sizes):
	m_xs = map_size(sizes[:, 0:1])
	m_ys = map_size(sizes[:, 1:2])
	return np.concatenate([m_xs, m_ys], axis=1)

def save_results_from_obs(obs, args, args_real, name):
	real_coords_obstacle = np.array([obs[i]['real_obstacle_info'][:2] for i in range(len(obs))])
	real_size_obstacle = np.array([obs[i]['real_obstacle_info'][3:5] for i in range(len(obs))])
	real_coords_puck = np.array([obs[i]['achieved_goal'][:2] for i in range(len(obs))])
	real_size_puck = np.array([obs[i]['real_size_goal'][:2] for i in range(len(obs))])

	real_distance_to_goal = args_real.dist_estimator.calculate_distance_batch(obs[0]['desired_goal'],
																			  real_coords_puck)
	plt.plot(np.arange(len(real_distance_to_goal)), real_distance_to_goal)
	plt.savefig('log/space_tests/real_distance_to_goal_{}.png'.format(name))
	plt.clf()
	real_distance_to_goal2 = np.clip(real_distance_to_goal, a_min=None, a_max=1.0)
	plt.plot(np.arange(len(real_distance_to_goal)), real_distance_to_goal2)
	plt.savefig('log/space_tests/real_distance_to_goal_clipped_{}.png'.format(name))
	plt.clf()

	latent_coords_obstacle = np.array([obs[i]['obstacle_latent'][0] for i in range(len(obs))])
	latent_size_obstacle = np.array([obs[i]['obstacle_size_latent'][0] for i in range(len(obs))])
	latent_coords_puck = np.array([obs[i]['achieved_goal_latent'] for i in range(len(obs))])
	latent_size_puck = np.array([obs[i]['achieved_goal_size_latent'] for i in range(len(obs))])

	latent_distance_to_goal = args.dist_estimator.calculate_distance_batch(obs[0]['desired_goal_latent'],
																		   latent_coords_puck)
	plt.plot(np.arange(len(latent_distance_to_goal)), latent_distance_to_goal)
	plt.savefig('log/space_tests/latent_distance_to_goal_{}.png'.format(name))
	plt.clf()
	latent_distance_to_goal2 = np.clip(latent_distance_to_goal, a_min=None, a_max=2.5)
	plt.plot(np.arange(len(latent_distance_to_goal)), latent_distance_to_goal2)
	plt.savefig('log/space_tests/latent_distance_to_goal_clipped_{}.png'.format(name))
	plt.clf()

	latent_coords_obstacle = map_coords(latent_coords_obstacle)
	latent_size_obstacle = map_sizes(latent_size_obstacle)
	latent_coords_puck = map_coords(latent_coords_puck)
	latent_size_puck = map_sizes(latent_size_puck)

	its = np.arange(len(real_coords_obstacle))
	fig, ax = plt.subplots()
	ax.plot(its, real_coords_obstacle[:, 0], label='real')
	ax.plot(its, latent_coords_obstacle[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis')
	plt.savefig('log/space_tests/coords_obstacle_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_coords_obstacle[:, 1], label='real')
	ax.plot(its, latent_coords_obstacle[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis')
	plt.savefig('log/space_tests/coords_obstacle_Y_{}.png'.format(name))
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(its, real_size_obstacle[:, 0], label='real')
	ax.plot(its, latent_size_obstacle[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis size')
	plt.savefig('log/space_tests/size_obstacle_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_size_obstacle[:, 1], label='real')
	ax.plot(its, latent_size_obstacle[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis size')
	plt.savefig('log/space_tests/size_obstacle_Y_{}.png'.format(name))
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(its, real_coords_puck[:, 0], label='real')
	ax.plot(its, latent_coords_puck[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis')
	plt.savefig('log/space_tests/coords_puck_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_coords_puck[:, 1], label='real')
	ax.plot(its, latent_coords_puck[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis')
	plt.savefig('log/space_tests/coords_puck_Y_{}.png'.format(name))
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(its, real_size_puck[:, 0], label='real')
	ax.plot(its, latent_size_puck[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis size')
	plt.savefig('log/space_tests/size_puck_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_size_puck[:, 1], label='real')
	ax.plot(its, latent_size_puck[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis size')
	plt.savefig('log/space_tests/size_puck_Y_{}.png'.format(name))
	plt.clf()


def calculate_comparison_player(args, args_real, env, player):
	o = env.reset()
	if player is None:
		name = 'none'
	else:
		name = 'player_{}'.format(player.play_epoch)
	n = np.random.randint(100)

	for timestep in range(n):
		if player is None:
			action = [0., 0., 0., 0.]
		else:
			action = player.my_step_batch([goal_based_process(o)])[0]
		o, _, _, info = env.step(action)
	image_objects_current = take_image_objects(env, args.img_size)
	image_env_current = take_env_image(env, args.img_size)
	lg, lg_s, lo, lo_s = [], [], [], []
	for _ in range(10):
		slg, slg_s, slo, slo_s = latents_from_images(np.array([image_objects_current.copy()]), args)
		lg.append(slg[0])
		lg_s.append(slg_s[0])
		lo.append(slo[0])
		lo_s.append(slo_s[0])
	lg = np.array(lg)
	lg_s = np.array(lg_s)
	lo = np.array(lo)
	lo_s = np.array(lo_s)

	im_current = Image.fromarray(image_env_current.copy().astype(np.uint8))
	im_current.save('log/space_tests/current_image_{}.png'.format(name))
	im_current_objects = Image.fromarray(image_objects_current.copy().astype(np.uint8))
	im_current_objects.save('log/space_tests/current_image_objects_{}.png'.format(name))


	real_coords_obstacle = np.array([o['real_obstacle_info'][:2]]*10)
	real_size_obstacle = np.array([o['real_obstacle_info'][3:5]]*10)
	real_coords_puck = np.array([o['achieved_goal'][:2]]*10)
	real_size_puck = np.array([o['real_size_goal'][:2]]*10)

	latent_coords_obstacle = lo[:, 0, :]
	latent_size_obstacle = lo_s[:, 0, :]
	latent_coords_puck = lg
	latent_size_puck = lg_s

	latent_coords_obstacle = map_coords(latent_coords_obstacle)
	latent_size_obstacle = map_sizes(latent_size_obstacle)
	latent_coords_puck = map_coords(latent_coords_puck)
	latent_size_puck = map_sizes(latent_size_puck)

	its = np.arange(len(real_coords_obstacle))
	fig, ax = plt.subplots()
	ax.plot(its, real_coords_obstacle[:, 0], label='real')
	ax.plot(its, latent_coords_obstacle[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis')
	plt.savefig('log/space_tests/static_coords_obstacle_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_coords_obstacle[:, 1], label='real')
	ax.plot(its, latent_coords_obstacle[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis')
	plt.savefig('log/space_tests/static_coords_obstacle_Y_{}.png'.format(name))
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(its, real_size_obstacle[:, 0], label='real')
	ax.plot(its, latent_size_obstacle[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis size')
	plt.savefig('log/space_tests/static_size_obstacle_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_size_obstacle[:, 1], label='real')
	ax.plot(its, latent_size_obstacle[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis size')
	plt.savefig('log/space_tests/static_size_obstacle_Y_{}.png'.format(name))
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(its, real_coords_puck[:, 0], label='real')
	ax.plot(its, latent_coords_puck[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis')
	plt.savefig('log/space_tests/static_coords_puck_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_coords_puck[:, 1], label='real')
	ax.plot(its, latent_coords_puck[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis')
	plt.savefig('log/space_tests/static_coords_puck_Y_{}.png'.format(name))
	plt.clf()

	fig, ax = plt.subplots()
	ax.plot(its, real_size_puck[:, 0], label='real')
	ax.plot(its, latent_size_puck[:, 0], label='latent_mapped')
	plt.legend()
	plt.title('x-axis size')
	plt.savefig('log/space_tests/static_size_puck_X_{}.png'.format(name))
	plt.clf()
	fig, ax = plt.subplots()
	ax.plot(its, real_size_puck[:, 1], label='real')
	ax.plot(its, latent_size_puck[:, 1], label='latent_mapped')
	plt.legend()
	plt.title('y-axis size')
	plt.savefig('log/space_tests/static_size_puck_Y_{}.png'.format(name))
	plt.clf()

	obs = []
	o = env.reset()
	env_images = []
	obs.append(o)
	env_images.append(take_env_image(env, args.img_size))

	for timestep in range(100):
		if player is None:
			action = [0., 0., 0., 0.]
		else:
			action = player.my_step_batch([goal_based_process(o)])[0]
		o, _, _, info = env.step(action)
		obs.append(o)
		env_images.append(take_env_image(env, args.img_size))
	create_rollout_video(env_images, args=args,
						 filename='episode_{}'.format(name))

	save_results_from_obs(obs, args, args_real, name)

	obs = []
	o = env.reset()
	obs.append(o)
	for ep in range(10):
		for timestep in range(100):
			if player is None:
				action = [0., 0., 0., 0.]
			else:
				action = player.my_step_batch([goal_based_process(o)])[0]
			o, _, _, info = env.step(action)
			obs.append(o)

	real_coords_obstacle = np.array([obs[i]['real_obstacle_info'][:2] for i in range(len(obs))])
	real_size_obstacle = np.array([obs[i]['real_obstacle_info'][3:5] for i in range(len(obs))])
	real_coords_puck = np.array([obs[i]['achieved_goal'][:2] for i in range(len(obs))])
	real_size_puck = np.array([obs[i]['real_size_goal'][:2] for i in range(len(obs))])

	#real_distance_to_goal = args_real.dist_estimator.calculate_distance_batch(obs[0]['desired_goal'], real_coords_puck)

	latent_coords_obstacle = np.array([obs[i]['obstacle_latent'][0] for i in range(len(obs))])
	latent_size_obstacle = np.array([obs[i]['obstacle_size_latent'][0] for i in range(len(obs))])
	latent_coords_puck = np.array([obs[i]['achieved_goal_latent'] for i in range(len(obs))])
	latent_size_puck = np.array([obs[i]['achieved_goal_size_latent'] for i in range(len(obs))])


	#latent_distance_to_goal = args.dist_estimator.calculate_distance_batch(obs[0]['desired_goal_latent'],
	#																	   latent_coords_puck)

	latent_coords_obstacle = map_coords(latent_coords_obstacle)
	latent_size_obstacle = map_sizes(latent_size_obstacle)
	latent_coords_puck = map_coords(latent_coords_puck)
	latent_size_puck = map_sizes(latent_size_puck)

	mse_coords_obstacle = ((real_coords_obstacle - latent_coords_obstacle) ** 2).mean()
	mse_size_obstacle = ((real_size_obstacle - latent_size_obstacle) ** 2).mean()
	mse_coords_puck = ((real_coords_puck - latent_coords_puck) ** 2).mean()
	mse_size_puck = ((real_size_puck - latent_size_puck) ** 2).mean()

	print(' mse_coords_obstacle: {}\n mse_size_obstacle: {}\n mse_coords_puck: {}\n mse_size_puck: {}\n'.format(
		mse_coords_obstacle, mse_size_obstacle, mse_coords_puck, mse_size_puck
	))

def compare_rollouts(args, args_real, env):
	print("None")
	calculate_comparison_player(args=args, args_real=args_real, env=env, player=None)

	print("player best")
	args.play_path = 'log/072-ddpg-FetchPushMovingObstacleEnv-v1-hgg-optimal-stop/'
	args.play_epoch = 'best'
	player = Player(args)
	calculate_comparison_player(args=args, args_real=args_real, env=env, player=player)

	print("player epoch 2")
	del player
	args.play_epoch = '2'
	player = Player(args)
	calculate_comparison_player(args=args, args_real=args_real, env=env, player=player)

def compare_at_positions(args, args_real, env):
	r1 = np. array([[1.115, 0.825], [1.189, 0.825], [1.263, 0.825], [1.337, 0.825], [1.411, 0.825], [1.485, 0.825]])
	r1 += np.array([0. , 0.002])
	r2 = np. array([[1.485, 0.7875],[1.485, 0.75],[1.485, 0.7125], [1.485, 0.675]])
	r3 = np.array([[1.411, 0.675], [1.337, 0.675], [1.263, 0.675], [1.189, 0.675], [1.115, 0.675]])
	r3 -= np.array([0. , 0.002])
	r4 = np.array([[1.115, 0.7125],[1.115, 0.75], [1.115, 0.7875], [1.115, 0.825]])
	routes = np.concatenate([r1, r2, r3, r4], axis=0)

	obs = []
	env_images = []
	env.reset()
	for px, py in routes:
		env.env.env._move_object(position=[px, py, 0.425])
		for timestep in range(10):
			'''o = env.get_obs()
			obs.append(o)
			env_images.append(take_env_image(env, args.img_size))'''
			action = [0., 0., 0., 0.]
			o, _, _, info = env.step(action)
			obs.append(o)
			env_images.append(take_env_image(env, args.img_size))

	save_results_from_obs(obs, args, args_real, 'around_obstacle')
	create_rollout_video(env_images, args=args,
						 filename='around_obstacle')

	pos_x = [1.1, 1.2, 1.3, 1.5]
	pos_y = np.linspace(0.5, 1.0, num=10, endpoint=True)
	for i, px in enumerate(pos_x):
		obs = []
		env_images = []
		env.reset()
		for py in pos_y:
			env.env.env._move_object(position=[px, py, 0.425])
			for timestep in range(3):
				action = [0., 0., 0., 0.]
				o, _, _, info = env.step(action)
				obs.append(o)
				env_images.append(take_env_image(env, args.img_size))

		save_results_from_obs(obs, args, args_real, 'on_row_{}'.format(i))
		create_rollout_video(env_images, args=args,
							 filename='on_row_{}'.format(i))



if __name__=='__main__':

	# Getting arguments from command line + defaults
	# Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
	args = get_args()
	# creates copy of args for the real coordinates
	args_real = copy.copy(args)
	#this class compares space generated by neuralt network (in this case Bbox) with real coordinates)
	load_vaes(args)
	env = make_env(args)
	load_field_parameters(args)
	load_dist_estimator(args, env)

	# setup for real coordinates
	args_real.vae_dist_help = False
	args_real.vae_type = None
	args_real.dist_estimator_type = 'multipleReal' if 'multiple' in args.dist_estimator_type else 'realCoords'
	env_real = make_env(args_real)
	load_field_parameters(args_real)
	load_dist_estimator(args_real, env_real)

	#print(args.field_center)
	print('latent_obstacles: {}'.format(args.dist_estimator.obstacles))
	#print(args_real.field_center)
	print('real_obstacles: {}'.format(args_real.dist_estimator.obstacles))

	mapped_to_real = np.array(copy.copy(args.dist_estimator.obstacles))
	coords_to_map = mapped_to_real[:, 0:2].copy()
	coords_mapped = map_coords(coords_to_map)
	sizes_to_map = mapped_to_real[:, 2:4].copy()
	sizes_mapped = map_sizes(sizes_to_map)
	for i in range(len(mapped_to_real)):
		mapped_to_real[i][0] = coords_mapped[i][0]
		mapped_to_real[i][1] = coords_mapped[i][1]
		mapped_to_real[i][2] = sizes_mapped[i][0]
		mapped_to_real[i][3] = sizes_mapped[i][1]
	print('latent_obstacles mapped: {}'.format(mapped_to_real))
	LoggerMock = namedtuple('Logger', ['my_log_dir'])
	args.logger = LoggerMock(my_log_dir='log/space_tests/')

	#compare rollouts
	#compare_rollouts(args=args, args_real=args_real, env=env)
	compare_at_positions(args=args, args_real=args_real, env=env)



	




