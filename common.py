import numpy as np
import copy
from envs import make_env, make_temp_env, clip_return_range, Robotics_envs_id
from utils.os_utils import get_arg_parser, get_logger, str2bool
from algorithm import create_agent
from learner import create_learner, learner_collection
from test import Tester
from algorithm.replay_buffer import ReplayBuffer_Episodic, goal_based_process, ReplayBuffer_Imaginary
import torch
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from j_vae.train_vae import load_Vae
from j_vae.train_monet import load_Vae as load_Monet
from j_vae.Bbox import load_Model as load_Bbox
from j_vae.faster_rcnn import load_faster_rcnn
from j_vae.common_data import vae_sb_weights_file_name, vae_weights_file_name
from PIL import Image
from vae_env_inter import take_env_image, take_image_objects
from j_vae.distance_estimation import calculate_distance
from dist_estimator import DistMovEst, DistMovEstReal, MultipleDist, MultipleDistReal, Estimator_DistNet
from SPACE.main_space import load_space_model
import matplotlib.pyplot as plt
import os



def get_args(do_just_test=False):#this parameter is just used for the name
	parser = get_arg_parser()

	parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
	parser.add_argument('--alg', help='backend algorithm', type=str, default='ddpg', choices=['ddpg', 'ddpg2', 'sac'])
	parser.add_argument('--learn', help='type of training method', type=str, default='hgg', choices=learner_collection.keys())

	parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1', choices=Robotics_envs_id)
	args, _ = parser.parse_known_args()
	if args.env=='HandReach-v0':
		parser.add_argument('--goal', help='method of goal generation', type=str, default='reach', choices=['vanilla', 'reach'])
	else:
		parser.add_argument('--goal', help='method of goal generation', type=str, default='interval',
							choices=['vanilla','fixobj','interval','custom' ,'intervalCollision','intervalExt',
									 'intervalColl',
									'intervalRewSub',
									'intervalRewVec',
									'intervalTestExtendedBbox',
									'intervalCollStop',
									'intervalRewMod',
									'intervalCollStopRegion',
									'intervalRewModStop',
									'intervalRewModRegion',
									'intervalRewModRegionStop',
									'intervalCollMinDist',
									 'intervalMinDistRewMod',
									 'intervalMinDistRewModStop',
									 'intervalTestExtendedMinDist',
									 'intervalCollPAV',
									 'intervalP', 'intervalPRewMod', 'intervalPRewModStop', 'intervalTestExtendedP',
									 'intervalPAV', 'intervalPAVRewMod', 'intervalPAVRewModStop',
									 'intervalTestExtendedPAV', 'intervalPRel', 'intervalPRelRewMod',
									 'intervalPRelRewModStop', 'intervalTestExtendedPRel', 'intervalPAVRel',
									 'intervalPAVRelRewMod', 'intervalPAVRelRewModStop', 'intervalTestExtendedPAVRel'
									 ])


		if args.env[:5]=='Fetch':
			parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32, default=1.0)
		elif args.env[:4]=='Hand':
			parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32, default=0.25)

	args, _ = parser.parse_known_args()
	if 'RewMod' in args.goal:
		parser.add_argument('--rew_mod_val', help='value to subtract on collision', type=np.float32, default=-2.)
	parser.add_argument('--graph', help='g-hgg yes or no', type=str2bool, default=False)
	parser.add_argument('--show_goals', help='number of goals to show', type=np.int32, default=0)
	parser.add_argument('--play_path', help='path to meta_file directory for play', type=str, default=None)
	parser.add_argument('--play_epoch', help='epoch to play', type=str, default='latest')
	parser.add_argument('--stop_hgg_threshold', help='threshold of goals inside goalspace, between 0 and 1, deactivated by default value 2!', type=np.float32, default=2)
	parser.add_argument('--agent_device', help='the device to load the agent', type=str, default='cpu')

	parser.add_argument('--n_x', help='number of vertices in x-direction for g-hgg', type=int, default=31)
	parser.add_argument('--n_y', help='number of vertices in y-direction for g-hgg', type=int, default=31)
	parser.add_argument('--n_z', help='number of vertices in z-direction for g-hgg', type=int, default=11)


	parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.98)
	parser.add_argument('--clip_return', help='whether to clip return value', type=str2bool, default=True)
	# these two arguments might be helpful if using other than sparse reward (-1, 0)
	parser.add_argument('--reward_min', help='discount factor', type=np.float32, default=-1.)
	parser.add_argument('--reward_max', help='discount factor', type=np.float32, default=0.)
	parser.add_argument('--eps_act', help='percentage of epsilon greedy explorarion', type=np.float32, default=0.3)
	parser.add_argument('--std_act', help='standard deviation of uncorrelated gaussian explorarion', type=np.float32, default=0.2)


	parser.add_argument('--pi_lr', help='learning rate of policy network', type=np.float32, default=1e-3)
	parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=1e-3)
	parser.add_argument('--act_l2', help='quadratic penalty on actions', type=np.float32, default=1.0)
	parser.add_argument('--polyak', help='interpolation factor in polyak averaging for DDPG', type=np.float32, default=0.95)

	parser.add_argument('--epoches', help='number of epoches', type=np.int32, default=20)
	parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=20)
	parser.add_argument('--episodes', help='number of episodes per cycle', type=np.int32, default=50)
	parser.add_argument('--timesteps', help='number of timesteps per episode', type=np.int32, default=(50 if args.env[:5]=='Fetch' else 100))
	parser.add_argument('--train_batches', help='number of batches to train per episode', type=np.int32, default=20)

	parser.add_argument('--buffer_size', help='number of episodes in replay buffer', type=np.int32, default=10000)
	parser.add_argument('--buffer_type', help='type of replay buffer / whether to use Energy-Based Prioritization', type=str, default='energy', choices=['normal','energy'])
	parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=256)
	parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=10000)
	parser.add_argument('--her', help='type of hindsight experience replay', type=str, default='future', choices=['none', 'final', 'future'])
	parser.add_argument('--her_ratio', help='ratio of hindsight experience replay', type=np.float32, default=0.8)
	parser.add_argument('--pool_rule', help='rule of collecting achieved states', type=str, default='full', choices=['full', 'final'])
	parser.add_argument('--imaginary_obstacle_transitions',
						help='expand obstacle transition', type=bool, default=False)


	parser.add_argument('--hgg_c', help='weight of initial distribution in flow learner', type=np.float32, default=3.0)
	parser.add_argument('--hgg_L', help='Lipschitz constant', type=np.float32, default=5.0)
	parser.add_argument('--hgg_pool_size', help='size of achieved trajectories pool', type=np.int32, default=1000)

	parser.add_argument('--save_acc', help='save successful rate', type=str2bool, default=True)

	#arguments for VAEs and images
	parser.add_argument('--vae_dist_help', help='using vaes yes or no', type=str2bool, default=False)
	parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)
	#type of VAE
	parser.add_argument('--vae_type', help='', type=str,
						default=None, choices=['sb', 'mixed', 'monet', 'space', 'bbox','faster_rcnn'])
	#type VAE for size
	parser.add_argument('--vae_size_type', help='', type=str,
						default='all', choices=['normal', 'sb', 'mixed', 'monet'])#if mixed or monet then representation is shared

	#parameters for VAE
	parser.add_argument('--latent_size_obstacle', help='size latent space obstacle', type=np.int32, default=None)
	parser.add_argument('--latent_size_goal', help='size latent space goal', type=np.int32, default=None)
	parser.add_argument('--obstacle_ind_1', help='index 1 component latent vector', type=np.int32, default=None)
	parser.add_argument('--obstacle_ind_2', help='index 2 component latent vector', type=np.int32, default=None)
	parser.add_argument('--goal_ind_1', help='index 1 component latent vector', type=np.int32, default=None)
	parser.add_argument('--goal_ind_2', help='index 2 component latent vector', type=np.int32, default=None)
	parser.add_argument('--goal_slot', help='', type=np.int32, default=None)
	parser.add_argument('--obstacle_slot', help='', type=np.int32, default=None)

	#parameter for size VAE
	parser.add_argument('--size_ind', help='index 2 component latent vector', type=np.int32, default=None)
	parser.add_argument('--size_ind_2', help='index 2 component latent vector', type=np.int32, default=None)

	parser.add_argument('--dist_estimator_type', help='the type if dist estimator to use or None if not using',
						type=str, default=None,
						choices=['normal', 'realCoords', 'multiple', 'multipleReal', 'net'])
	#for dense reward transformation
	parser.add_argument('--transform_dense', help='if transform to dense with VAES or not', type=str2bool, default=False)

	args = parser.parse_args()
	args.num_vertices = [args.n_x, args.n_y, args.n_z]
	args.goal_based = (args.env in Robotics_envs_id)
	args.clip_return_l, args.clip_return_r = clip_return_range(args)


	base_name = args.alg + '-' + args.env + '-' + args.goal + '-' + args.learn
	if do_just_test:
		if args.play_path is not None:
			remaining, last = os.path.split(args.play_path)
			if args.env in last:
				logger_name = 'TEST-'+last
			else:
				remaining, last = os.path.split(remaining)
				if args.env in last:
					logger_name = 'TEST-' + last
				else:
					logger_name = 'TEST-' + base_name

		else:
			logger_name = 'TEST-' + base_name
	else:
		logger_name = base_name
		if args.tag!='': logger_name = args.tag+'-'+logger_name
		if args.graph:
			logger_name =logger_name + '-graph'
		if args.stop_hgg_threshold < 1:
			logger_name = logger_name + '-stop'
		if args.dist_estimator_type is not None:
			logger_name = logger_name+'-' + args.dist_estimator_type
		if args.vae_type is not None:
			logger_name = logger_name +'-'+ args.vae_type
		if 'RewMod' in args.goal:
			logger_name = logger_name +'-rewmodVal('+ str(args.rew_mod_val)+')'
		if args.imaginary_obstacle_transitions:
			logger_name = logger_name + '-IMAGINARY'
	args.logger = get_logger(logger_name)


	for key, value in args.__dict__.items():
		if key!='logger':
			args.logger.info('{}: {}'.format(key,value))

	cuda = torch.cuda.is_available()
	torch.manual_seed(1)
	device = torch.device("cuda" if cuda else "cpu")
	args.device = device

	#extensions from intervale_ext


	if args.goal in ['intervalRewVec']:
		args.reward_dims = 2
	else:
		args.reward_dims = 1

	args.colls_test_check_envs = ['intervalTestExtendedBbox', 'intervalTestExtendedMinDist', 'intervalTestExtendedP',
								  'intervalTestExtendedPAV', 'intervalTestExtendedPRel', 'intervalTestExtendedPAVRel']


	return args


def load_vaes(args):
	base_data_dir = 'data/'
	data_dir = base_data_dir + args.env + '/'

	#load VAES for positional data
	if args.vae_type == 'space':
		args.vae_model = load_space_model(checkpoint_path='data/FetchGenerativeEnv-v1/',
							 check_name='data/FetchGenerativeEnv-v1/model_000030001.pth', device='cuda:0')
		return
	elif args.vae_type == 'bbox':
		args.vae_model = load_Bbox(path='data/FetchGenerativeEnv-v1/model_bbox',img_size=args.img_size, latent_size=0,
								   device='cuda:0', num_slots=5)#latent size is not being used for now
		args.vae_model.eval()
		return
	elif args.vae_type == 'faster_rcnn':
		args.vae_model = load_faster_rcnn(path='data/FetchGenerativeEnv-v1/model_rcnn.pth', device='cuda:0')  # latent size is not being used for now
		args.vae_model.eval()
		return


	if args.vae_type == 'sb':
		weights_path_goal = data_dir + vae_sb_weights_file_name['goal']
		args.weights_path_goal = weights_path_goal
		weights_path_obstacle = data_dir + vae_sb_weights_file_name['obstacle']
		args.weights_path_obstacle = weights_path_obstacle
		args.vae_model_obstacle = load_Vae_SB(weights_path_obstacle, args.img_size, args.latent_size_obstacle)
		args.vae_model_obstacle.eval()
		args.vae_model_goal = load_Vae_SB(weights_path_goal, args.img_size, args.latent_size_goal)
		args.vae_model_goal.eval()
	elif args.vae_type == 'mixed':
		assert args.goal_ind_1 == args.obstacle_ind_1 and args.goal_ind_2 == args.obstacle_ind_2
		assert args.latent_size_obstacle == args.latent_size_goal
		weights_path = data_dir + vae_sb_weights_file_name['mixed']
		args.weights_path_goal = weights_path
		args.weights_path_obstacle = weights_path
		model = load_Vae_SB(weights_path, args.img_size, args.latent_size_obstacle)
		model.eval()
		args.vae_model_obstacle = model
		args.vae_model_goal = model
	elif args.vae_type == 'monet':
		assert args.goal_ind_1 == args.obstacle_ind_1 and args.goal_ind_2 == args.obstacle_ind_2
		assert args.latent_size_obstacle == args.latent_size_goal
		assert args.goal_slot is not  None
		assert args.obstacle_slot is not  None
		weights_path = data_dir + vae_sb_weights_file_name['all']
		args.weights_path_goal = weights_path
		args.weights_path_obstacle = weights_path
		model = load_Monet(path=weights_path, img_size=args.img_size, latent_size=args.latent_size_obstacle)
		#model.eval()TODO with eval it does not work, is there a solution??
		args.vae_model_obstacle = model
		args.vae_model_goal = model
	else:
		raise Exception("VAE type invalid or not given")

	if args.vae_size_type == 'normal':
		weights_path_obstacle_sizes = data_dir + vae_weights_file_name['obstacle_sizes']
		args.weights_path_obstacle_sizes = weights_path_obstacle_sizes
		args.vae_model_size = load_Vae(path=weights_path_obstacle_sizes, img_size=args.img_size, latent_size=1)
		args.vae_model_size.eval()
	elif args.vae_size_type == 'mixed' or args.vae_size_type == 'monet':
		assert args.size_ind != args.obstacle_ind_1 and args.size_ind != args.obstacle_ind_2
		args.weights_path_obstacle_sizes = data_dir + vae_sb_weights_file_name['obstacle']
		args.vae_model_size = args.vae_model_obstacle
	else:
		raise Exception("VAE type of size invalid or not given")


#This loads the field in 2D since methods used extract information in this way
def load_field_parameters(args):
	def load_real_field_params():
		if args.env in ['FetchPushLabyrinth-v1', 'FetchPushObstacleFetchEnv-v1','FetchPushMovingObstacleEnv-v1',
						'FetchPushMovingObstacleEnv-v2', 'FetchPushMovingComEnv-v1','FetchPushMovingComEnv-v2',
						'FetchPushMovingObstacleEnv-v3', 'FetchPushMovingComEnv-v3',
						'FetchPushMovingDoubleObstacleEnv-v1','FetchPushMovingDoubleObstacleEnv-v2',
						'FetchPushMovingDoubleObstacleEnv-v3']:
			args.real_field_center = [1.3, 0.75]
			args.real_field_size = [0.25, 0.25]
		else:
			raise Warning(
				'The environment used does not have predefined field dimensions. Assure they are not needed')

	if args.vae_dist_help:
		if args.vae_type == 'space' or args.vae_type == 'bbox':
			#model space is trained to create measures in range [-1, 1], a bit more space is given for the calculations
			args.field_center = [0., 0.]
			args.field_size = [1.0, 1.0]
			load_real_field_params()

		elif args.vae_type == 'faster_rcnn':
			args.field_center = [args.img_size / 2., args.img_size / 2.]
			args.field_size = [args.img_size / 2., args.img_size / 2.]
			load_real_field_params()
		else:
			raise Warning('Using a VAE or model, with own space. Assure that the transformations in this space are correct')
	else:
		load_real_field_params()
		args.field_center = args.real_field_center
		args.field_size = args.real_field_size


def load_dist_estimator(args, env):
	if args.dist_estimator_type == 'normal':
		args.dist_estimator = DistMovEst()
	elif args.dist_estimator_type == 'realCoords':
		args.dist_estimator = DistMovEstReal()
	elif args.dist_estimator_type == 'multipleReal':
		args.dist_estimator = MultipleDistReal()
	elif args.dist_estimator_type == 'multiple':
		args.dist_estimator = MultipleDist()
	elif args.dist_estimator_type == 'net':
		this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
		base_data_dir = this_file_dir + 'data/'
		env_data_dir = base_data_dir + args.env + '/'
		weights_file_path = env_data_dir + 'dist_model_best'
		info_csv = env_data_dir + 'dist_info.csv'
		args.dist_estimator = Estimator_DistNet(net_weights_path=weights_file_path, csv_dist_filepath=info_csv)
		return
	else:
		raise Exception('logic for dist estimator type not implemented yet')

	size_goal_box = np.array([0., 0.])
	counter = 0
	for rs in range(5):
		if args.dist_estimator_type == 'normal' or args.dist_estimator_type == 'multiple':
			obstacle_latents = []
			obstacle_size_latents = []
		elif args.dist_estimator_type == 'realCoords' or args.dist_estimator_type == 'multipleReal':
			obstacle_real = []
		else:
			raise Exception('logic for dist estimator type not implemented yet')

		env.reset()
		obs = env.get_obs()
		if args.dist_estimator_type == 'normal' or args.dist_estimator_type == 'multiple':
			obstacle_latents.append(obs['obstacle_latent'].copy())
			obstacle_size_latents.append(obs['obstacle_size_latent'].copy())
		elif args.dist_estimator_type == 'realCoords' or args.dist_estimator_type == 'multipleReal':
			obstacle_real.append(obs['real_obstacle_info'])
		else:
			raise Exception('logic for dist estimator type not implemented yet')


		for timestep in range(args.timesteps):
			# get action from the ddpg policy
			action = env.action_space.sample()
			obs, _, _, _ = env.step(action)
			if args.dist_estimator_type == 'normal' or args.dist_estimator_type == 'multiple':
				obstacle_latents.append(obs['obstacle_latent'].copy())
				obstacle_size_latents.append(obs['obstacle_size_latent'].copy())
				if obs['achieved_goal_latent'][0] != 100.:
					size_goal_box += obs['achieved_goal_size_latent']
					counter +=1
			elif args.dist_estimator_type == 'realCoords' or args.dist_estimator_type == 'multipleReal':
				obstacle_real.append(obs['real_obstacle_info'])
				size_goal_box += obs['real_size_goal'][:2]
				counter +=1
			else:
				raise Exception('logic for dist estimator type not implemented yet')



		if args.dist_estimator_type == 'normal' or args.dist_estimator_type == 'multiple':
			args.dist_estimator.update(obstacle_latents, obstacle_size_latents)
		elif args.dist_estimator_type == 'realCoords' or args.dist_estimator_type == 'multipleReal':
			args.dist_estimator.update(obstacle_real, [])
		else:
			raise Exception('logic for dist estimator type not implemented yet')

		#since this are just randomly not increase
		args.dist_estimator.update_calls = 0
	size_goal_box /= counter
	if args.dist_estimator_type == 'normal' or args.dist_estimator_type == 'realCoords' or args.dist_estimator_type == 'multiple' or args.dist_estimator_type == 'multipleReal':
		n_ve = 100
		args.dist_estimator.initialize_internal_distance_graph([args.field_center[0], args.field_center[1],
																args.field_size[0], args.field_size[1]],
															   num_vertices=[n_ve, n_ve], size_increase=0)#size_goal_box[0])#todo use real or other depending of va
		args.dist_estimator.graph.plot_graph(save_path='env_graph_created', elev=90, azim=0)
	plt.clf()


def experiment_setup(args):
	if args.vae_dist_help:
		load_vaes(args)

	#since some extensions of the envs use the distestimator this load is used with the interval wrapper#todo use other?
	load_field_parameters(args)
	if args.dist_estimator_type is not None:
		temp_env = make_temp_env(args)
		load_dist_estimator(args, temp_env)
		del temp_env


	env = make_env(args)
	env_test = make_env(args)

	#rgb_array = take_env_image(env, args.img_size)
	#img = Image.fromarray(rgb_array)
	#img.show()
	#img.close()

	if args.goal_based:

		args.obs_dims = list(goal_based_process(env.reset()).shape)
		args.acts_dims = [env.action_space.shape[0]]
		args.compute_reward = env.compute_reward
		args.compute_distance = env.compute_distance

	if args.imaginary_obstacle_transitions:
		#relative small buffer size so it always have most recent collisions
		args.train_every = 5
		args.train_every_counter = 0
		args.normalizer_every = 5
		args.normalizer_every_counter = 0
		args.imaginary_buffer = ReplayBuffer_Imaginary(args, buffer_size=200)
	args.buffer = buffer = ReplayBuffer_Episodic(args)
	args.learner = learner = create_learner(args)
	args.agent = agent = create_agent(args)
	args.logger.info('*** network initialization complete ***')
	args.tester = tester = Tester(args)
	args.logger.info('*** tester initialization complete ***')
	args.timesteps = env.env.env.spec.max_episode_steps


	return env, env_test, agent, buffer, learner, tester


def experiment_setup_test(args):

	if args.vae_dist_help:
		load_vaes(args)

	load_field_parameters(args)
	if args.dist_estimator_type is not None:
		temp_env = make_temp_env(args)
		load_dist_estimator(args, temp_env)
		del temp_env
	env = make_env(args)


	if args.goal_based:
		args.obs_dims = list(goal_based_process(env.reset()).shape)
		args.acts_dims = [env.action_space.shape[0]]
		args.compute_reward = env.compute_reward
		args.compute_distance = env.compute_distance

	from play import Player
	args.agent = agent = Player(args)
	args.tester = tester = Tester(args)
	args.timesteps = env.env.env.spec.max_episode_steps


	return env, agent, tester