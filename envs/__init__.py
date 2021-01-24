import gym
import envs.fetch as fetch_env
import envs.hand as hand_env
from .utils import goal_distance, goal_distance_obs

Robotics_envs_id = [
	'FetchReach-v1',
	'FetchPush-v1',
	'FetchSlide-v1',
	'FetchSlideMovingObstacle-v1',
	'FetchPickAndPlace-v1',
	'FetchPushNew-v1',
	'FetchCurling-v1',
	'FetchPushObstacle-v1',
	'FetchPushObstacleFetchEnv-v1',
	'FetchPushMovingObstacleEnv-v1',
	'FetchPushMovingObstacleEnv-v2',
	'FetchPushMovingObstacleEnv-v3',
	'FetchPushMovingDoubleObstacleEnv-v1',
	'FetchPushMovingDoubleObstacleEnv-v2',
	'FetchPushMovingDoubleObstacleEnv-v3',
	'FetchTwinkleObstacleEnv-v1',
	'FetchPushMovingComEnv-v1',
	'FetchPushMovingComEnv-v2',
	'FetchPushMovingComEnv-v3',
	'FetchPushMovingComEnv-v4',
	'FetchGenerativeEnv-v1',
	'FetchPickObstacle-v1',
	'FetchPushNoObstacle-v1',
	'FetchPickNoObstacle-v1',
	'FetchPushLabyrinth-v1',
	'FetchPickAndThrow-v1',
	'FetchPickAndSort-v1',
	'HandManipulateBlock-v0',
	'HandManipulateEgg-v0',
	'HandManipulatePen-v0',
	'HandReach-v0'
]

def make_env(args):
	assert args.env in Robotics_envs_id
	if args.env[:5]=='Fetch':
		return fetch_env.make_env(args)
	else: # Hand envs
		return hand_env.make_env(args)

def make_temp_env(args):
	assert args.env in Robotics_envs_id
	if args.env[:5] == 'Fetch':
		return fetch_env.make_temp_env(args)

def clip_return_range(args):
	gamma_sum_min = args.reward_min/(1.0-args.gamma)
	gamma_sum_max = args.reward_max / (1.0 - args.gamma)
	return {
		'FetchReach-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPush-v1': (gamma_sum_min, gamma_sum_max),
		'FetchSlide-v1': (gamma_sum_min, gamma_sum_max),
		'FetchSlideMovingObstacle-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPickAndPlace-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPickObstacle-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPickNoObstacle-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPushObstacle-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPushObstacleFetchEnv-v1':(gamma_sum_min, gamma_sum_max),
		'FetchPushMovingObstacleEnv-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingObstacleEnv-v2': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingObstacleEnv-v3': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingDoubleObstacleEnv-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingDoubleObstacleEnv-v2': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingDoubleObstacleEnv-v3': (gamma_sum_min, gamma_sum_max),
		'FetchTwinkleObstacleEnv-v1':(gamma_sum_min, gamma_sum_max),
		'FetchPushMovingComEnv-v1':(gamma_sum_min, gamma_sum_max),
		'FetchPushMovingComEnv-v2': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingComEnv-v3': (gamma_sum_min, gamma_sum_max),
		'FetchPushMovingComEnv-v4': (gamma_sum_min, gamma_sum_max),
		'FetchGenerativeEnv-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPushLabyrinth-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPickAndThrow-v1': (gamma_sum_min, gamma_sum_max),
		'FetchPickAndSort-v1': (gamma_sum_min, gamma_sum_max),
		'HandManipulateBlock-v0': (gamma_sum_min, gamma_sum_max),
		'HandManipulateEgg-v0': (gamma_sum_min, gamma_sum_max),
		'HandManipulatePen-v0': (gamma_sum_min, gamma_sum_max),
		'HandReach-v0': (gamma_sum_min, gamma_sum_max)
	}[args.env]