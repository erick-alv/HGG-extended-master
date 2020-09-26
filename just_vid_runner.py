import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir, LoggerExtra
from utils.image_util import create_rollout_video
from j_vae.distance_estimation import calculate_distance, calculate_distance_real
from vae_env_inter import take_env_image
from envs import make_env
from common import get_args, load_vaes
import copy
import matplotlib.pyplot as plt
from j_vae.distance_estimation import DistMovEst


if __name__ == '__main__':
    #read args and init env
    args = get_args()
    load_vaes(args)
    env = make_env(args)
    dist_estimator = DistMovEst()

    #run env

    acc_sum, obs = 0.0, []
    prev_obs = []


    env_images = []
    o = env.reset()
    obs.append(o)
    prev_obs.append(o)
    env_images.append(take_env_image(env, args.img_size))
    for timestep in range(args.timesteps):
        action = env.action_space.sample()
        o, _, _, info = env.step(action)
        obs.append(o)
        env_images.append(take_env_image(env, args.img_size))
    create_rollout_video(np.array(env_images), args=args, filename='vid_env')
    dist_estimator.update([o['obstacle_latent'] for o in obs], [o['obstacle_size_latent'] for o in obs])

    '''dist_estimator.update([[-0.368, 0], [0.368, 0]],[0.192])
    goal_pos = np.array([0, -0.9])
    positions = np.array([[-1.,0.2],[-0.85,0.2],[-0.2,0.2],[0.2,0.2],[1,0.2]])
    distances = dist_estimator.calculate_distance_batch(goal_pos, positions)'''
    obs_x_pos = np.array([o['obstacle_latent'][0] for o in obs])
    obs_y_pos = np.array([o['obstacle_latent'][1] for o in obs])
    obs_size_pos = np.array([o['obstacle_size_latent'] for o in obs])

    print(obs_x_pos)
    print(obs_y_pos)
    print(obs_size_pos)
    s = np.mean(obs_size_pos)
    print(s)
    plt.clf()


    t = np.arange(len(obs_x_pos))
    plt.plot(t, obs_x_pos, label='x')
    plt.plot(t, obs_y_pos, label='y')
    plt.plot(t, obs_size_pos, label='size')

    plt.xlabel('Iteration')
    plt.ylabel('measure')
    plt.legend()
    plt.savefig('measures.png')
    plt.close()
    '''from scipy.stats import norm, truncnorm, uniform

    vals = np.linspace(-1, 1, 100)
    mean_x = np.mean(obs_x_pos)
    std_x = np.std(obs_x_pos)
    vals_pr = norm.pdf(vals, loc=mean_x, scale=std_x)
    plt.plot(vals, vals_pr)

    min_x = np.min(obs_x_pos)
    max_x = np.max(obs_x_pos)
    print(min_x)
    print(max_x)
    print(max_x - min_x)
    print('----------------')

    u1, u2 = uniform.fit(obs_x_pos)
    print(u1)
    print(u2)
    vals_pr_3 = uniform.pdf(vals,loc=min_x,scale=max_x-min_x)
    plt.plot(vals, vals_pr_3)
    plt.show()
    plt.close()'''
    print('hi')
