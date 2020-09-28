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
    for timestep in range(200):
        action = env.action_space.sample()
        o, _, _, info = env.step(action)
        print(o['obstacle_latent'])
        print(o['obstacle_size_latent'])
        obs.append(o)
        env_images.append(take_env_image(env, args.img_size))
    create_rollout_video(np.array(env_images), args=args, filename='vid_env')
    dist_estimator.update([o['obstacle_latent'] for o in obs], [o['obstacle_size_latent'] for o in obs])


    obs_x_pos = np.array([o['obstacle_latent'][0] for o in obs])
    obs_y_pos = np.array([o['obstacle_latent'][1] for o in obs])
    obs_size_pos = np.array([o['obstacle_size_latent'] for o in obs])

    '''print(obs_x_pos)
    print(obs_y_pos)
    print(obs_size_pos)
    s = np.mean(obs_size_pos)
    print(s)
    plt.clf()'''


    t = np.arange(len(obs_x_pos))
    plt.plot(t, obs_x_pos, label='x')
    plt.plot(t, obs_y_pos, label='y')
    plt.plot(t, obs_size_pos, label='size')

    plt.xlabel('Iteration')
    plt.ylabel('measure')
    plt.legend()
    plt.savefig('measures.png')
    plt.close()

    #dist_estimator.update([[-0.368, 0], [0.368, 0]],[0.192])
    goal_pos = np.array([0, -0.9])
    positions = np.array([[-1.,0.2],[-0.85,0.2],[-0.2,0.2],[0.2,0.2],[1,0.2]])
    distances = dist_estimator.calculate_distance_batch(goal_pos, positions)
    print('hi')
    print(distances)
    print('----')

    a = np.array([1.3, 0.52])
    b = np.array([[1.02, 0.6], [1.11, 0.6], [1.18, 0.6], [1.3, 0.6],
                  [1.38, 0.6], [1.49, 0.6], [1.54, 0.6]])
    update_points = [[1.18, 0.75], [1.38, 0.75]]
    dist2= DistMovEst()
    dist2.update(update_points, [0.048])
    real_ds = dist2.calculate_distance_batch(a,b)
    print(real_ds)
    direct_real_ds = np.linalg.norm(b - a, axis=1)
    print(direct_real_ds)
    print('----')

    from j_vae.latent_space_transformations import table_map
    '''la = table_map(a)
    lb = table_map(b)
    l_update_points = table_map(update_points)
    dist3 = DistMovEst()
    dist3.update(l_update_points, [4*0.048])
    l_dists = dist3.calculate_distance_batch(la, lb)
    print(l_dists)
    l_dir_dists = np.linalg.norm(lb - la, axis=1)
    print(l_dists)
    print('----')

    data_set = np.empty([1 + len(b), 84, 84, 3])
    from vae_env_inter import goal_latent_from_images, take_goal_image

    i = 0
    env.env.env._move_object(position=np.array([a[0], a[1], 0.425]))
    data_set[i] = take_goal_image(env, args.img_size, make_table_invisible=False)
    i+=1
    for p in b:
        env.env.env._move_object(position=np.array([p[0], p[1], 0.425]))
        data_set[i] = take_goal_image(env, args.img_size, make_table_invisible=False)
        i+=1
    all_l = goal_latent_from_images(data_set, args)
    est_dists = dist_estimator.calculate_distance_batch(all_l[0], all_l[1:])
    print(est_dists)
    direct_all = np.linalg.norm(all_l[1:] - all_l[0], axis=1)
    print(direct_all)'''


