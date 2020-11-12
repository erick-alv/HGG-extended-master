import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir, LoggerExtra
from utils.image_util import create_rollout_video
from j_vae.distance_estimation import calculate_distance, calculate_distance_real
from vae_env_inter import take_env_image, take_objects_image_training, take_image_objects
from envs import make_env
from common import get_args, load_vaes
import torch
import copy
import matplotlib.pyplot as plt
from j_vae.distance_estimation import DistMovEst
import time

'''obs_x_pos = np.array([o['obstacle_latent'][0] for o in obs])
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

    a = np.array([1.3, 0.57])
    b = np.array([[1.05, 0.78], [1.08, 0.8], [1.2, 0.8], [1.4, 0.8],
                  [1.52, 0.8], [1.55, 0.78]])
    update_points = [[1.18, 0.75], [1.38, 0.75]]
    data_set = np.empty([1 + len(b), 84, 84, 3])
    from vae_env_inter import goal_latent_from_images, take_goal_image

    i = 0
    env.env.env._move_object(position=np.array([a[0], a[1], 0.425]))
    data_set[i] = take_goal_image(env, args.img_size, make_table_invisible=False)
    i += 1
    for p in b:
        env.env.env._move_object(position=np.array([p[0], p[1], 0.425]))
        data_set[i] = take_goal_image(env, args.img_size, make_table_invisible=False)
        i += 1
    all_l = goal_latent_from_images(data_set, args)
    est_dists = dist_estimator.calculate_distance_batch(all_l[0], all_l[1:])
    print(est_dists)
    direct_all = np.linalg.norm(all_l[1:] - all_l[0], axis=1)
    print(direct_all)

    dist2= DistMovEst()
    dist2.update(update_points, [0.048])
    real_ds = dist2.calculate_distance_batch(a,b)
    print(real_ds)
    direct_real_ds = np.linalg.norm(b - a, axis=1)
    print(direct_real_ds)
    print('----')

    la = table_map(a)
    lb = table_map(b)
    l_update_points = table_map(update_points)
    dist3 = DistMovEst()
    dist3.update(l_update_points, [4*0.048])
    l_dists = dist3.calculate_distance_batch(la, lb)
    print(l_dists)
    l_dir_dists = np.linalg.norm(lb - la, axis=1)
    print(l_dists)
    print('----')
    '''


if __name__ == '__main__':
    #read args and init env
    args = get_args()
    #load_vaes(args)
    env = make_env(args)
    #dist_estimator = DistMovEst()

    #run env
    acc_sum, obs = 0.0, []
    prev_obs = []
    env_images = []
    start_time = time.time()


    for vid in range(5):
        o = env.reset()
        obs.append(o)
        prev_obs.append(o)
        #env_images.append(take_image_objects(env, args.img_size))
        env_images.append(take_env_image(env, args.img_size))
        actions = [[0., 1., 0., 0.]]*4 + [[1., -1., 0., 0.]]*100
        for timestep in range(100):

            #env.env.env._rotate(["cube"], 0., 10. * timestep, 10. * timestep)
            #env.env.env._rotate(["cylinder"], 0., 10. * timestep, 0.)
            #env.env.env._change_color(["cylinder"], 0.1, 0.1 * timestep, 0.1)

            #action = env.action_space.sample()
            #action = np.array([-1., -1., 0., 0.])
            action = actions[timestep]
            '''if timestep < 8:
                o, _, _, info = env.step([-1., 0, 0., 0.])
            elif timestep < 14:
                o, _, _, info = env.step([0., 1., 0., 0.])
            elif timestep < 40:
                o, _, _, info = env.step([1., -0.5, 0., 0.])
            else:'''
            o, _, _, info = env.step(action)
            #o, _, _, info = env.step(action)
            #print('pos: {}'.format(o['obstacle_latent']))
            #print('size: {}'.format(o['obstacle_size_latent']))
            obs.append(o)
            #env_images.append(take_image_objects(env, args.img_size))
            env_images.append(take_env_image(env, args.img_size))
        create_rollout_video(env_images, args=args, filename='vid_{}_env'.format(vid))
    env_images = np.array(env_images)
    '''with torch.no_grad():
        batch_size = 101
        idx_set = np.arange(len(env_images))
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        for batch_idx, idx_select in enumerate(idx_set):
            data = env_images[idx_select]
            data = torch.from_numpy(data).float().cuda()
            data /= 255
            data = data.permute([0, 3, 1, 2])
            args.vae_model_goal.encode(data)'''

    '''obstacle1_inf = np.array([o['real_obstacle_info'][0] for o in obs])
    obstacle2_inf = np.array([o['real_obstacle_info'][1] for o in obs])
    print("obstacle 1 max x coord: {} min x coord {} \nmax y coord {}, min y coord {}".format(
        np.max(obstacle1_inf[:, 0]), np.min(obstacle1_inf[:, 0]), np.max(obstacle1_inf[:, 1]),
        np.min(obstacle1_inf[:, 1])))
    print("obstacle 2 max x coord: {} min x coord {} \nmax y coord {}, min y coord {}".format(
        np.max(obstacle2_inf[:, 0]), np.min(obstacle2_inf[:, 0]), np.max(obstacle2_inf[:, 1]),
        np.min(obstacle2_inf[:, 1])))'''


    #dist_estimator.update([o['obstacle_latent'] for o in obs], [o['obstacle_size_latent'] for o in obs])
    #dist_estimator.update_sizes([o['obstacle_latent'] for o in obs], [o['achieved_goal_latent'] for o in obs])

    end_time = time.time()
    print('Time transcurred: {}'.format(end_time - start_time))
    np.save('data/FetchGenerativeEnv-v1/double_env.pny', env_images)





