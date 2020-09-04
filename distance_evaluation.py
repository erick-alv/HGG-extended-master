from custom_start import get_args_and_initialize
from vae.vae_torch import setup_distance_ae_and_trainer
from envs import make_env
import numpy as np
import csv
from utils.image_util import make_text_im, create_black_im_of_dims, stack_images_column_2, \
    stack_images_row, store_image_array_at, fill_width, make_video
import os
import io
from utils.transforms_utils import extend_obs
from algorithm.replay_buffer import Trajectory
from sklearn.linear_model import LinearRegression
from agents.distance_estimator import DistanceEstimator
import pickle
from utils.image_util import rgb_array_to_image
import time
import matplotlib.pyplot as plt
from utils.transforms_utils import BA
from algorithm.replay_buffer import ReplayBuffer_Episodic
import numpy as np
import copy


def l1_dist(a,b):
    return np.linalg.norm(a-b, ord=1)


def l2_dist(a,b):
    return np.linalg.norm(a - b)


def max_dist_component(a,b):
    return np.linalg.norm(a-b, ord=np.inf)


def min_dist_component(a,b):
    return np.linalg.norm(a-b, ord=-np.inf)


def eq_dist(a,b):
    return np.linalg.norm(a-b, ord=0)


def abs_dist_as_vector(a,b):
    return np.abs(a-b)


def distances_to_specific(trajectory, dist_function, from_keyword, specific_keyword, specific_index = None):
    if specific_index is not None:
        specific = trajectory[specific_index][specific_keyword]
    else:
        specific = trajectory[0][specific_keyword]
    l = len(trajectory)
    return np.array([dist_function(trajectory[i][from_keyword], specific) for i in range(l)])


def distances_to_next(trajectory, dist_function, from_keyword, to_keyword):
    '''
    :param trajectory:
    :param from_keyword: from which type
    :param to_keyword: to which type
    :param dist_function: function to estimate the distance mist support the inputs returned by the trajectory with from
    keyword and to_keword
    :return: the distance to next step as well the cummulated distance from each step to end
    '''
    l = len(trajectory)
    to_next = np.array([dist_function(trajectory[i][from_keyword], trajectory[i + 1][to_keyword])
                        for i in range(l - 1)] + [dist_function(trajectory[l - 1][from_keyword],
                                                               trajectory[l - 1][to_keyword])])
    return to_next, np.flip(np.cumsum(np.flip(to_next)))


def process_distances(trajectory, dist_estimator, dist_estimator2, args):
    if dist_estimator.observation_type == 'latent':
        from_keyword = 'state_latent'
    elif dist_estimator.observation_type == 'real':
        from_keyword = 'observation'
    elif dist_estimator.observation_type == 'concat':
        raise Exception('Not implemented yet')
    else:
        raise Exception('observation type not valid')

    if dist_estimator.goal_type == 'latent':
        goal_keyword = 'goal_latent'
        last_keyword = 'state_latent'
        to_keyword = 'state_latent'
    elif dist_estimator.goal_type == 'goal_space':
        goal_keyword = 'desired_goal'
        last_keyword = trajectory[-1]['achieved_goal']
        to_keyword = 'achieved_goal'
    elif dist_estimator.goal_type == 'state':
        goal_keyword = 'goal_state'
        last_keyword = 'observation'
        to_keyword = 'observation'
    elif dist_estimator.goal_type == 'concat':
        raise Exception('Not implemented yet')
    else:
        raise Exception('goal obs type not valid')
    d = {}
    d['to_goal_l2'] = distances_to_specific(trajectory, args.goal_distance, 'achieved_goal', 'desired_goal')
    d['to_goal_est'] = distances_to_specific(trajectory, dist_estimator.estimate_distance, from_keyword, goal_keyword)
    d['to_goal_2est'] = distances_to_specific(trajectory, dist_estimator2.estimate_distance, from_keyword, goal_keyword)
    d['to_last_l2'] = distances_to_specific(trajectory, args.goal_distance, 'achieved_goal', 'achieved_goal', -1)
    d['to_last_est'] = distances_to_specific(trajectory, dist_estimator.estimate_distance, from_keyword,
                                             last_keyword, -1)
    d['to_last_2est'] = distances_to_specific(trajectory, dist_estimator2.estimate_distance, from_keyword,
                                              last_keyword, -1)
    l = len(trajectory)
    d['to_last_steps'] = np.flip(np.arange(0, l), 0)

    d['to_next_l2'], d['to_last_l2_traj'] = distances_to_next(trajectory, args.goal_distance,
                                                              'achieved_goal', 'achieved_goal')

    d['to_next_est'], d['to_last_est_traj'] = distances_to_next(trajectory, dist_estimator.estimate_distance,
                                                                from_keyword, to_keyword)

    d['to_next_2est'], d['to_last_2est_traj'] = distances_to_next(trajectory, dist_estimator2.estimate_distance,
                                                                  from_keyword, to_keyword)
    return d, [['to_goal_l2', 'to_goal_est','to_goal_2est'],
               ['to_next_l2', 'to_next_est', 'to_next_2est'],
               ['to_last_l2','to_last_est', 'to_last_2est'],
               ['to_last_steps'],
               ['to_last_l2_traj', 'to_last_est_traj', 'to_last_2est_traj']]


def process_distances_k_f_list(trajectory, k_f_list, k_f_traj_list):
    d = {}
    for from_keyword, specific_keyword, index, f in k_f_list:
        if isinstance(f, tuple):
            f_name = f[1]
            f = f[0]
        else:
            f_name = f.__name__
        d['from_'+from_keyword+'_to_'+specific_keyword+'_'+f_name] = distances_to_specific(trajectory, f,
                                                                                               from_keyword,
                                                                                               specific_keyword,
                                                                                               specific_index=index)

    for from_keyword, to_keyword, f in k_f_traj_list:
        if isinstance(f, tuple):
            f_name = f[1]
            f = f[0]
        else:
            f_name = f.__name__
        d[to_keyword+'_to_next_'+f_name], d[to_keyword+'_'+f_name+'_traj'] = distances_to_next(trajectory, f,
                                                                                                       from_keyword,
                                                                                                       to_keyword)
    return d


def distances_against(vectors_a, vectors_b, dist_functions_list):
    d = {}
    l = len(vectors_a)
    for f in dist_functions_list:
        if isinstance(f, tuple):
            f_name = f[1]
            f = f[0]
        else:
            f_name = f.__name__
        d[f_name] = np.array([f(vectors_a[i], vectors_b[i]) for i in range(l)])
    return d


def create_rollout_video_with_distances(args, trajectory, distances_images, filename):
    goal = trajectory[0]['goal_image']
    last_state = trajectory[-1]['state_image']
    for i in range(len(distances_images)):
        ims = stack_images_row([trajectory[i]['state_image'], last_state, goal])
        s, d = fill_width(ims, distances_images[i])
        st = stack_images_column_2([s,d])
        store_image_array_at(st, args.dirpath + 'temp/', 'frame_{}'.format(i))
    make_video(args.dirpath+'temp/', '.png', path_to_save=args.dirpath+'distances_videos/', filename_save=filename)
    for f in os.listdir(args.dirpath+'temp/'):
        if f.endswith('.png'):
            os.remove(args.dirpath+'temp/'+f)


def create_rollout_video(args, trajectory, filename):
    goal = trajectory[0]['goal_image']
    last_state = trajectory[-1]['state_image']
    for i in range(len(trajectory)):
        ims = stack_images_row([trajectory[i]['state_image'], last_state, goal])
        store_image_array_at(ims, args.dirpath + 'temp/', 'frame_{}'.format(i))
    make_video(args.dirpath+'temp/', '.png', path_to_save=args.dirpath+'distances_videos/', filename_save=filename)
    for f in os.listdir(args.dirpath+'temp/'):
        if f.endswith('.png'):
            os.remove(args.dirpath+'temp/'+f)

class TrajectoryBufferForVid:
    def __init__(self):
        pass

    def store_trajectory(self, trajectory):
        self.buffer = trajectory.ep['obs']

    def get_trajectory(self):
        return self.buffer


def create_dist_images(distances_dict, keys_order, args):
    def make_im(label, distance):
        return make_text_im(90, 15, '{}:\n{:.12f}'.format(label, distance))
    max_g = max([len(gr) for gr in keys_order])
    for k in distances_dict.keys():
        l = len(distances_dict[k])
        break
    text_ims = []
    for i in range(l):
        grouped_ims = []
        for group in keys_order:
            ims = [make_im(label, distances_dict[label][i]) for label in group]
            for _ in range(max_g-len(ims)):
                ims.append(create_black_im_of_dims(90, 30))
            im = stack_images_row(ims)
            grouped_ims.append(im)
        it_im = stack_images_column_2(grouped_ims)
        text_ims.append(it_im)
    return text_ims


def write_distances_to_csv(distances_dict, filename, args):
    path = args.dirpath + 'csv_logs/'+filename+'.csv'
    fieldnames = []
    for k in distances_dict.keys():
        fieldnames.append(k)
        l = len(distances_dict[k])
    with open(path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(l):
            row = {}
            for k in fieldnames:
                val = distances_dict[k][i]
                if isinstance(val, np.ndarray):
                    out = io.StringIO()
                    np.savetxt(out, val)
                    val = out.getvalue()
                row[k] = val
            writer.writerow(row)


def setup(args, recover_filename=None):
    env = make_env(args)
    args.compute_reward = env.env.env.compute_reward
    args.goal_distance = env.env.env.goal_distance
    #VAE
    ae, trainer_ae = setup_distance_ae_and_trainer(args, recover_filename=args.ae_recover_filename, compat_extra=True)
    #currently ranges are not used
    return env, ae, trainer_ae


def create_points(x, y, interval_starts, steps, from_right_to_left=False):
    step_size = (x[-1] - x[0]) / steps
    xs = np.arange(start=x[0], stop=x[-1], step=step_size).reshape((-1, 1))
    ys = []
    for i in range(len(interval_starts)):
        if i == len(interval_starts)-1:
            a = interval_starts[i]
            b = steps-1
        else:
            a = interval_starts[i]
            b = interval_starts[i+1]
        x_inter = np.array([xs[a], xs[b]]).reshape((-1, 1))
        y_inter = np.array([y[i], y[i+1]])
        model = LinearRegression()
        model.fit(x_inter, y_inter)
        if i == len(interval_starts)-1:
            ys_i = model.predict(xs[a:, :])
        else:
            ys_i = model.predict(xs[a:b,:])
        ys.append(ys_i)

    ys = np.concatenate(ys)
    xs = xs.reshape((-1))
    assert len(xs)==steps
    assert len(ys)==steps
    if from_right_to_left:
        return np.flip(xs), np.flip(ys)
    return xs, ys


def follow_points_with_goals(x_points_obj, y_points_obj, x_points_gr, y_points_gr, env, ae, args, x_goal=1.3, y_goal=0.65):
    ae.eval()
    env.reset()
    env.env.env.set_goal(x_goal, y_goal)
    #move objects to start position
    env.env.env.move_gripper_to(x_points_gr[0], y_points_gr[0])
    env.env.env.move_object_to(x_points_obj[0], y_points_obj[0], 0.45)
    obs = env.get_obs()
    obs = extend_obs(ae, obs, args)
    current = Trajectory(obs)
    buffer_args = BA('none', 1)
    buffer = ReplayBuffer_Episodic(buffer_args)
    for i in range(len(x_points_obj)):
        env.env.env.move_object_to(x_points_obj[i], y_points_obj[i],0.45)
        env.env.env.move_gripper_to(x_points_gr[i], y_points_gr[i])
        next_obs = env.get_obs()
        next_obs = extend_obs(ae, next_obs, args)
        current.store_step(np.array([]), next_obs, np.array([]), np.array([]))
        if i == len(x_points_obj) - 1:
            buffer.store_trajectory(current)
            break
        obs = next_obs
    trajectory = buffer.buffer['obs'][0]
    return trajectory


def create_routes(env, ae, args):
    def save_tr_and_video(tr, filename):
        with open(args.dirpath + 'distances_videos/' + filename, 'wb') as f:
            pickle.dump(tr, f)
        create_rollout_video(args, trajectory=tr, filename=filename)

    obs = env.reset()
    obs = extend_obs(ae, obs, args)
    current = Trajectory(obs)
    buffer_args = BA('none', 1)
    buffer = ReplayBuffer_Episodic(buffer_args)
    for step in range(50):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(ae, next_obs, args)
        current.store_step(action, next_obs, reward, done)
        if done or step == 50 - 1:
            buffer.store_trajectory(current)
            break
        obs = next_obs
    trajectory = buffer.buffer['obs'][0]
    save_tr_and_video(trajectory, 'distance_evaluation_random')
    ########################################################
    xs_o, ys_o = create_points([0.5, 1.0], [1.25, 1.25, 1.2, 1.1, 1.2, 1.25, 1.25],
                               [0, 10, 15, 25, 35, 40], 50, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = np.concatenate([np.repeat(1.1, 5), np.repeat(1.05, 5), xs_gr[:-10]])
    ys_gr = np.concatenate([np.repeat(1.25, 10), ys_gr[:-10]])
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are fipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args, y_goal=0.45)
    save_tr_and_video(trajectory, 'distance_evaluation_route_1')
    ##########################################################
    xs_o, ys_o = create_points([0.5, 1.0], [1.1, 1.10], [0], 50, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = xs_gr + 0.1
    ys_gr = np.concatenate([np.repeat(1.10, 10), ys_gr[:-10]])
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are fipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args, y_goal=0.45)
    save_tr_and_video(trajectory, 'distance_evaluation_route_2')
    ###########################################################
    xs_o, ys_o = create_points([0.82, 1.0], [1.1, 1.1], [0], 25, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = xs_gr + 0.1
    ys_gr = np.concatenate([np.repeat(1.1, 8), ys_gr[:-8]])
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are fipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args)
    save_tr_and_video(trajectory, 'distance_evaluation_route_3_1')
    ################################################################
    xs_o, ys_o = create_points([0.82, 1.0], [1.3, 1.3], [0], 25, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = xs_gr + 0.1
    ys_gr = np.concatenate([np.repeat(1.3, 8), ys_gr[:-8]])
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are fipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args)
    save_tr_and_video(trajectory, 'distance_evaluation_route_3_2')
    #################################################################################
    xs_o, ys_o = create_points([0.8, 1.0], [1.15, 1.3], [0], 25, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = xs_gr + 0.1
    ys_gr = ys_gr + 0.1
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are fipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args)
    save_tr_and_video(trajectory, 'distance_evaluation_route_3_3')
    ########################################################################
    xs_o, ys_o = create_points([0.8, 1.0], [1.15, 1.15], [0], 10, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = xs_gr + 0.1
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are fipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args, x_goal=1.2, y_goal=0.65)
    save_tr_and_video(trajectory, 'distance_evaluation_route_4_1')
    ###########################################################################
    xs_o, ys_o = create_points([0.8, 1.0], [1.25, 1.25], [0], 10, from_right_to_left=True)
    xs_gr, ys_gr = xs_o.copy(), ys_o.copy()
    xs_gr = xs_gr + 0.1
    xs_o, ys_o = ys_o, xs_o  # this is just because axis are flipped in env
    xs_gr, ys_gr = ys_gr, xs_gr
    trajectory = follow_points_with_goals(xs_o, ys_o, xs_gr, ys_gr, env, ae, args, x_goal=1.2, y_goal=0.65)
    save_tr_and_video(trajectory, 'distance_evaluation_route_4_2')

def create_groups(env, ae, args):
    def save_tr_and_video(tr_1,tr_2, filename):
        with open(args.dirpath + 'distances_videos/' + filename+'_1', 'wb') as f:
            pickle.dump(tr_1, f)
        with open(args.dirpath + 'distances_videos/' + filename+'_2', 'wb') as f:
            pickle.dump(tr_2, f)
        create_rollout_video(args, trajectory=tr_1, filename=filename+'_1')
        create_rollout_video(args, trajectory=tr_2, filename=filename + '_2')
    trajectory1 = follow_points_with_goals([1.15] * 5, [0.5, 0.6, 0.7, 0.8, 0.9],
                                           [1.05] * 5, [0.5, 0.6, 0.7, 0.8, 0.9],
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.45] * 5, [0.5, 0.6, 0.7, 0.8, 0.9],
                                           [1.35] * 5, [0.5, 0.6, 0.7, 0.8, 0.9],
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_a')
    ##
    trajectory1 = follow_points_with_goals([1.3, 1.5, 1.5, 1.3], [0.6, 0.6, 0.9, 0.9],
                                           [1.3, 1.5, 1.5, 1.3], [0.6 - 0.1, 0.6 - 0.1, 0.9 + 0.1, 0.9 + 0.1],
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.3, 1.3, 1.5, 1.5], [0.9, 0.6, 0.6, 0.9],
                                           [1.3, 1.3, 1.5, 1.5], [0.9 + 0.1, 0.6 - 0.1, 0.6 - 0.1, 0.9 + 0.1],
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_b')
    ##
    trajectory1 = follow_points_with_goals([1.1, 1.3, 1.5], [0.7] * 3,
                                           [1.1, 1.3, 1.5], [0.6] * 3,
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.1, 1.3, 1.5], [0.8] * 3,
                                           [1.1, 1.3, 1.5], [0.9] * 3,
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_c')
    ##
    trajectory1 = follow_points_with_goals([1.1, 1.3, 1.5], [1.0] * 3,
                                           [1.1, 1.3, 1.5], [1.1] * 3,
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.1, 1.3, 1.5], [0.9] * 3,
                                           [1.1, 1.3, 1.5], [0.8] * 3,
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_d')
    ##
    trajectory1 = follow_points_with_goals([1.1, 1.3, 1.5], [0.6] * 3,
                                           [1.1, 1.3, 1.5], [0.7] * 3,
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.1, 1.3, 1.5], [0.5] * 3,
                                           [1.1, 1.3, 1.5], [0.4] * 3,
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_e')
    ##
    trajectory1 = follow_points_with_goals([1.45, 1.35, 1.25, 1.15, 1.08], [0.5, 0.6, 0.7, 0.8, 0.9],
                                           [1.45 - 0.1, 1.35 - 0.1, 1.25 - 0.1, 1.15 - 0.1, 1.05 - 0.1],
                                           [0.5, 0.6, 0.7, 0.8, 0.9],
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.08, 1.15, 1.25, 1.35, 1.45], [0.9, 0.8, 0.7, 0.6, 0.5],
                                           [1.05 - 0.1, 1.15 - 0.1, 1.25 - 0.1, 1.35 - 0.1, 1.45 - 0.1],
                                           [0.9, 0.8, 0.7, 0.6, 0.5],
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_f')
    #
    trajectory1 = follow_points_with_goals([1.2] * 5, [1.0, 0.9, 0.8, 0.7, 0.6],
                                           [1.2] * 5, [1.0 + 0.1, 0.9 + 0.1, 0.8 + 0.1, 0.7 + 0.1, 0.6 + 0.1],
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.2] * 5, [1.0 - 0.1, 0.9 - 0.1, 0.8 - 0.1, 0.7 - 0.1, 0.6 - 0.1],
                                           [1.2] * 5, [1.0, 0.9, 0.8, 0.7, 0.6],
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_g')
    #
    trajectory1 = follow_points_with_goals([1.3] * 5, [1.0, 0.9, 0.8, 0.7, 0.6],
                                           [1.3] * 5, [1.0 + 0.1, 0.9 + 0.1, 0.8 + 0.1, 0.7 + 0.1, 0.6 + 0.1],
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.3] * 5, [1.0 - 0.1, 0.9 - 0.1, 0.8 - 0.1, 0.7 - 0.1, 0.6 - 0.1],
                                           [1.3] * 5, [1.0, 0.9, 0.8, 0.7, 0.6],
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_h')
    #
    trajectory1 = follow_points_with_goals([1.4] * 5, [1.0, 0.9, 0.8, 0.7, 0.6],
                                           [1.4] * 5, [1.0 + 0.1, 0.9 + 0.1, 0.8 + 0.1, 0.7 + 0.1, 0.6 + 0.1],
                                           env, ae, args)
    trajectory2 = follow_points_with_goals([1.4] * 5, [1.0 - 0.1, 0.9 - 0.1, 0.8 - 0.1, 0.7 - 0.1, 0.6 - 0.1],
                                           [1.4] * 5, [1.0, 0.9, 0.8, 0.7, 0.6],
                                           env, ae, args)
    save_tr_and_video(trajectory1, trajectory2, 'group_i')


if __name__ == "__main__":
    args = get_args_and_initialize()
    env, ae, trainer_ae = setup(args)
    ae.eval()
    state_dim = 10
    goal_dim = 10

    #create_routes(env, ae, args)
    #create_groups(env,ae,args)
    ###########################################################################
    epoch=390
    dist_estimator = DistanceEstimator(output_dim=1, state_dim=state_dim, goal_dim=goal_dim, min_range=[0],
                                       max_range=[200], observation_type=args.observation_type,
                                       goal_type=args.goal_type, args=args)
    dist_estimator.load(filename='dist_estimator_simplified_{}'.format(epoch))
    dist_estimator2 = DistanceEstimator(output_dim=1, state_dim=state_dim, goal_dim=goal_dim, min_range=[0],
                                        max_range=[200], observation_type=args.observation_type,
                                        goal_type=args.goal_type, args=args)
    dist_estimator2.load(filename='dist_estimator2_simplified_{}'.format(epoch))
    #
    to_goal = [('achieved_goal', 'desired_goal', 0,f) for
                       f in [l2_dist, args.goal_distance]]
    to_goal_latent = [('state_latent', 'goal_latent', 0, f) for
                      f in [l2_dist, l1_dist,(dist_estimator.estimate_distance, 'estimator1'),
                            (dist_estimator2.estimate_distance, 'estimator2')]]
    to_last = [('achieved_goal', 'achieved_goal', -1,f) for
               f in [l2_dist, l1_dist, args.goal_distance]]
    to_last_latent = [('state_latent', 'state_latent', -1, f) for
                      f in [l2_dist, l1_dist, (dist_estimator.estimate_distance, 'estimator1'),
                            (dist_estimator2.estimate_distance, 'estimator2')]]
    to_last_traj = [('achieved_goal', 'achieved_goal', f) for
                    f in [l2_dist, l1_dist, abs_dist_as_vector, max_dist_component, args.goal_distance]]
    to_last_traj_latent = [('state_latent', 'state_latent', f) for
                           f in [l2_dist, l1_dist, (dist_estimator.estimate_distance, 'estimator1'),
                                 (dist_estimator2.estimate_distance, 'estimator2')]]

    def read_trajectory_to_csv(filename):
        with open(args.dirpath + 'distances_videos/'+filename, 'rb') as f:
            tr = pickle.load(f)
            distances_dict = process_distances_k_f_list(tr, to_goal + to_goal_latent + to_last + to_last_latent,
                                                       to_last_traj + to_last_traj_latent)
            l = len(tr)
            distances_dict.update({'to_last_steps': np.flip(np.arange(0, l), 0)})
            write_distances_to_csv(distances_dict, filename, args)

    l = ['distance_evaluation_random','distance_evaluation_route_1','distance_evaluation_route_2',
              'distance_evaluation_route_3_1','distance_evaluation_route_3_2','distance_evaluation_route_3_3',
              'distance_evaluation_route_4_1','distance_evaluation_route_4_2']
    for r in l:
        read_trajectory_to_csv(r)


    dist_funcs = [l2_dist]
    dist_funcs_latent = [(dist_estimator.estimate_distance, 'estimator1'),
                         (dist_estimator2.estimate_distance, 'estimator2')]
    def evl(filename):
        with open(args.dirpath + 'distances_videos/' + filename+'_1', 'rb') as f1:
            tr_1 = pickle.load(f1)
            with open(args.dirpath + 'distances_videos/' + filename+'_2', 'rb') as f2:
                tr_2 = pickle.load(f2)
                
                as_v = [s['achieved_goal'] for s in tr_1[1:]]
                as_z = [s['state_latent'] for s in tr_1[1:]]
                bs_v = [s['achieved_goal'] for s in tr_2[1:]]
                bs_z = [s['state_latent'] for s in tr_2[1:]]
                d1 = distances_against(as_v, bs_v, dist_funcs)
                d2 = distances_against(as_z, bs_z, dist_funcs_latent)
                d1.update(d2)
                write_distances_to_csv(d1, filename, args)

    for gr in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
        evl('group_'+gr)



