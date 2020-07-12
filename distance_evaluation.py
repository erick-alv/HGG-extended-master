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
        d['to_'+specific_keyword+'_'+f.__name__] = distances_to_specific(trajectory, f, from_keyword,
                                                                         specific_keyword, specific_index=index)

    for from_keyword, to_keyword, f in k_f_traj_list:
        d['to_next_'+f.__name__], d[f.__name__+'_traj'] = distances_to_next(trajectory, f, from_keyword, to_keyword)
    return d

def distances_against(vectors_a, vectors_b, dist_functions_list):
    d = {}
    l = len(vectors_a)
    for f in dist_functions_list:
        d[f.__name__] = np.array([f(vectors_a[i], vectors_b[i]) for i in range(l)])
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
        #s, d = fill_width(ims, distances_images[i])
        #st = stack_images_column_2([s,d])
        store_image_array_at(ims, args.dirpath + 'temp/', 'frame_{}'.format(i))
    make_video(args.dirpath+'temp/', '.png', path_to_save=args.dirpath+'distances_videos/', filename_save=filename)
    for f in os.listdir(args.dirpath+'temp/'):
        if f.endswith('.png'):
            os.remove(args.dirpath+'temp/'+f)


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
    args.compute_reward = env.compute_reward
    #VAE
    ae, trainer_ae = setup_distance_ae_and_trainer(args, recover_filename=args.ae_recover_filename)
    #currently ranges are not used
    return env, ae, trainer_ae

from sklearn.linear_model import LinearRegression
def create_points(x, y, interval_starts, steps, from_right_to_left=False):
    step_size = (x[-1] - x[0]) / steps
    xs = np.arange(start=x[0], stop=x[-1], step=step_size).reshape((-1, 1))
    ys =[]
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

def follow_points(x_points, y_points, env, ae):
    #set fixed goal
    env.reset()
    env.e

import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    args = get_args_and_initialize()
    env, ae, trainer_ae = setup(args)
    ae.eval()
    obs = env.reset()
    obs = extend_obs(ae, obs, args)
    current = Trajectory(obs)
    for step in range(100):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(ae, next_obs, args)
        current.store_step(action, next_obs, reward, done)
        if done or step == args.env_steps - 1:
            break
        obs = next_obs
    create_rollout_video(args, trajectory=current, filename='distance_evaluation_random')
    '''x_points, y_points = create_points([0.5, 1.0],
                                       [1.3, 1.3,1.2, 1.1,1.2,1.3,1.3],
                                       [0,35,40,50,60, 65],
                                       100)
    plt.plot(x_points, y_points, color='blue', linewidth=3)
    plt.show()
    print('hello')'''
