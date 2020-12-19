import numpy as np
import os
import copy
from utils.os_utils import make_dir
from vae_env_inter import take_env_image, take_obstacle_image, take_goal_image, take_objects_image_training
from PIL import Image
import csv
import io
from common import get_args, load_vaes, load_field_parameters, make_temp_env
from envs.distance_graph import DistanceGraph2D
import time

def gen_random_point_pairs(N, field):
    # points pairs inside the field
    center_x, center_y, field_lx, field_ly = field
    # inside
    p1 = np.random.uniform(low=[center_x - field_lx, center_y - field_ly],
                           high=[center_x + field_lx, center_y + field_ly], size=[N, 2])
    p2 = np.random.uniform(low=[center_x - field_lx, center_y - field_ly],
                           high=[center_x + field_lx, center_y + field_ly], size=[N, 2])

    ppairs = np.stack([p1, p2], axis=1)
    return ppairs

def gen_same_dist_points_pairs(N, field):

    min_div_n = 100
    if min_div_n ** 2 < N:
        min_div_n = np.ceil(np.sqrt(N)).astype(int)

    center_x, center_y, field_lx, field_ly = field
    xmin = center_x-field_lx
    xmax = center_x+field_lx
    ymin = center_y-field_ly
    ymax = center_y+field_ly


    div = np.zeros(shape=(1),dtype=np.complex)#it must be complex where j in the vision
    div.imag = np.array([min_div_n])

    xx, yy = np.mgrid[xmin:xmax:div[0],ymin:ymax:div[0]]
    points = np.vstack([xx.ravel(), yy.ravel()])
    points = points.T
    inds1 = np.arange(len(points))
    np.random.shuffle(inds1)
    inds2 = np.arange(len(points))
    np.random.shuffle(inds2)
    ppairs = np.stack([points[inds1], points[inds2]], axis=1)
    ppairs = ppairs[:N]
    return ppairs

def gen_around_obstacle(N, field, list_obstacles, density):
    field_center_x, field_center_y, field_lx, field_ly = field
    field_left = field_center_x - field_lx
    field_right = field_center_x + field_lx
    field_down = field_center_y - field_ly
    field_up = field_center_y + field_ly

    points_list = []
    N_per_obstacle = np.ceil(N / len(list_obstacles)).astype(int)
    for obstacle in list_obstacles:
        center_x, center_y, lx, ly = obstacle
        left = center_x - lx
        right = center_x + lx
        down = center_y - ly
        up = center_y + ly
        l_left_limit = max(field_left, left-2*density)
        l_right_limit = min(right, left+1*density)


        r_right_limit = min(field_right, right + 2*density)
        r_left_limit = max(left, right - 1*density)

        u_up_limit = min(field_up, up + 2*density)
        u_down_limit = max(down, up - 1 * density)

        d_down_limit = max(field_down, down - 2*density)
        d_up_limit = min(up, down + 1*density)

        N_side = np.ceil(N_per_obstacle / 6).astype(int)
        p_left = np.random.uniform(low=[l_left_limit, d_down_limit],
                                   high=[l_right_limit, u_up_limit], size=[N_side, 2])
        p_right = np.random.uniform(low=[r_left_limit, d_down_limit],
                                   high=[r_right_limit, u_up_limit], size=[N_side, 2])
        p_up = np.random.uniform(low=[l_left_limit, u_down_limit],
                                   high=[r_right_limit, u_up_limit], size=[N_side, 2])
        p_down = np.random.uniform(low=[l_left_limit, d_down_limit],
                                   high=[r_right_limit, d_up_limit], size=[N_side, 2])
        points_list.append(np.stack([p_left, p_right], axis=1))
        points_list.append(np.stack([p_left, p_up], axis=1))
        points_list.append(np.stack([p_left, p_down], axis=1))
        points_list.append(np.stack([p_up, p_right], axis=1))
        points_list.append(np.stack([p_up, p_down], axis=1))
        points_list.append(np.stack([p_right, p_down], axis=1))
    points_list = np.concatenate(points_list)
    points_list = points_list[:N]
    return points_list

def connect_new_ppairs(N, ppairs, ppairs_around_obstacle):
    assert N <= len(ppairs)
    assert N <= len(ppairs_around_obstacle)
    inds1 = np.arange(N)
    np.random.shuffle(inds1)
    inds2 = np.arange(N)
    np.random.shuffle(inds2)
    new_ppairs = np.stack([ppairs[inds1][:, 0, :], ppairs_around_obstacle[inds2][:, 0, :]], axis=1)
    return new_ppairs


def flat_entries(bboxes_list, ppair):
    return np.concatenate([bboxes_list.ravel(), ppair.ravel()])

if __name__ == "__main__":
    args = get_args()
    # create data folder if it does not exist, corresponding folders, and files where to store data
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    base_data_dir = this_file_dir + 'data/'
    env_data_dir = base_data_dir + args.env + '/'
    make_dir(env_data_dir, clear=False)

    if args.vae_dist_help:
        load_vaes(args)
    load_field_parameters(args)
    env = make_temp_env(args)

    field_names = ['ppair', 'bbox', 'distance']
    csv_file_path = env_data_dir + 'distances.csv'
    csv_file_path_val = env_data_dir + 'distances_val.csv'
    csv_file_path_test = env_data_dir + 'distances_test.csv'
    for csv_path in [csv_file_path, csv_file_path_val, csv_file_path_test]:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        with open(csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

    #create about 20 000 samples for each step
    #have about 20 steps per each rollout
    #have about 5 rollouts
    #the last 2 are the more expensive since we need to create a graph in each of them
    n_rollouts = 5
    n_steps = 5#20

    field_center = args.field_center
    field_size = args.field_size
    field = field_center + field_size
    start_time = time.time()
    max_dist = -1.
    min_dist = 100.

    #variables for mean and std
    n_input = 0

    n_output = 0
    n_output_unreachable = 0

    for rollout_i in range(n_rollouts):
        print('starting rollout_i {}'.format(rollout_i))
        env.reset()
        obs = env.get_obs()
        for timestep in range(n_steps):
            print('At_timestep {}'.format(timestep))
            # get action from the ddpg policy
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            if args.vae_dist_help:

                obstacles_list = np.concatenate([obs['obstacle_latent'].copy(), obs['obstacle_size_latent'].copy()],
                                                axis=1)
            else:
                obstacles_list = obs['real_obstacle_info'].copy()
                if obstacles_list.ndim == 1:
                    obstacles_list = np.array([obstacles_list])
                assert obstacles_list.shape[1] == 6
                obstacles_list = np.concatenate([obstacles_list[:, :2], obstacles_list[:, 3:5]], axis=1)


            assert len(field) == 4

            buffer_bboxes = io.BytesIO()
            np.savetxt(buffer_bboxes, obstacles_list)

            #create graph
            graph = DistanceGraph2D(args=None, field=field, obstacles=obstacles_list,
                                    num_vertices=[100, 100], size_increase=0.0, use_discrete=False)
            graph.compute_cs_graph()
            graph.compute_dist_matrix()
            density = field_size[0]*2 / 100

            #create points pairs
            ppairs1 = gen_random_point_pairs(5000, field)
            ppairs2 = gen_same_dist_points_pairs(2000, field)
            ppairs3 = gen_around_obstacle(8000, field, list_obstacles=obstacles_list, density=density)
            ppairs4 = connect_new_ppairs(5000, ppairs1, ppairs3)
            ppairs = np.concatenate([ppairs1, ppairs2, ppairs3, ppairs4], axis=0)

            # store them in a file
            with open(csv_file_path, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=field_names)
                for i in range(len(ppairs)):
                    ppair = ppairs[i]
                    distance = graph.get_dist(ppair[0], ppair[1])
                    distance = distance[0] if distance[0] != np.inf else 9999

                    buffer_ppair = io.BytesIO()
                    np.savetxt(buffer_ppair, ppair)
                    entry = flat_entries(obstacles_list, ppair)
                    length_input = len(entry)
                    if n_input == 0:
                        it_input_mean = entry.copy()
                        it_input_std = np.zeros(shape=entry.shape)
                    else:
                        it_input_mean_prev = it_input_mean.copy()
                        it_input_mean = it_input_mean + (entry - it_input_mean)/(n_input + 1)
                        it_input_std = it_input_std + (entry - it_input_mean_prev)*(entry-it_input_mean)
                    n_input += 1

                    if distance != 9999 and distance > max_dist:
                        max_dist = distance.copy()
                    if distance != 9999 and distance < min_dist:
                        min_dist = distance.copy()

                    if distance == 9999:
                        n_output_unreachable += 1
                    else:
                        if n_output == 0:
                            it_output_mean = distance
                            it_output_std = 0
                        else:
                            it_output_mean_prev = copy.copy(it_output_mean)
                            it_output_mean = it_output_mean + (distance - it_output_mean) / (it_output_mean + 1)
                            it_output_std = it_output_std + (distance - it_output_mean_prev)*(distance-it_output_mean)
                        n_output += 1


                    values = {'ppair': buffer_ppair.getvalue().decode('utf-8'),
                              'bbox': buffer_bboxes.getvalue().decode('utf-8'),
                              'distance': distance}
                    writer.writerow(values)

            #same creation for validation and train set
            ppairs1_val = gen_random_point_pairs(1000, field)
            ppairs2_val = gen_same_dist_points_pairs(400, field)
            ppairs3_val = gen_around_obstacle(1600, field, list_obstacles=obstacles_list, density=density)
            ppairs4_val = connect_new_ppairs(1000, ppairs1, ppairs3)
            ppairs_val = np.concatenate([ppairs1, ppairs2, ppairs3, ppairs4], axis=0)
            with open(csv_file_path_val, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=field_names)
                for i in range(len(ppairs_val)):
                    ppair = ppairs_val[i]
                    distance = graph.get_dist(ppair[0], ppair[1])
                    distance = distance[0] if distance[0] != np.inf else 9999

                    buffer_ppair = io.BytesIO()
                    np.savetxt(buffer_ppair, ppair)

                    values = {'ppair': buffer_ppair.getvalue().decode('utf-8'),
                              'bbox': buffer_bboxes.getvalue().decode('utf-8'),
                              'distance': distance}
                    writer.writerow(values)



            ppairs1_test = gen_random_point_pairs(1000, field)
            ppairs2_test = gen_same_dist_points_pairs(400, field)
            ppairs3_test = gen_around_obstacle(1600, field, list_obstacles=obstacles_list, density=density)
            ppairs4_test = connect_new_ppairs(1000, ppairs1, ppairs3)
            ppairs_test = np.concatenate([ppairs1, ppairs2, ppairs3, ppairs4], axis=0)
            with open(csv_file_path_test, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=field_names)
                for i in range(len(ppairs_test)):
                    ppair = ppairs_test[i]
                    distance = graph.get_dist(ppair[0], ppair[1])
                    distance = distance[0] if distance[0] != np.inf else 9999

                    buffer_ppair = io.BytesIO()
                    np.savetxt(buffer_ppair, ppair)

                    values = {'ppair': buffer_ppair.getvalue().decode('utf-8'),
                              'bbox': buffer_bboxes.getvalue().decode('utf-8'),
                              'distance': distance}
                    writer.writerow(values)


    end_time = time.time()
    for _ in range(n_output_unreachable):
        it_output_mean_prev = copy.copy(it_output_mean)
        it_output_mean = it_output_mean + (max_dist + 1. - it_output_mean) / (it_output_mean + 1)
        it_output_std = it_output_std + (max_dist + 1. - it_output_mean_prev) * (max_dist + 1. - it_output_mean)
        n_output += 1
    print("time: {}".format(end_time - start_time))
    #calculate std from variance
    it_output_std = np.sqrt(it_output_std / (n_output - 1))
    for a in it_input_std:
        print(a)
        assert a >= 0.
        assert (a / (n_input-1)) >= 0

    it_input_std = np.sqrt(it_input_std / (n_input - 1))

    fn = ['max', 'min', 'mean_input', 'std_input', 'mean_output', 'std_output', 'input_size']
    csv_file_pat_vals = env_data_dir + 'dist_info.csv'
    if os.path.exists(csv_file_pat_vals):
        os.remove(csv_file_pat_vals)
    else:
        print('No previous csv file found')
    with open(csv_file_pat_vals, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        writer.writeheader()

        buffer_input_mean= io.BytesIO()
        np.savetxt(buffer_input_mean, it_input_mean)

        buffer_input_std = io.BytesIO()
        np.savetxt(buffer_input_std, it_input_std)

        values = {'max': max_dist,
                  'min': min_dist,
                  'mean_input': buffer_input_mean.getvalue().decode('utf-8'),
                  'std_input': buffer_input_std.getvalue().decode('utf-8'),
                  'mean_output': it_output_mean,
                  'std_output': it_output_std,
                  'input_size': length_input}

        writer.writerow(values)

