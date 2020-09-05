import numpy as np
from common import get_args, experiment_setup
import time
from PIL import Image
from envs import make_env
import copy
from j_vae.common_data import center_obstacle, range_x, range_y, obstacle_size, \
    min_obstacle_size, max_obstacle_size, puck_size, z_table_height

encoding_of = 'obstacle'
if encoding_of == 'goal':
    file = '../data/FetchPushObstacle/goal_set.npy'
elif encoding_of == 'obstacle':
    file = '../data/FetchPushObstacle/obstacle_set.npy'
elif encoding_of == 'size':
    file = '../data/FetchPushObstacle/obstacle_sizes_set.npy'
elif encoding_of == 'size_and_position':
    file = '../data/FetchPushObstacle/obstacle_sizes_points_set.npy'

points_file = '../data/FetchPushObstacle/points_for_sample.npy'
points_goal_file = '../data/FetchPushObstacle/goals_for_sample.npy'
size_file = '../data/FetchPushObstacle/sizes_for_sample.npy'


count = 1280*15#25#some more since now more positions exist
img_size = 84

def generate_points(range_x, range_y, z, total, object_x_y_size):
    rx = copy.deepcopy(range_x)
    ry = copy.deepcopy(range_y)
    rx[0] += object_x_y_size[0]
    rx[1] -= object_x_y_size[0]
    ry[0] += object_x_y_size[1]
    ry[1] -= object_x_y_size[1]
    xs = np.linspace(start=rx[0], stop=rx[1], num=total, endpoint=True)
    ys = np.linspace(start=ry[0], stop=ry[1], num=total, endpoint=True)
    points = []
    for i in range(total):
        for j in range(total):
            points.append([xs[i], ys[j], z])
    return points

def random_pos_inside(range_x, range_y, z, object_x_y_size):
    rx = copy.deepcopy(range_x)
    ry = copy.deepcopy(range_y)
    rx[0] += object_x_y_size[0]
    rx[1] -= object_x_y_size[0]
    ry[0] += object_x_y_size[1]
    ry[1] -= object_x_y_size[1]
    x = np.random.uniform(rx[0], rx[1])
    y = np.random.uniform(ry[0], ry[1])
    return [x, y, z]


def random_size_at(min_size, max_size, range_x, range_y, pos):
    distances_to_edge = [np.abs(range_x[0] - pos[0]), np.abs(range_x[1] - pos[0]),
                         np.abs(range_y[0] - pos[1]), np.abs(range_y[1] - pos[1])]
    min_d = min(distances_to_edge)
    maximum = min(max_size, min_d)
    s = np.random.uniform(min_size, maximum)
    return np.array([s, 0.035, 0.0])

def get_max_size(max_size, range_x, range_y, pos):
    distances_to_edge = [np.abs(range_x[0] - pos[0]), np.abs(range_x[1] - pos[0]),
                         np.abs(range_y[0] - pos[1]), np.abs(range_y[1] - pos[1])]
    min_d = min(distances_to_edge)
    maximum = min(max_size, min_d)
    return maximum


def loop_goals(env):
    train_data = np.empty([count, img_size, img_size, 3])
    print('Generating ', count, ' images for VAE-training...')

    env.env.env._set_arm_visible(visible=False)
    #move the obstacle away fro table
    env.env.env._set_position(names_list=['obstacle'], position=[2.0, 2.0, 2.0])


    env.env.env._set_size(names_list=['object0'], size=[puck_size, 0.035, 0.0])
    # The puck can be at the edgeneeds other dimensions
    points = np.load(points_goal_file)

    for i in range(count):
        if i < len(points):
            env.env.env._move_object(position=points[i])
        else:
            p = random_pos_inside(range_x, range_y, z=0.435, object_x_y_size=[0.015, 0.015])
            env.env.env._move_object(position=p)
        rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
        train_data[i] = rgb_array
        #if i % 640 == 0:
        #    img = Image.fromarray(rgb_array)
        #    img.show()
        #    img.close()

    np.save(file, train_data)


def create_and_save_sizes(quantity):
    m = 300
    sizes = np.linspace(min_obstacle_size, max_obstacle_size, num=m)
    data = []
    for i in range(quantity):
        if i < m:
            data.append(np.array([sizes[i], 0.035, 0.0]))
        else:
            r = np.random.uniform(0,4)
            if r < 3:
                data.append(random_size_at(min_obstacle_size, max_obstacle_size, range_x=range_x, range_y=range_y,
                                           pos=center_obstacle))
            else:
                data.append(np.array([sizes[i%m], 0.035, 0.0]))
    np.save(size_file, data)


def create_and_save_points(quantity, file, object_size):
    points = generate_points(range_x=range_x, range_y=range_y,  z=0.435, total=quantity,
                             object_x_y_size=[object_size, object_size])
    np.save(file, points)


def loop_obstacles(env, generate_sizes=False):
    train_data = np.empty([count, img_size, img_size, 3])
    print('Generating ', count, ' images for VAE-training...')
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=0.0)
    env.env.env._set_size(names_list=['obstacle'],size=[obstacle_size, 0.035, 0.0])
    #move other away
    env.env.env._move_object(position=[2.,2.,2.])

    if generate_sizes:
        sizes = np.load(size_file)
    else:
        points = np.load(points_file)
    for i in range(count):
        if generate_sizes:
            env.env.env._set_position(names_list=['obstacle'], position=center_obstacle)
            env.env.env._set_size(names_list=['obstacle'], size=sizes[i])
        else:
            if i < len(points):
                env.env.env._set_position(names_list=['obstacle'], position=points[i])
            else:
                p = random_pos_inside(range_x=[1.05, 1.55], range_y=[0.5, 1.0], z=0.435,
                                      object_x_y_size=[obstacle_size, obstacle_size])
                env.env.env._set_position(names_list=['obstacle'], position=p)

        rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
        train_data[i] = rgb_array
        #if i % 5120 == 0:
        #    img = Image.fromarray(rgb_array)
        #    img.show()
        #    img.close()
    np.save(file, train_data)


def loop_obstacles_precreated_points(env,):
    train_data = np.empty([count, img_size, img_size, 3])
    print('Generating ', count, ' images for VAE-training...')

    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=0.0)
    points = np.load(points_file)
    t = 0
    for i in range(len(points)):
        p = points[i]
        max_size = get_max_size(0.12, [1.05, 0.5], [1.55, 1.], p)
        actual_range_dist = max_size - 0.04
        ideal_range_dist = 0.12 - 0.04
        percentage = actual_range_dist / ideal_range_dist
        max_t = 30
        T = int(max_t * percentage)
        if T <= 0:
            T = 1
        for j in range(T):
            s = random_size_at(0.04, 0.12, [1.05, 0.5], [1.55, 1.], p)
            env.env.env._set_size(names_list=['obstacle'], size=s)
            env.env.env._set_position(names_list=['obstacle'], position=p)

            rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
            train_data[t] = rgb_array
            t += 1

    for _ in range(t, count):
        p = random_pos_inside([1.05, 1.55], [0.5, 1.0], 0.435, [min_obstacle_size, min_obstacle_size])
        env.env.env._set_position(names_list=['obstacle'],
                                  position=p)
        s = random_size_at(min_obstacle_size, 0.12, [1.05, 0.5], [1.55, 1.], p)
        env.env.env._set_size(names_list=['obstacle'], size=s)
        rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
        train_data[t] = rgb_array
        t += 1

    np.save(file, train_data)


if __name__ == '__main__':
    #create samples before hand
    #create_and_save_points(40, points_file, obstacle_size)
    # The puck can be at the edgeneeds other dimensions
    #create_and_save_points(40, points_goal_file, 0.015)
    #create_and_save_sizes(count)
    # Getting arguments from command line + defaults
    # Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
    args = get_args()
    env = make_env(args)

    # Generate VAE-dataset
    # Train VAE
    if encoding_of == 'goal':
        loop_goals(env)
    elif encoding_of == 'obstacle':
        loop_obstacles(env)
    elif encoding_of == 'size':
        loop_obstacles(env, generate_sizes=True)
    elif encoding_of == 'size_and_position':
        loop_obstacles_precreated_points(env)

    print('Finished generating dataset!')

