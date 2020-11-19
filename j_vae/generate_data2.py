import argparse
import numpy as np
import os
import copy
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import make_dir
from vae_env_inter import take_env_image, take_obstacle_image, take_goal_image, take_objects_image_training
from PIL import Image
from j_vae.common_data import train_file_name, puck_size, obstacle_size, z_table_height_obstacle, min_obstacle_size,\
    max_obstacle_size,z_table_height_goal



#make range bigger than table so there are samples falling and in other positions
extend_region_parameters_goal = {'FetchPushObstacleFetchEnv-v1':
                                     {'center':np.array([1.3, 0.75, 0.425]), 'range':0.25},
                                 'FetchPushMovingObstacleEnv-v1':
                                     {'center':np.array([1.3, 0.75, 0.425]), 'range':0.25},
                                 'FetchGenerativeEnv-v1':
                                     {'center':np.array([1.3, 0.75, 0.425]), 'range':0.25},
                            }


size_visible_range = {'FetchPushObstacleFetchEnv-v1':
                                     {'x_r':np.array([1.25, 1.35]), 'y_r':np.array([0.7, 0.8]),
                                      'red_prc':min_obstacle_size[
                                                    'FetchPushObstacleFetchEnv-v1']/max_obstacle_size[
                                          'FetchPushObstacleFetchEnv-v1']
                                      },

                      'FetchPushMovingObstacleEnv-v1':
                          {'end_x_r': np.array([1.05, 1.55]), 'end_y_r': np.array([0.5, 1.]),
                           'start_x_r': np.array([1.26, 1.34]), 'start_y_r': np.array([0.71, 0.79]),
                           'start_max_size': 0.25,
                           'end_max_size': 0.04}
                      }



def extend_sample_region(env, args):
    env.object_center = extend_region_parameters_goal[args.env]['center']
    if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
        env.obj_range = extend_region_parameters_goal[args.env]['range'] + 0.05
    else:
        #todo?? is this necessary
        env.obj_range = extend_region_parameters_goal[args.env]['range']

#IMPORTANT: this is a temporal solution. THe robot has difficultirs to reach the 'bottom' part of the table
# the actual solution would be to change the model of the envrionment so robot can reach evry point equallt easy
def goal_random_pos_recom(env, args):
    if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
        a = np.random.randint(low=0, high=43, size=1)
        if a == 0:
            x = np.random.uniform(1.33, 1.41, size=1)
            y = np.random.uniform(0.7, 0.83,size=1)
            p = np.array([x, y, z_table_height_goal])
            env.env.env._move_object(position=p)


def move_other_objects_away(env, args):
    if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
        env.env.env._set_position(names_list=['obstacle'], position=[2.0, 2.0, 2.0])
    else:
        env.env.env._move_object(position=[2., 2., 2.])


def random_size_at(env, min_size, max_size, pos, args):
    if args.env == 'FetchPushObstacleFetchEnv-v1':
        prc = 1.0
        #if entering not visible want to have max size from 0.25 to 1 the regalar max_size
        basic_prc_size = size_visible_range[args.env]['red_prc']
        prc_dist = np.abs(1.0 - basic_prc_size)

        #calculating distance if just at the activation region then amz size keeps the same otherwise basic_prc_size of it
        if pos[0] < size_visible_range[args.env]['x_r'][0]:
            limit = (env.object_center[0] - env.obj_range)
            base_d = np.abs(limit - size_visible_range[args.env]['x_r'][0])
            d = np.abs(limit - pos[0])
            local_prc = d / base_d
            new_prc = basic_prc_size + local_prc*prc_dist
            if new_prc < prc:
                prc = new_prc
        if pos[0] > size_visible_range[args.env]['x_r'][1]:
            limit = (env.object_center[0] + env.obj_range)
            base_d = np.abs(limit - size_visible_range[args.env]['x_r'][1])
            d = np.abs(limit - pos[0])
            local_prc = d / base_d
            new_prc = basic_prc_size + local_prc * prc_dist
            if new_prc < prc:
                prc = new_prc
        if pos[1] < size_visible_range[args.env]['y_r'][0]:
            limit = (env.object_center[1] - env.obj_range)
            base_d = np.abs(limit - size_visible_range[args.env]['y_r'][0])
            d = np.abs(limit - pos[1])
            local_prc = d / base_d
            new_prc = basic_prc_size + local_prc * prc_dist
            if new_prc < prc:
                prc = new_prc
        if pos[1] > size_visible_range[args.env]['y_r'][1]:
            limit = (env.object_center[1] + env.obj_range)
            base_d = np.abs(limit - size_visible_range[args.env]['y_r'][1])
            d = np.abs(limit - pos[1])
            local_prc = d / base_d
            new_prc = basic_prc_size + local_prc * prc_dist
            if new_prc < prc:
                prc = new_prc

        maximum = max_size*prc
        s = np.random.uniform(min_size, maximum)
        return np.array([s, 0.035, 0.0])
    elif args.env == 'FetchPushMovingObstacleEnv-v1':
        max_size = 0.25
        min_size = 0.025
        x_max_size = None
        y_max_size = None


        if pos[0] < size_visible_range[args.env]['start_x_r'][0]:
            d = np.abs(size_visible_range[args.env]['end_x_r'][0] - size_visible_range[args.env]['start_x_r'][0])
            prc = np.abs(pos[0] - size_visible_range[args.env]['end_x_r'][0])/d
            x_max_size = size_visible_range[args.env]['end_max_size'] +\
                         prc * np.abs(size_visible_range[args.env]['start_max_size']-size_visible_range[args.env]['end_max_size'])
        elif pos[0] > size_visible_range[args.env]['start_x_r'][1]:
            d = np.abs(size_visible_range[args.env]['end_x_r'][1] - size_visible_range[args.env]['start_x_r'][1])
            prc = np.abs(pos[0] - size_visible_range[args.env]['end_x_r'][1]) / d
            x_max_size = size_visible_range[args.env]['end_max_size'] + \
                         prc * np.abs(
                size_visible_range[args.env]['start_max_size'] - size_visible_range[args.env]['end_max_size'])
        else:
            x_max_size = max_size

        if pos[1] < size_visible_range[args.env]['start_y_r'][0]:
            d = np.abs(size_visible_range[args.env]['end_y_r'][0] - size_visible_range[args.env]['start_y_r'][0])
            prc = np.abs(pos[1] - size_visible_range[args.env]['end_y_r'][0]) / d
            y_max_size = size_visible_range[args.env]['end_max_size'] + \
                         prc * np.abs(
                size_visible_range[args.env]['start_max_size'] - size_visible_range[args.env]['end_max_size'])
        elif pos[1] > size_visible_range[args.env]['start_y_r'][1]:
            d = np.abs(size_visible_range[args.env]['end_y_r'][1] - size_visible_range[args.env]['start_y_r'][1])
            prc = np.abs(pos[1] - size_visible_range[args.env]['end_y_r'][1]) / d
            y_max_size = size_visible_range[args.env]['end_max_size'] + \
                         prc * np.abs(
                size_visible_range[args.env]['start_max_size'] - size_visible_range[args.env]['end_max_size'])
        else:
            y_max_size = max_size

        s_x = np.random.uniform(min_size, x_max_size)
        s_y = np.random.uniform(min_size, y_max_size)
        return np.array([s_x, s_y, 0.0])

    else:
        raise Exception('np configuration for {} env of random_size_at'.format(args.env))


def obstacle_random_pos_and_size(env, args):
    if args.enc_type == 'obstacle' or args.enc_type == 'obstacle':
        pos = env.object_center + np.random.uniform(-env.obj_range, env.obj_range, size=3)
        #pos = env.object_center + np.array([0.04, 0, 0.])
        pos[2] = z_table_height_obstacle
        env.env.env._set_position(names_list=['obstacle'], position=pos)
        size = random_size_at(env, min_obstacle_size, max_obstacle_size, pos, args)
        #size = [0.25, 0.25, 0.035]
        env.env.env._set_size(names_list=['obstacle'], size=size)

def gen_all_data(env, args):
    def get_current_max_size(pos, edge_max, max):
        min_distance_to_edge = None
        begin_distance_for_reduce = max - edge_max
        if np.abs(1.55 - pos[0]) <= begin_distance_for_reduce:
            min_distance_to_edge = np.abs(1.55 - pos[0])
        if np.abs(pos[0] - 1.05) <= begin_distance_for_reduce:
            if min_distance_to_edge is None:
                min_distance_to_edge = np.abs(pos[0] - 1.05)
            elif np.abs(pos[0] - 1.05) < min_distance_to_edge:
                min_distance_to_edge = np.abs(pos[0] - 1.05)
        if np.abs(1.0 - pos[1]) <= begin_distance_for_reduce:
            if min_distance_to_edge is None:
                min_distance_to_edge = (1.0 - pos[1])
            elif np.abs(pos[0] - 1.05) < min_distance_to_edge:
                min_distance_to_edge = (1.0 - pos[1])
        if np.abs(pos[1] - 0.5) <= begin_distance_for_reduce:
            if min_distance_to_edge is None:
                min_distance_to_edge = np.abs(pos[1] - 0.5)
            elif np.abs(pos[0] - 1.05) < min_distance_to_edge:
                min_distance_to_edge = np.abs(pos[1] - 0.5)
        if min_distance_to_edge is None:
            return max
        else:
            prc = 1.0 - min_distance_to_edge / begin_distance_for_reduce
            return max - prc * begin_distance_for_reduce
    # choose and move other away
    p1 = np.array([-20., 20., 20.])
    p2 = np.array([-20., -20., 20.])
    
    obj_ind = np.random.randint(0,3)
    if obj_ind == 0:
        obj='cylinder'
        env.env.env._set_position(names_list=['rectangle'], position=p1)
        env.env.env._set_position(names_list=['cube'], position=p2)
    elif obj_ind == 1:
        obj = 'rectangle'
        env.env.env._set_position(names_list=['cylinder'], position=p1)
        env.env.env._set_position(names_list=['cube'], position=p2)
    elif obj_ind == 2:
        obj = 'cube'
        env.env.env._set_position(names_list=['rectangle'], position=p1)
        env.env.env._set_position(names_list=['cylinder'], position=p2)
    #pos
    pos = env.object_center + np.random.uniform(-env.obj_range, env.obj_range, size=3)
    #pos = env.object_center
    #pos[0] -= np.random.uniform(0.1, env.obj_range, size=1)
    #pos[1] += np.random.uniform(0.1, env.obj_range, size=1)
    #pos = np.array([1.5, 0.52, 0.435])

    # rotation
    if obj == 'cylinder':
        rot_x = np.random.uniform(0., 180.)
        rot_y = np.random.uniform(0., 180.)
        env.env.env._rotate([obj], rot_x, rot_y, 0.)
    elif obj == 'rectangle':
        rot_z = np.random.uniform(0., 180.)
        env.env.env._rotate([obj], 0., 0., rot_z)
    elif obj == 'cube':
        rot_x = np.random.uniform(0., 90.)
        rot_y = np.random.uniform(0., 90.)
        rot_z = np.random.uniform(0., 90.)
        env.env.env._rotate([obj], rot_x, rot_y, rot_z)

    # size and eventually change in height
    if obj == 'cylinder':
        max_r = 0.08
        max_at_edge = 0.05
        current_max = get_current_max_size(pos, max_at_edge, max_r)
        r = np.random.uniform(0.025, current_max)
        height = np.random.uniform(0.02, 0.035)
        size = [r, height, 0.0]        
    elif obj == 'rectangle':
        max_l_1 = 0.25
        max_l_2 = 0.08
        max_at_edge = 0.035
        begin_reduce = 0.2
        begin_reduce_2 = 0.14
        max_at_edge_2 = 0.04
        dist_to_edge = None
        base_dist_used = None
        max_at = None
        if 0. <= rot_z <= 45. or 135. <= rot_z <= 180.:
            if np.abs(1.55 - pos[0]) <= begin_reduce:
                d = np.abs(1.55 - pos[0])
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce
                    max_at = max_at_edge
            if np.abs(pos[0] - 1.05) <= begin_reduce:
                d = np.abs(pos[0] - 1.05)
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce
                    max_at = max_at_edge
            #
            if np.abs(pos[1] - 0.5) <= begin_reduce_2:
                d = np.abs(pos[1] - 0.5)
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce_2
                    max_at = max_at_edge_2
            if np.abs(1.0 - pos[1]) <= begin_reduce_2:
                d = np.abs(1.0 - pos[1])
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce_2
                    max_at = max_at_edge_2

        if 45. <= rot_z <= 90. or 90. <= rot_z <= 135.:
            if np.abs(pos[1] - 0.5) <= begin_reduce:
                d = np.abs(pos[1] - 0.5)
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce
                    max_at = max_at_edge
            if np.abs(1.0 - pos[1]) <= begin_reduce:
                d = np.abs(1.0 - pos[1])
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce
                    max_at = max_at_edge
            #
            if np.abs(1.55 - pos[0]) <= begin_reduce_2:
                d = np.abs(1.55 - pos[0])
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce_2
                    max_at = max_at_edge_2
            if np.abs(pos[0] - 1.05) <= begin_reduce_2:
                d = np.abs(pos[0] - 1.05)
                if dist_to_edge is None or d < dist_to_edge:
                    dist_to_edge = d.copy()
                    base_dist_used = begin_reduce_2
                    max_at = max_at_edge_2

        if dist_to_edge is None:
            current_max = max_l_1
        else:
            diff_size = max_l_1 - max_at
            prc = dist_to_edge / base_dist_used
            current_max = max_at + prc * diff_size

        assert current_max > 0.02
        long_length = np.random.uniform(0.02, current_max)
        if long_length > max_l_2:
            max_s = 0.04
        else:
            max_s = 0.5 * long_length
        if max_s < 0.02:
            short_length = max_s
        else:
            short_length = np.random.uniform(0.02, max_s)
        height = np.random.uniform(0.02, 0.035)
        size = [long_length, short_length, height]
    elif obj == 'cube':
        max_l = 0.06
        max_at_edge = 0.035
        current_max = get_current_max_size(pos, max_at_edge, max_l)
        s = np.random.uniform(0.02, current_max)
        size = [s, s, s]
        height = s
    #pos after settinh height
    env.env.env._set_size(names_list=[obj], size=size)
    pos[2] = 0.4 + height
    env.env.env._set_position(names_list=[obj], position=pos)
    

    
    #color
    r_color = np.random.randint(0, 5)
    if r_color == 0:
        env.env.env._change_color([obj], 1., 0., 0.)
    elif r_color == 1:
        env.env.env._change_color([obj], 0., 0., 1.)
    elif r_color == 2:
        env.env.env._change_color([obj], 0.5, 0., 0.5)
    elif r_color == 3:
        env.env.env._change_color([obj], 0., 0., 0.)
    else:
        env.env.env._change_color([obj], 0.5, 0.5, 0.)

def _get_reduce_factor(distance_start_reduce, max_size, max_size_at_edge, current_size):
    if current_size <= max_size_at_edge:
        return 0.
    else:
        reduce_prc = (current_size - max_size_at_edge) / (max_size - max_size_at_edge)
        return distance_start_reduce * reduce_prc

def _get_pos(ar_x, ar_y, center, range_left, range_right, range_up, range_down, size):
    def _gen_pos():
        p = center
        p[0] += np.random.uniform(range_left, range_right, size=1)
        p[1] += np.random.uniform(range_up, range_down, size=1)
        p[2] = 0.4 + size[2]
        return p

    def is_inside_ocuped_areas(p, s):
        for i in range(len(ar_x)):
            if (ar_x[i][0] <= p[0] + s[0] <= ar_x[i][1] and ar_y[i][0] <= p[1] + s[1] <= ar_y[i][1]):
                return True
            elif (ar_x[i][0] <= p[0] - s[0] <= ar_x[i][1] and ar_y[i][0] <= p[1] - s[1] <= ar_y[i][1]):
                return True
            elif (ar_x[i][0] <= p[0] <= ar_x[i][1] and ar_y[i][0] <= p[1] <= ar_y[i][1]):
                return True
        return False

    t = 0
    while True:
        t += 1
        pos = _gen_pos()
        if not is_inside_ocuped_areas(pos, [size[0]+0.01, size[1]+0.01]):
            return pos
        if t >= 100:
            print('took to much time')
            return None

def _gen_cylinder(ocuped_areas_x, ocuped_areas_y):
    #size
    max_r = 0.08
    max_at_edge = 0.05
    min_size = 0.035
    start_reduce = 0.03

    r = np.random.uniform(min_size, max_r)
    height = np.random.uniform(0.02, 0.035)
    size = [r, height, 0.0]


    # rotation
    rot_x = np.random.uniform(0., 180.)
    rot_y = np.random.uniform(0., 180.)


    #position
    reduce_factor = _get_reduce_factor(start_reduce, max_r, max_at_edge, r)
    assert reduce_factor >= 0.
    assert env.obj_range - reduce_factor > 0
    pos = _get_pos(ocuped_areas_x, ocuped_areas_y, env.object_center.copy(), -env.obj_range + reduce_factor,
                   env.obj_range - reduce_factor, -env.obj_range + reduce_factor, env.obj_range - reduce_factor,
                   [r, r, height])#for the cylinder emulates a square bounding box
    if pos is None:
        return None, None
    else:
        return size, rot_x, rot_y, pos, [pos[0] - size[0], pos[0] + size[0]], [pos[1] - size[1], pos[1] + size[1]]

def _gen_cube(ocuped_areas_x, ocuped_areas_y):
    #size
    max_l = 0.05
    max_at_edge = 0.035
    min_size = 0.03
    start_reduce = 0.015
    s = np.random.uniform(min_size, max_l)
    size = [s, s, s]


    # rotation
    rot_x = np.random.uniform(0., 90.)
    rot_y = np.random.uniform(0., 90.)
    rot_z = np.random.uniform(0., 90.)

    #position
    reduce_factor = _get_reduce_factor(start_reduce, max_l, max_at_edge, s)
    assert reduce_factor >= 0.
    assert env.obj_range - reduce_factor > 0
    pos = _get_pos(ocuped_areas_x, ocuped_areas_y, env.object_center.copy(), -env.obj_range + reduce_factor,
                   env.obj_range - reduce_factor, -env.obj_range + reduce_factor, env.obj_range - reduce_factor,
                   size)
    if pos is None:
        return None, None
    else:

        return size, rot_x, rot_y, rot_z, pos, [pos[0] - size[0], pos[0] + size[0]], \
               [pos[1] - size[1], pos[1] + size[1]]

def _gen_rectangle():
    # size obstacle
    max_l = 0.25
    min_l = 0.08
    long_length = np.random.uniform(min_l, max_l)
    short_length = np.random.uniform(0.02, 0.04)
    height_obstacle = np.random.uniform(0.02, 0.035)
    size = [long_length, short_length, height_obstacle]

    # rotation obstacle
    r_case = np.random.randint(0, 2)
    if r_case == 0:
        rot_z = 0.
    else:
        rot_z = 90.

    # pos_obstacle
    begin_reduce = 0.2  # distance to edge when when lenght 0.25
    reduce_prc = long_length / max_l
    reduce_factor = begin_reduce * reduce_prc
    pos_obstacle = env.object_center.copy()
    if rot_z == 0.:
        pos_obstacle[0] += np.random.uniform(-env.obj_range + reduce_factor, env.obj_range - reduce_factor, size=1)
        pos_obstacle[1] += np.random.uniform(-env.obj_range, env.obj_range, size=1)
        occuped_area_x = [pos_obstacle[0] - long_length, pos_obstacle[0] + long_length]
        occuped_area_y = [pos_obstacle[1] - short_length, pos_obstacle[1] + short_length]
        bbox = [pos_obstacle[0], pos_obstacle[1], long_length, short_length]
    else:
        pos_obstacle[0] += np.random.uniform(-env.obj_range, env.obj_range, size=1)
        pos_obstacle[1] += np.random.uniform(-env.obj_range + reduce_factor, env.obj_range - reduce_factor, size=1)
        occuped_area_x = [pos_obstacle[0] - short_length, pos_obstacle[0] + short_length]
        occuped_area_y = [pos_obstacle[1] - long_length, pos_obstacle[1] + long_length]
        bbox = [pos_obstacle[0], pos_obstacle[1], short_length, long_length]
    pos_obstacle[2] = 0.4 + height_obstacle
    return size, rot_z, pos_obstacle, occuped_area_x, occuped_area_y, bbox

def gen_all_data_mixed(env, args):
    for n in ['rectangle', 'rectangle1', 'rectangle2', 'cylinder', 'cylinder1', 'cube', 'cube1']:
        env.env.env._set_position(names_list=[n], position=[10., 10., 0.])

    colors = [(1., 0., 0.), (0., 0., 1.), (0.5, 0., 0.5), (0.8, 0.8, 0.)]
    color_count = {}
    for i in range(len(colors)):
        color_count[i] = 0

    def select_and_change_color(obj_name):
        color_ind = np.random.randint(0, len(colors))
        while color_count[color_ind] >= 1:
            color_ind = (color_ind +1) % len(colors)
        r, g, b = colors[color_ind]
        env.env.env._change_color([obj_name], r, g, b)
        color_count[color_ind] +=1


    occuped_areas_x = []
    occuped_areas_y = []
    max_els = 4

    number_rectangles = np.random.choice(a=[0, 1, 2, 3], p = [0.3, 0.2, 0.3, 0.2])
    if number_rectangles == 4 or number_rectangles == 5:#this is done just to aument the possibility to have less els
        number_rectangles = 0
    rem_els = max_els - number_rectangles
    for i in range(number_rectangles):
        if i == 0:
            rect_name = 'rectangle'
        elif i == 1:
            rect_name = 'rectangle1'
        else:
            rect_name = 'rectangle2'

        size, rot_z, pos_obstacle, occuped_area_x_rectangle, occuped_area_y_rectangle = _gen_rectangle()
        env.env.env._set_size(names_list=[rect_name], size=size)
        env.env.env._rotate([rect_name], 0., 0., rot_z)
        env.env.env._set_position(names_list=[rect_name], position=pos_obstacle)
        select_and_change_color(rect_name)
        occuped_areas_x.append(occuped_area_x_rectangle)
        occuped_areas_y.append(occuped_area_y_rectangle)

    #generate other objects

    if rem_els > 0:
        max_n_cylynders = min(2, rem_els)
        n_cylinders = np.random.randint(0, max_n_cylynders+1)
        rem_els = rem_els - n_cylinders
        for i in range(n_cylinders):
            if i == 0:
                cyl_name = 'cylinder'
            elif i == 1:
                cyl_name = 'cylinder1'

            size_cyl, rot_x_cyl, rot_y_cyl, pos_cyl, oc_x_cyl, oc_y_cyl = _gen_cylinder(occuped_areas_x, occuped_areas_y)
            env.env.env._set_size(names_list=[cyl_name], size=size_cyl)
            env.env.env._rotate([cyl_name], rot_x_cyl, rot_y_cyl, 0.)
            env.env.env._set_position(names_list=[cyl_name], position=pos_cyl)
            select_and_change_color(cyl_name)
            occuped_areas_x.append(oc_x_cyl)
            occuped_areas_y.append(oc_y_cyl)

    if rem_els > 0:
        max_n_cubes = min(1, rem_els)
        n_cubes = np.random.randint(0, max_n_cubes + 1)

        for i in range(n_cubes):
            if i == 0:
                cube_name = 'cube'
            elif i == 1:
                cube_name = 'cube1'

            size_cube, rot_x_cube, rot_y_cube, rot_z_cube, pos_cube, oc_x_cube, oc_y_cube = _gen_cube(occuped_areas_x, occuped_areas_y)
            env.env.env._set_size(names_list=[cube_name], size=size_cube)
            env.env.env._rotate([cube_name], rot_x_cube, rot_y_cube, rot_z_cube)
            env.env.env._set_position(names_list=[cube_name], position=pos_cube)
            select_and_change_color(cube_name)

            occuped_areas_x.append(oc_x_cube)
            occuped_areas_y.append(oc_y_cube)

    '''case = np.random.randint(0, 4)
    if case == 0:
        #just cylinder
        #env.env.env._set_position(names_list=['cube'], position=p2)
        oc_x, oc_y = _gen_cylinder(occuped_areas_x, occuped_areas_y)
        if oc_x is None:
            print('no pos created; genrating just obstacle')
            #env.env.env._set_position(names_list=['cylinder'], position=p1)
    elif case == 1:
        #just cube
        #env.env.env._set_position(names_list=['cylinder'], position=p1)
        oc_x, oc_y = _gen_cube(occuped_areas_x, occuped_areas_y)
        if oc_x is None:
            print('no pos created; genrating just obstacle')
            #env.env.env._set_position(names_list=['cube'], position=p2)
    elif case == 2:
        #both
        oc_x, oc_y = _gen_cylinder(occuped_areas_x, occuped_areas_y)
        if oc_x is None:
            print('no pos created for cyl in both; generating just cube, perhaps')
            #env.env.env._set_position(names_list=['cylinder'], position=p1)
        else:
            occuped_areas_x.append(oc_x)
            occuped_areas_y.append(oc_y)

        oc_x_c, oc_y_c = _gen_cube(occuped_areas_x, occuped_areas_y)
        if oc_x_c is None:
            print('no pos created for cube in both; generating just obstacle')
            #env.env.env._set_position(names_list=['cube'], position=p2)

    else:
        #no other objects
        #env.env.env._set_position(names_list=['cylinder'], position=p1)
        #env.env.env._set_position(names_list=['cube'], position=p2)
        pass'''



gen_setup_env_ops = {'FetchPushObstacleFetchEnv-v1':[extend_sample_region],
                     'FetchPushMovingObstacleEnv-v1':[extend_sample_region],
                     'FetchGenerativeEnv-v1':[extend_sample_region]
                     }

after_env_reset_ops = {'FetchPushObstacleFetchEnv-v1':[move_other_objects_away, obstacle_random_pos_and_size,
                                                       goal_random_pos_recom],
                       'FetchPushMovingObstacleEnv-v1':[move_other_objects_away, obstacle_random_pos_and_size,
                                                       goal_random_pos_recom],
                       'FetchGenerativeEnv-v1':[]
                       }


during_loop_ops =  {'FetchPushObstacleFetchEnv-v1':[obstacle_random_pos_and_size],
                    'FetchPushMovingObstacleEnv-v1':[obstacle_random_pos_and_size],
                    'FetchGenerativeEnv-v1':[gen_all_data_mixed]
                    }

if __name__ == "__main__":
    # call plot.py to plot success stored in progress.csv files

    # get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', help='the task for the generation of data', type=str,
                        default='generate', choices=['generate', 'mix', 'show'], required=True)
    parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1', choices=Robotics_envs_id)
    args, _ = parser.parse_known_args()
    if args.task == 'mix':
        parser.add_argument('--file_1', help='first file to mix', type=str)
        parser.add_argument('--file_2', help='second file to mix', type=str)
        parser.add_argument('--output_file', help='name of output file for mixed dataset', type=str)
        args = parser.parse_args()

        this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        base_data_dir = this_file_dir + '../data/'
        env_data_dir = base_data_dir + args.env + '/'
        make_dir(env_data_dir, clear=False)
        data_file_1 = env_data_dir + args.file_1
        data_file_2 = env_data_dir + args.file_2

        data_1 = np.load(data_file_1)
        data_2 = np.load(data_file_2)
        mixed_data = np.concatenate([data_1, data_2], axis=0)
        np.random.shuffle(mixed_data)

        output_file = env_data_dir + args.output_file
        np.save(output_file, mixed_data)

    else:
        if args.env == 'HandReach-v0':
            parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                                choices=['vanilla', 'reach'])
        args, _ = parser.parse_known_args()
        if args.env == 'HandReach-v0':
            parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                                choices=['vanilla', 'reach'])
        else:
            parser.add_argument('--goal', help='method of goal generation', type=str, default='interval',
                                choices=['vanilla', 'fixobj', 'interval', 'custom'])
            if args.env[:5] == 'Fetch':
                parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32,
                                    default=1.0)
            elif args.env[:4] == 'Hand':
                parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32,
                                    default=0.25)

        parser.add_argument('--enc_type', help='the type of attribute that we want to generate/encode', type=str,
                            choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes', 'all'])
        parser.add_argument('--count', help='number of samples', type=np.int32, default=1280*30)
        parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=64)#84)
        args = parser.parse_args()

        # create data folder if it does not exist, corresponding folders, and files where to store data
        this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        base_data_dir = this_file_dir + '../data/'
        env_data_dir = base_data_dir + args.env + '/'
        make_dir(env_data_dir, clear=False)
        data_file = env_data_dir + train_file_name[args.enc_type]

        if args.task == 'generate':
            #load environment
            env = make_env(args)
            #setup env(change generation region; move other objects(just leave those we need)??)
            for func in gen_setup_env_ops[args.env]:
                func(env, args)

            #loop through(moving object and making)
            train_data = np.empty([args.count, args.img_size, args.img_size, 3])
            i = 0
            while i < args.count:
                env.reset()
                for func in after_env_reset_ops[args.env]:
                    func(env, args)
                # goal_loop
                if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
                    rgb_array = take_goal_image(env, img_size=args.img_size)
                    train_data[i] = rgb_array.copy()
                    i += 1
                    for _ in range(3):
                        if i < args.count:
                            for a_t in range(25):
                                action = env.action_space.sample()
                                env.step(action)
                            rgb_array = take_goal_image(env, img_size=args.img_size)
                            train_data[i] = rgb_array.copy()
                            i += 1
                        else:
                            break
                #obstacle loop
                elif args.enc_type == 'obstacle' or args.enc_type == 'obstacle_sizes':
                    rgb_array = take_obstacle_image(env, img_size=args.img_size)
                    train_data[i] = rgb_array.copy()
                    i += 1
                    for _ in range(5):
                        if i < args.count:
                            for func in during_loop_ops[args.env]:
                                func(env, args)
                            rgb_array = take_obstacle_image(env, img_size=args.img_size)
                            train_data[i] = rgb_array.copy()
                            i += 1
                        else:
                            break
                elif args.enc_type == 'all':
                    for func in during_loop_ops[args.env]:
                        func(env, args)
                    rgb_array = take_objects_image_training(env, img_size=args.img_size)
                    train_data[i] = rgb_array.copy()
                    i += 1


                '''if i % 10 == 0:
                    im = Image.fromarray(rgb_array)
                    im.show()
                    im.close()'''
            # store files
            np.save(data_file, train_data)
        else:
            train_data = np.load(data_file)
            all_idx = np.arange(len(train_data)).tolist()
            def show_some_sampled_images():
                n = 10
                a = None
                spacer_x = np.zeros(shape=(args.img_size, 5, 3))
                spacer_y = np.zeros(shape=(5, n*args.img_size + 5*(n-1) ,3))
                for i in range(n):
                    b = None
                    for j in range(n):
                        id = np.random.choice(all_idx, size=1, replace=False)[0]
                        all_idx.remove(id)
                        j_im = train_data[id].copy()
                        if b is None:
                            b = j_im.copy()
                        else:
                            b = np.concatenate([b, spacer_x, j_im], axis=1)
                    if a is None:
                        a = b.copy()
                    else:
                        a = np.concatenate([a, spacer_y, b], axis=0)
                img = Image.fromarray(a.astype(np.uint8))
                img.show()
                img.close()
            for _ in range(3):
                show_some_sampled_images()
