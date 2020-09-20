import argparse
import numpy as np
import os
import copy
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import make_dir
from vae_env_inter import take_env_image, take_obstacle_image, take_goal_image
from PIL import Image
from j_vae.common_data import train_file_name, puck_size, obstacle_size, z_table_height_obstacle, min_obstacle_size,\
    max_obstacle_size



#make range bigger than table so there are samples falling and in other positions
extend_region_parameters_goal = {'FetchPushObstacleFetchEnv-v1':
                                     {'center':np.array([1.3, 0.75, 0.425]), 'range':0.25}
                            }



size_visible_range = {'FetchPushObstacleFetchEnv-v1':
                                     {'x_r':np.array([1.2, 1.4]), 'y_r':np.array([0.65, 0.85])}}



def extend_sample_region_goal(env, args):
    env.object_center = extend_region_parameters_goal[args.env]['center']
    if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
        env.obj_range = extend_region_parameters_goal[args.env]['range'] + 0.05
    else:
        #todo?? is this necessary
        env.obj_range = extend_region_parameters_goal[args.env]['range']


def move_other_objects_away(env, args):
    if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
        env.env.env._set_position(names_list=['obstacle'], position=[2.0, 2.0, 2.0])
    else:
        env.env.env._move_object(position=[2., 2., 2.])

def random_size_at(env, min_size, max_size, pos, args):
    prc = 1.0
    #if not visible want to have max size from 0.25 to 1 the regalar max_size
    prc_dist = np.abs(1.0 - 0.3)

    #calculating distance if just at the activation region then amz size keeps the same otherwise 0.3 of it
    if pos[0] < size_visible_range[args.env]['x_r'][0]:
        limit = (env.object_center[0] - env.obj_range)
        base_d = np.abs(limit - size_visible_range[args.env]['x_r'][0])
        d = np.abs(limit - pos[0])
        local_prc = d / base_d
        new_prc = 0.3 + local_prc*prc_dist
        if new_prc < prc:
            prc = new_prc
    if pos[0] > size_visible_range[args.env]['x_r'][1]:
        limit = (env.object_center[0] + env.obj_range)
        base_d = np.abs(limit - size_visible_range[args.env]['x_r'][1])
        d = np.abs(limit - pos[0])
        local_prc = d / base_d
        new_prc = 0.3 + local_prc * prc_dist
        if new_prc < prc:
            prc = new_prc
    if pos[1] < size_visible_range[args.env]['y_r'][0]:
        limit = (env.object_center[1] - env.obj_range)
        base_d = np.abs(limit - size_visible_range[args.env]['y_r'][0])
        d = np.abs(limit - pos[1])
        local_prc = d / base_d
        new_prc = 0.3 + local_prc * prc_dist
        if new_prc < prc:
            prc = new_prc
    if pos[1] > size_visible_range[args.env]['y_r'][1]:
        limit = (env.object_center[1] + env.obj_range)
        base_d = np.abs(limit - size_visible_range[args.env]['y_r'][1])
        d = np.abs(limit - pos[1])
        local_prc = d / base_d
        new_prc = 0.3 + local_prc * prc_dist
        if new_prc < prc:
            prc = new_prc

    maximum = max_size*prc
    s = np.random.uniform(min_size, maximum)
    return np.array([s, 0.035, 0.0])

def obstacle_random_pos_and_size(env, args):
    if args.enc_type == 'obstacle' or args.enc_type == 'obstacle':
        pos = env.object_center + np.random.uniform(-env.obj_range, env.obj_range, size=3)
        pos[2] = z_table_height_obstacle
        env.env.env._set_position(names_list=['obstacle'], position=pos)
        size = random_size_at(env, min_obstacle_size, max_obstacle_size, pos, args)
        env.env.env._set_size(names_list=['obstacle'], size=size)


gen_setup_env_ops = {'FetchPushObstacleFetchEnv-v1':[extend_sample_region_goal]
                     }

after_env_reset_ops = {'FetchPushObstacleFetchEnv-v1':[move_other_objects_away, obstacle_random_pos_and_size]
                       }

during_loop_ops =  {'FetchPushObstacleFetchEnv-v1':[obstacle_random_pos_and_size]
                       }

if __name__ == "__main__":
    # call plot.py to plot success stored in progress.csv files

    # get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', help='the task for the generation of data', type=str,
                        default='generate', choices=['generate', 'mix'], required=True)
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
        data_1 = data_1[:int(len(data_1)/2)]
        data_2 = np.load(data_file_2)
        data_2 = data_2[:int(len(data_2)/2)]
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
                            choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes'])
        parser.add_argument('--count', help='number of samples', type=np.int32, default=1280 * 40)
        parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)
        args = parser.parse_args()


        #create data folder if it does not exist, corresponding folders, and files where to store data
        this_file_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        base_data_dir = this_file_dir + '../data/'
        env_data_dir = base_data_dir + args.env + '/'
        make_dir(env_data_dir, clear=False)
        data_file = env_data_dir + train_file_name[args.enc_type]

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
            else:
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


            #if i % 10 == 0:
            #    im = Image.fromarray(rgb_array)
            #    im.show()
            #    im.close()
        #store files
        np.save(data_file, train_data)


        '''train_data = np.load(data_file)
        all_idx = np.arange(len(train_data)).tolist()
        def show_some_sampled_images():
            n = 10
            a = None
            for i in range(n):
                b = None
                for j in range(n):
                    id = np.random.choice(all_idx, size=1)[0]
                    all_idx.remove(id)
                    j_im = train_data[id].copy()
                    if b is None:
                        b = j_im.copy()
                    else:
                        b = np.concatenate([b, j_im], axis=1)
                if a is None:
                    a = b.copy()
                else:
                    a = np.concatenate([a, b], axis=0)
            img = Image.fromarray(a.astype(np.uint8))
            img.show()
            img.close()
        for _ in range(5):
            show_some_sampled_images()'''
