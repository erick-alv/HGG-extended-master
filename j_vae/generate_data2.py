import argparse
import numpy as np
import os
import copy
from envs import make_env, clip_return_range, Robotics_envs_id
from utils.os_utils import make_dir
from vae_env_inter import take_env_image, take_obstacle_image, take_goal_image
from PIL import Image
from j_vae.common_data import train_file_name




#make range biiger than table so there are samples falling and in other positions
extend_region_parameters_goal = {'FetchPushObstacleFetchEnv-v1':
                                     {'center':np.array([1.3, 0.75, 0.425]), 'range':0.32}
                            }

def extend_sample_region_goal(env, args):
    env.object_center = extend_region_parameters_goal[args.env]['center']
    env.obj_range = extend_region_parameters_goal[args.env]['range']

def move_other_objects_away(env, args):
    if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
        env.env.env._set_position(names_list=['obstacle'], position=[2.0, 2.0, 2.0])
    else:
        env.env.env._move_object(position=[2., 2., 2.])

gen_setup_env_ops = {'FetchPushObstacleFetchEnv-v1':[extend_sample_region_goal, move_other_objects_away]
                     }

after_env_reset_ops = {'FetchPushObstacleFetchEnv-v1':[]
                       }

if __name__ == "__main__":
    # call plot.py to plot success stored in progress.csv files

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1', choices=Robotics_envs_id)
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
                        default='goal',choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes'])
    parser.add_argument('--count', help='number of samples', type=np.int32, default=1280 * 20)
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

    #loop through(moving object and making
    train_data = np.empty([args.count, args.img_size, args.img_size, 3])
    i = 0
    while i < args.count:
        env.reset()
        for func in after_env_reset_ops[args.env]:
            func(env, args)
        if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
            rgb_array = take_goal_image(env, img_size=args.img_size)
        else:
            rgb_array = take_obstacle_image(env, img_size=args.img_size)
        train_data[i] = rgb_array.copy()
        i +=1
        for _ in range(3):
            if i < args.count:
                action = env.action_space.sample()
                env.step(action)
                if args.enc_type == 'goal' or args.enc_type == 'goal_sizes':
                    rgb_array = take_goal_image(env, img_size=args.img_size)
                else:
                    rgb_array = take_obstacle_image(env, img_size=args.img_size)
                train_data[i] = rgb_array.copy()
                i += 1
            else:
                break

    '''def show_some_sampled_images():
        n = 8
        a = None
        for i in range(n):
            b = None
            for j in range(n):
                j_im = train_data[np.random.randint(low=0, high=args.count)].copy()
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
    show_some_sampled_images()'''

    #store files
    np.save(data_file, train_data)
