from envs.utils import goal_distance
from utils.stable_baselines_plotter import plot_curves
import numpy as np

from j_vae.generate_vae_data import random_pos_inside, size_file,random_size_at,generate_points
from j_vae.common_data import  min_obstacle_size, max_obstacle_size, range_x, range_y, obstacle_size, \
    puck_size, z_table_height, center_obstacle, train_file_name, vae_sb_latent_size, vae_sb_weights_file_name
import torch
import matplotlib.pyplot as plt
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from envs import make_env
from vae_env_inter import take_goal_image, take_obstacle_image
from j_vae.train_vae import load_Vae
from j_vae.latent_space_transformations import create_rotation_matrix, rotate_list_of_points, map_points, \
    goal_map_x, goal_map_y, obstacle_map_y, obstacle_map_x, angle_obstacle, angle_goal,\
    get_size_in_space, map_size_space, torch_get_size_in_space


def visualization_grid_points(env, model, size_to_use, img_size, n, enc_type, using_sb=True,
                              select_components=False, comp_ind_1=None, comp_ind_2=None):
    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=n,
                             object_x_y_size=[size_to_use, size_to_use])

    n_labels = np.arange(len(points))

    points = np.array(points)
    print_max_and_min(points)

    xs = points[:, 0]
    ys = points[:, 1]
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(xs,ys)
    plt.title('real')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (xs[i], ys[i]))


    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    #move other objects to plaecs they do not disturb
    if enc_type == 'goal':
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    elif enc_type == 'obstacle':
        env.env.env._move_object(position=[2.,2.,0.4])
    else:
        raise Exception('Not supported enc type')
    for i,p in enumerate(points):
        if enc_type == 'goal':
            env.env.env._move_object(position=p)
            data_set[i] = take_goal_image(env, img_size)
        elif enc_type == 'obstacle':
            env.env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env, img_size)
        else:
            raise Exception('Not supported enc type')
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    if not using_sb:
        mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
    else:
        mu, logvar = model.encode(data)
    mu = mu.detach().cpu().numpy()

    if select_components:
        assert comp_ind_1 is not None and comp_ind_2 is not None and comp_ind_1 != comp_ind_2
        mu = np.concatenate([np.expand_dims(mu[:, comp_ind_1], axis=1),
                             np.expand_dims(mu[:, comp_ind_2], axis=1)], axis=1)


    if enc_type == 'goal':
        #rm = create_rotation_matrix(angle_goal)
        #mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, goal_map_x, goal_map_y)
        pass
    elif enc_type == 'obstacle':
        #for i, p in enumerate(mu):
        #    mu[i] = reflect_obstacle_transformation(p)
        #rm = create_rotation_matrix(angle_obstacle)
        #mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, obstacle_map_x, obstacle_map_y)
        pass
    else:
        raise Exception('Not supported enc type')
    print_max_and_min(mu)

    lxs = mu[:, 0]
    lys = mu[:, 1]
    plt.subplot(212)
    plt.scatter(lxs, lys)
    plt.title('latent')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (lxs[i], lys[i]))

    plt.show()
    plt.close()

def print_max_and_min(points):
    assert isinstance(points, np.ndarray)
    xs = points[:, 0]
    ys = points[:, 1]


    print('max x: {}'.format(xs.max()))
    print('min x: {}'.format(xs.min()))
    print('max y: {}'.format(ys.max()))
    print('min y: {}'.format(ys.min()))


def save_corners(env, size_to_use, file_corners, img_size, enc_type):
    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=2,
                             object_x_y_size=[size_to_use, size_to_use])
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    # move other objects to plaecs they do not disturb
    if enc_type == 'goal':
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    elif enc_type == 'obstacle':
        env.env.env._move_object(position=[2., 2., 0.4])
    else:
        raise Exception('Not supported enc_type')
    for i, p in enumerate(points):
        if enc_type == 'goal':
            env.env.env._move_object(position=p)
            data_set[i] = take_goal_image(env, img_size)
        elif enc_type == 'obstacle':
            env.env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env, img_size)
        else:
            raise Exception('Not supported enc_type')
    np.save(file_corners, data_set)


def visualization_sizes(env, model, img_size, n):
    #sizes = np.linspace(min_obstacle_size, max_obstacle_size, num=n)
    sizes = np.linspace(obstacle_size, obstacle_size, num=n)
    n_labels = np.arange(len(sizes))

    sizes = np.array(sizes)

    xs = np.repeat(1, len(sizes))
    ys = sizes
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(xs, ys)
    plt.title('real')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (xs[i], ys[i]))

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")


    # sample images
    data_set = np.empty([len(sizes), img_size, img_size, 3])
    for i, p in enumerate(sizes):
        env.env.env._set_position(names_list=['obstacle'], position=center_obstacle)
        env.env.env._set_size(names_list=['obstacle'], size=np.array([sizes[i], 0.035, 0.0]))
        data_set[i] = take_obstacle_image(env, img_size)
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
    mu = mu.detach().cpu().numpy()

    for i, p in enumerate(mu):
        mu[i] = get_size_in_space(mu[i])
    #    mu[i] = map_size_space(mu[i])
    #    mu[i] = get_size_in_space(map_size_space(mu[i]))

    lxs = np.repeat(1, len(sizes))
    lys = mu[:, 0]

    plt.subplot(212)
    plt.scatter(lxs, lys)
    plt.title('latent')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (lxs[i], lys[i]))

    plt.show()
    plt.close()



if __name__ =='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1')
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
                        default='goal', choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes'])
    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)

    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    base_data_dir = this_file_dir + '../data/'
    data_dir = base_data_dir + args.env + '/'
    train_file = data_dir + train_file_name[args.enc_type]
    weights_path = data_dir + vae_sb_weights_file_name[args.enc_type]

    # load the latent_size and model
    args.latent_size = vae_sb_latent_size[args.enc_type]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = load_Vae_SB(weights_path, args.img_size, args.latent_size)

    # load environment
    env = make_env(args)

    #other arguments for the algorithms
    if args.enc_type == 'goal':
        size_to_use = 0.15 #
    elif args.enc_type == 'obstacle':
        size_to_use = obstacle_size

    visualization_grid_points(n=7, env=env, model=model,size_to_use=size_to_use, img_size=args.img_size,
                              enc_type=args.enc_type, select_components=True, comp_ind_1=5, comp_ind_2=9)


