from envs.utils import goal_distance
from utils.stable_baselines_plotter import plot_curves
import numpy as np
from j_vae.train_vae import load_Vae
from j_vae.generate_vae_data import random_pos_inside, size_file,random_size_at,generate_points
from j_vae.common_data import  min_obstacle_size, max_obstacle_size,  \
    range_x, range_y, obstacle_size, puck_size, z_table_height, center_obstacle
import torch
from torchvision.utils import save_image
from common import get_args
from envs import make_env
from j_vae.vae_mix_test import setup_env_sizes, take_goal_image, take_obstacle_image
import matplotlib.pyplot as plt
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from j_vae.latent_space_transformations import create_rotation_matrix, rotate_list_of_points, map_points, \
    goal_map_x, goal_map_y, obstacle_map_y, obstacle_map_x, angle_obstacle, angle_goal,\
    get_size_in_space, map_size_space, torch_get_size_in_space

using_sb = True
using_goal = False
img_size = 84
if not using_sb:
    if using_goal:
        model = load_Vae(path='../data/FetchPushObstacle/vae_model_obstacle', img_size=img_size, latent_size=2)
        size_to_use = 0.015
        corners_save_file = '../data/FetchPushObstacle/goal_corners.npy'
    else:
        model = load_Vae(path='../data/FetchPushObstacle/vae_model_goal', img_size=img_size, latent_size=2)
        size_to_use = obstacle_size
        corners_save_file = '../data/FetchPushObstacle/obstacle_corners.npy'
else:
    if using_goal:
        model = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_goal')
        size_to_use = 0.015
        corners_save_file = '../data/FetchPushObstacle/goal_corners.npy'
    else:
        model = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_obstacle')
        size_to_use = obstacle_size
        corners_save_file = '../data/FetchPushObstacle/obstacle_corners.npy'


def visualization_grid_points(n):
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

    args = get_args()
    env = make_env(args)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    setup_env_sizes(env)
    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    #move other objects to plaecs they do not disturb
    if using_goal:
        env.env.env._set_position(names_list=['obstacle'], position=[2.,2.,0.4])

    else:
        env.env.env._move_object(position=[2.,2.,0.4])
    for i,p in enumerate(points):
        if using_goal:
            env.env.env._move_object(position=p)
            data_set[i] = take_goal_image(env)
        else:
            env.env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env)
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    if not using_sb:
        mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
    else:
        mu, logvar = model.encode(data)
    #rec = vae_model_goal.decode(mu).reshape(-1, 3, img_size, img_size)
    mu = mu.detach().cpu().numpy()

    #images = data
    #save_image(images.cpu(), 'results/gridpoints.png',
    #           nrow=n)
    #save_image(rec.cpu(), 'results/gridpoints_reconstructed.png',
    #           nrow=n)


    if using_goal:
        #rm = create_rotation_matrix(angle_goal)
        #mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, goal_map_x, goal_map_y)
        pass
    else:
        #for i, p in enumerate(mu):
        #    mu[i] = reflect_obstacle_transformation(p)
        rm = create_rotation_matrix(angle_obstacle)
        mu = rotate_list_of_points(mu, rm)
        mu = map_points(mu, obstacle_map_x, obstacle_map_y)
        pass
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


def save_corners():
    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=2,
                             object_x_y_size=[size_to_use, size_to_use])
    args = get_args()
    env = make_env(args)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    setup_env_sizes(env)
    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    # move other objects to plaecs they do not disturb
    if using_goal:
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    else:
        env.env.env._move_object(position=[2., 2., 0.4])
    for i, p in enumerate(points):
        if using_goal:
            env.env.env._move_object(position=p)
            data_set[i] = take_goal_image(env)
        else:
            env.env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env)
    np.save(corners_save_file, data_set)


def visualization_sizes(n):
    from j_vae.train_vae import load_Vae
    vae_model_size = load_Vae(path='../data/FetchPushObstacle/vae_model_sizes', img_size=img_size, latent_size=1)

    sizes = np.linspace(min_obstacle_size, max_obstacle_size, num=n)
    #sizes = np.linspace(obstacle_size, obstacle_size, num=n)
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

    args = get_args()
    env = make_env(args)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    setup_env_sizes(env)
    # sample images
    data_set = np.empty([len(sizes), img_size, img_size, 3])
    for i, p in enumerate(sizes):
        env.env.env._set_position(names_list=['obstacle'], position=center_obstacle)
        env.env.env._set_size(names_list=['obstacle'], size=np.array([sizes[i], 0.035, 0.0]))
        data_set[i] = take_obstacle_image(env)
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = vae_model_size.encode(data.reshape(-1, img_size * img_size * 3))
    mu = mu.detach().cpu().numpy()

    #for i, p in enumerate(mu):
    #    mu[i] = get_size_in_space(mu[i])
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
    visualization_grid_points(7)
    #visualization_sizes(5)
    #save_corners()


