import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()
import argparse
from plot import smooth_reward_curve, load_results, pad

from common import get_args
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from j_vae.train_vae import load_Vae
from j_vae.common_data import vae_sb_weights_file_name, vae_weights_file_name
from envs import make_env

def show_points(points_list, save_file, space_of):
    if space_of == 'real':
        support_points = np.array([[1.05, 0.5, 0.43], [1.05, 1.0, 0.43], [1.55, 0.5, 0.43], [1.55, 1.0, 0.43]])
    elif space_of == 'latent':
        support_points = np.array([[-1., -1., 0.], [-1., 1, 0.], [1, -1., 0.], [1, 1, 0.]])
    
    if points_list.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2])
        ax.scatter(support_points[:, 0], support_points[:, 1], support_points[:, 2], c='red')
        plt.savefig('{}_3D'.format(save_file))
        plt.close()
        show_points(points_list[:, :2], save_file+'_2D', space_of)
    elif points_list.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(points_list[:, 0], points_list[:, 1])
        ax.scatter(support_points[:, 0], support_points[:, 1], c='red')
        plt.savefig(save_file)
        plt.close()
    else:
        print('cannot create visualization of shape {}'.format(points_list.shape))

def compare_points(env, save_file, args):
    from j_vae.latent_space_transformations import table_map_x, table_map_y
    support_points = np.array([[-1., -1., 0.], [-1., 1, 0.], [1, -1., 0.], [1, 1, 0.]])
    obs = env.reset()
    xs = table_map_x(np.array([obs['achieved_goal'][0], obs['desired_goal'][0]]))
    ys = table_map_y(np.array([obs['achieved_goal'][1], obs['desired_goal'][1]]))
    real_points = np.concatenate([np.expand_dims(xs,axis=1), np.expand_dims(ys, axis=1)], axis=1)
    latent_points = np.array([obs['achieved_goal_latent'][:2], obs['desired_goal_latent'][:2]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(real_points[:, 0], real_points[:, 1], c='blue')
    a_circle = plt.Circle((0., 0.), 0.52, alpha=0.3, color='blue')
    ax.add_artist(a_circle)
    ax.scatter(latent_points[:, 0], latent_points[:, 1], c='green')
    b_circle = plt.Circle((obs['obstacle_latent'][0], obs['obstacle_latent'][1]), obs['obstacle_size_latent'],
                          alpha=0.3, color='green')
    ax.add_artist(b_circle)
    ax.scatter(support_points[:, 0], support_points[:, 1], c='red')
    plt.savefig(save_file)
    plt.close()

if __name__ == '__main__':
    args = get_args()

    base_data_dir = 'data/'
    data_dir = base_data_dir + args.env + '/'
    weights_path_goal = data_dir + vae_sb_weights_file_name['goal']
    args.weights_path_goal = weights_path_goal
    weights_path_obstacle = data_dir + vae_sb_weights_file_name['obstacle']
    args.weights_path_obstacle = weights_path_obstacle
    weights_path_obstacle_sizes = data_dir + vae_weights_file_name['obstacle_sizes']
    args.weights_path_obstacle_sizes = weights_path_obstacle_sizes
    args.vae_model_obstacle = load_Vae_SB(weights_path_obstacle, args.img_size, args.latent_size_obstacle)
    args.vae_model_obstacle.eval()
    args.vae_model_goal = load_Vae_SB(weights_path_goal, args.img_size, args.latent_size_goal)
    args.vae_model_goal.eval()
    args.vae_model_size = load_Vae(path=weights_path_obstacle_sizes, img_size=args.img_size, latent_size=1)
    args.vae_model_size.eval()

    env = make_env(args)
    compare_points(env, 'somefile.png', args)
