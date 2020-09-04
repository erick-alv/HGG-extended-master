from envs.utils import goal_distance
from utils.stable_baselines_plotter import plot_curves
import numpy as np
from j_vae.train_vae import load_Vae
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from j_vae.generate_vae_data import random_pos_inside, size_file,random_size_at
from j_vae.common_data import  min_obstacle_size, max_obstacle_size,  \
    range_x, range_y, obstacle_size, puck_size
import torch
from torchvision.utils import save_image
from common import get_args
from envs import make_env
from j_vae.latent_space_transformations import obstacle_transformation, goal_transformation

from PIL import Image

using_sb = True

img_size = 84
if not using_sb:
    vae_model_obstacle = load_Vae(path='../data/FetchPushObstacle/vae_model_obstacle')
    vae_model_goal = load_Vae(path='../data/FetchPushObstacle/vae_model_goal')
else:
    vae_model_obstacle = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_obstacle')
    vae_model_goal = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_goal')



number_samples = 5
def multiple_points_test():
    args = get_args()
    env = make_env(args)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    setup_env_sizes(env)
    obstacle_points = []
    for _ in range(3):
        p_o = np.array(random_pos_inside(range_x=range_x, range_y=range_y, z=0.435,
                                         object_x_y_size=[obstacle_size, obstacle_size]))
        obstacle_points.append(p_o)
    for p_o_index, p_o in enumerate(obstacle_points):
        goals_points = []
        while len(goals_points) < number_samples:
            #puck can be at edge
            p_g = np.array(random_pos_inside(range_x=range_x, range_y=range_y, z=0.435, object_x_y_size=[0.15, 0.15]))
            if goal_distance(p_g, p_o) >= obstacle_size+puck_size:
                goals_points.append(p_g)

        #sample for obstacle
        # move temporally object to another position
        env.env.env._move_object(position=[2.0,2.0,2.0])
        env.env.env._set_position(names_list=['obstacle'], position=p_o)
        im_o = take_obstacle_image(env)
        im_o = torch.from_numpy(im_o).float().to(device)
        im_o /= 255
        im_o = im_o.permute([2, 0, 1])
        if not using_sb:
            mu_o, logvar_o = vae_model_obstacle.encode(im_o.reshape(-1, img_size * img_size * 3))
        else:
            im_o = torch.unsqueeze(im_o, dim=0)
            mu_o, logvar_o = vae_model_obstacle.encode(im_o)
        mu_o = mu_o.detach().cpu().numpy()

        #sample for goals
        data_set = np.empty([number_samples, img_size, img_size, 3])
        for i in range(number_samples):
            env.env.env._move_object(position=goals_points[i])
            data_set[i] = take_goal_image(env)
        data = torch.from_numpy(data_set).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        if not using_sb:
            mu, logvar = vae_model_goal.encode(data.reshape(-1, img_size * img_size * 3))
        else:
            mu, logvar = vae_model_goal.encode(data)
        mu = mu.detach().cpu().numpy()

        if using_sb:
            transformed_mu_o = obstacle_transformation(mu_o)
            mu_o = transformed_mu_o[0]
            #mu_o = mu_o[0]
            mu = goal_transformation(mu)

        distances_latent = []
        distances_real = []
        for i in range(number_samples):
            distances_real.append(goal_distance(goals_points[i], p_o))
            distances_latent.append(goal_distance(mu[i], mu_o))

        distances_latent = np.array(distances_latent)
        distances_real = np.array(distances_real)
        xs = np.arange(0, len(distances_latent))
        plot_curves([(xs, distances_real)], '', 'distance',
                    window=1, labels=['real'], title='distance_real obstacle at coordinate ({},{})'.format(p_o[0], p_o[1]),
                    filename='results/distance_real_obstacle_{}'.format(p_o_index))
        plot_curves([(xs, distances_latent)], '', 'distance',
                    window=1, labels=['latent'], title='distance_latent obstacle at coordinate ({},{})'.format(p_o[0], p_o[1]),
                    filename='results/distance_latent_obstacle_{}'.format(p_o_index))
        if using_sb:
            images = torch.cat([im_o, data])
        else:
            images = torch.cat([torch.unsqueeze(im_o, 0), data])
        save_image(images.cpu(), 'results/envdistance_obstacle_{}.png'.format(p_o_index), nrow=10)

def setup_env_sizes(env):
    env.env.env._set_size(names_list=['object0'], size=[puck_size, 0.035, 0.0])
    env.env.env._set_size(names_list=['obstacle'], size=[obstacle_size, 0.035, 0.0])

def take_obstacle_image(env):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=0.0)
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    #img = Image.fromarray(rgb_array)
    #img.show()
    #img.close()
    return rgb_array

def take_goal_image(env):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=1.0)
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    #img = Image.fromarray(rgb_array)
    #img.show()
    #img.close()
    return rgb_array


if __name__ == '__main__':
    multiple_points_test()