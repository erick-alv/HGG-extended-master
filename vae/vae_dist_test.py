import numpy as np
from common import get_args
from envs import make_env
from envs.utils import goal_distance
from utils.stable_baselines_plotter import plot_curves
import numpy as np
from vae.train_vae import load_Vae

from vae.train_vae_sb import load_Vae as load_Vae_SB
from vae.generate_vae_data import points_file, generate_points
from vae.common_data import obstacle_size
import torch
from torchvision.utils import save_image

using_sb = False
doing_goal = False
img_size = 84

if not using_sb:
    if doing_goal:
        vae_model = load_Vae(path='../data/FetchPushObstacle/vae_model_goal')
        size_to_use = 0.015
        data_set = np.load('../data/FetchPushObstacle/goal_set.npy')
    else:
        vae_model = load_Vae(path='../data/FetchPushObstacle/vae_model_obstacle')
        size_to_use = obstacle_size
        data_set = np.load('../data/FetchPushObstacle/obstacle_set.npy')
else:
    if doing_goal:
        vae_model = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_goal')
        size_to_use = 0.015
        data_set = np.load('../data/FetchPushObstacle/goal_set.npy')
    else:
        vae_model = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_obstacle')
        size_to_use = obstacle_size
        data_set = np.load('../data/FetchPushObstacle/obstacle_set.npy')



points = generate_points([1.05, 1.55], [0.5, 1.0], 0.43, 40, [size_to_use, size_to_use])




if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")


    start_index = 400
    end_index = 500
    to_compare_index = start_index - 1
    data = data_set[start_index - 1:end_index, :, :, :]

    data = torch.from_numpy(data).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    if not using_sb:
        mu_points, logvar_points = vae_model.encode(data.reshape(-1, img_size * img_size * 3))
    else:
        mu_points, logvar_points = vae_model.encode(data)
    #reparametrized_points = vae_model.reparameterize(mu_points, logvar_points)
    mu_points = mu_points.detach().cpu().numpy()
    #reparametrized_points = reparametrized_points.detach().cpu().numpy()


    # Generate VAE-dataset
    # Train VAE
    # Show mocap settings in robot_env.py!
    dist_latent = []
    #dist_latent_reparametrized = []
    dist_real = []
    for i in range(start_index, end_index):
        dist_real.append(goal_distance(np.array(points[to_compare_index]), np.array(points[i])))
        #the latent have one element more and the point to compare is the first one
        j = i+1-start_index
        dist_latent.append(goal_distance(mu_points[0], mu_points[j]))
        #dist_latent_reparametrized.append(goal_distance(reparametrized_points[0], reparametrized_points[j]))

    dist_real = np.array(dist_real)
    dist_latent = np.array(dist_latent)
    #dist_latent_reparametrized = np.array(dist_latent_reparametrized)
    xs = np.arange(0, len(dist_latent))
    plot_curves([(xs, dist_real)], 'step', 'distance',
                window=1, labels=['real'], title='distances_real', #title='distances_real',
                filename='results/distances_real')
    plot_curves([(xs, dist_latent)], 'step', 'distance',
                window=1, labels=['latent'], title='distances_latent',#title='distances_latent',
                filename='results/distances_latent')
    #plot_curves([(xs, dist_latent)], 'step', 'distance',
    #            window=1, labels=['latent_reparametrized'], title='distances_latent_reparametrized',  # title='distances_latent',
    #            filename='/home/erick/RL/HGG-extended/HGG-Extended-master/vae/distances_latent_reparametrized')
    images = data
    if doing_goal:
        fn = 'results/env_goals.png'
    else:
        fn = 'results/env_obstacles.png'

    save_image(images.cpu(), fn, nrow=10)



