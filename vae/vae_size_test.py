from envs.utils import goal_distance
from utils.stable_baselines_plotter import plot_curves
import numpy as np
from j_vae.train_vae import load_Vae
from j_vae.generate_vae_data import random_pos_inside, size_file,random_size_at
from j_vae.common_data import min_obstacle_size, max_obstacle_size, \
    range_x, range_y, obstacle_size, upper_limits, lower_limits
import torch
from torchvision.utils import save_image
from common import get_args
from envs import make_env

img_size = 84
vae_model = load_Vae(path='../data/FetchPushObstacle/vae_model_obstacle_sizes')



def just_sizes_test():
    data_set = np.load('../data/FetchPushObstacle/obstacle_sizes_set.npy')
    sizes = np.load(size_file)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    idx = np.random.randint(low=0, high=len(sizes), size=20)
    data = data_set[idx]
    data = torch.from_numpy(data).float().to(device)
    del data_set
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = vae_model.encode(data.reshape(-1, img_size * img_size * 3))
    mu = mu.detach().cpu().numpy()

    size_latent = []
    size_real = []
    for t, index in enumerate(idx):
        size_real.append(sizes[index][0])
        size_latent.append(mu[t])

    size_latent = np.array(size_latent)
    size_real = np.array(size_real)
    xs = np.arange(0, len(size_latent))
    plot_curves([(xs, size_real)], '', 'size',
                window=1, labels=['real'], title='size_real',
                filename='results/size_real')
    plot_curves([(xs, size_latent)], '', 'size',
                window=1, labels=['latent'], title='size_latent',
                filename='results/size_latent')
    images = data
    save_image(images.cpu(), 'results/env_sizes_at_center.png')

number_samples = 20
def multiple_points_sizes_test():
    args = get_args()
    env = make_env(args)
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=0.0)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    points = []
    for _ in range(3):
        p = random_pos_inside(range_x=range_x, range_y=range_y, z=0.435, object_x_y_size=[obstacle_size, obstacle_size])
        points.append(p)
    for p_index, p in enumerate(points):
        sizes = []
        data_set = np.empty([number_samples, img_size, img_size, 3])
        env.env.env._set_position(names_list=['obstacle'], position=p)
        for i in range(number_samples):
            s = random_size_at(min_size=min_obstacle_size, max_size=max_obstacle_size,
                                        lower_limits=lower_limits,upper_limits=upper_limits, pos=p)
            sizes.append(s)
            env.env.env._set_size(names_list=['obstacle'], size=s)
            rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
            data_set[i] = rgb_array
        data = torch.from_numpy(data_set).float().to(device)
        del data_set
        data /= 255
        data = data.permute([0, 3, 1, 2])
        mu, logvar = vae_model.encode(data.reshape(-1, img_size * img_size * 3))
        mu = mu.detach().cpu().numpy()

        size_latent = []
        size_real = []
        for i in range(number_samples):
            size_real.append(sizes[i][0])
            size_latent.append(mu[i])

        size_latent = np.array(size_latent)
        size_real = np.array(size_real)
        xs = np.arange(0, len(size_latent))
        plot_curves([(xs, size_real)], '', 'size',
                    window=1, labels=['real'], title='size_real point at coordinate ({},{})'.format(p[0], p[1]),
                    filename='results/size_real_point_{}'.format(p_index))
        plot_curves([(xs, size_latent)], '', 'size',
                    window=1, labels=['latent'], title='size_latent point at coordinate ({},{})'.format(p[0], p[1]),
                    filename='results/size_latent_point_{}'.format(p_index))
        images = data
        save_image(images.cpu(), 'results/env_sizes_point_{}.png'.format(p_index))


if __name__ == '__main__':
    #just_sizes_test()
    multiple_points_sizes_test()
