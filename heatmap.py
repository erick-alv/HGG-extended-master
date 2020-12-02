import numpy as np
import time
from common import get_args, experiment_setup, load_vaes, make_env, load_field_parameters, load_dist_estimator
from vae_env_inter import take_env_image, take_image_objects, latents_from_images
import copy
from j_vae.latent_space_transformations import interval_map_function
import matplotlib.pyplot as plt
from play import Player
from algorithm.replay_buffer import goal_based_process
from utils.image_util import create_rollout_video
from collections import namedtuple
from PIL import Image
import seaborn as sns

def create_heatmap_qvalues(args, env, player):
    nx = 100
    ny = 100

    pos_x = np.linspace(1.05, 1.55, num=nx, endpoint=True)
    pos_y = np.linspace(0.5, 1.0, num=ny, endpoint=True)

    for timestep in range(10):
        obs_orginal = []
        obs = []
        env.reset()
        data = np.zeros(shape=(nx, ny))

        #pass some steps so that moving obstacles are in other part
        do_steps = timestep
        while do_steps > 0:
            do_steps -= 1
            action = [0., 0., 0., 0.]
            env.step(action)

        env_image = take_env_image(env, args.img_size)
        im_current = Image.fromarray(env_image.astype(np.uint8))
        im_current.save('log/heatmaps/env_at_timestep_{}.png'.format(timestep))

        for i, px in enumerate(pos_x):
            for j, py in enumerate(pos_y):
                env.env.env._move_object(position=[px, py, 0.425])#todo use height of env
                '''for timestep in range(3):
                    action = [0., 0., 0., 0.]
                    o, _, _, info = env.step(action)
                    obs.append(o)
                    env_images.append(take_env_image(env, args.img_size))'''
                o = env.get_obs()
                obs_orginal.append(o)
                obs.append(goal_based_process(o))
                q_val = player.get_q_pi([goal_based_process(o)])[0]
                data[i, j] = q_val

        ax = plt.gca()
        im = ax.imshow(data, cmap='cool', interpolation='nearest')
        # Create colorbar
        cbar_kw = {}
        cbarlabel = ""
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.title('heatmap')
        plt.savefig('log/heatmaps/map_at_timestep_{}.png'.format(timestep))
        plt.clf()



if __name__ == '__main__':

    # Getting arguments from command line + defaults
    args = get_args()
    # creates copy of args for the real coordinates
    # this class compares space generated by neuralt network (in this case Bbox) with real coordinates)
    env = make_env(args)
    load_field_parameters(args)
    assert args.play_path is not None
    player = Player(args)

    create_heatmap_qvalues(args=args, env=env, player=player)