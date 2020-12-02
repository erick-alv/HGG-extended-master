import numpy as np
import time
from common import get_args, experiment_setup, load_vaes, make_env, load_field_parameters, \
    load_dist_estimator, make_temp_env
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


def create_heatmap_distances(args, env):
    nx = 100
    ny = 100

    pos_x = np.linspace(1.05, 1.55, num=nx, endpoint=True)
    pos_y = np.linspace(0.5, 1.0, num=ny, endpoint=True)

    for timestep in range(10):
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
        im_current.save('log/heatmaps/distance_env_at_timestep_{}.png'.format(timestep))

        for i, px in enumerate(pos_x):
            for j, py in enumerate(pos_y):
                env.env.env._move_object(position=[px, py, 0.425])#todo use height of env
                o = env.get_obs()
                if args.vae_dist_help:
                    # have to render here since get obs wont actualize the image position (internally just updated on step)
                    image = take_image_objects(env, args.img_size)
                    if args.vae_type == 'space' or args.vae_type == 'bbox' or args.vae_type == 'faster_rcnn':
                        lg, lg_s, lo, lo_s = latents_from_images(np.array([image]), args)
                        #achieved_goal_size_latent = lg_s[0].copy()
                    else:
                        lg, lo, lo_s = latents_from_images(np.array([image]), args)
                    achieved_goal_latent = lg[0].copy()
                    #obstacle_latent = lo[0].copy()
                    #obstacle_size_latent = lo_s[0].copy()


                    distance_to_goal = args.dist_estimator.calculate_distance_batch(
                        goal_pos=o['desired_goal_latent'].copy(),
                        current_pos_batch=np.array([achieved_goal_latent]))[0]
                    #clipped to 100 since estimator puts regions inside obstacle too far away and
                    #then heatmap does not represent any important info (these clipping depend of env)
                    distance_to_goal = np.clip(distance_to_goal, a_min=None, a_max=3.2)
                else:

                    distance_to_goal = args.dist_estimator.calculate_distance_batch(
                        goal_pos=o['desired_goal'].copy(),
                        current_pos_batch=np.array([o['achieved_goal']]))[0]
                    # clipped to 0.8 since estimator puts regions inside obstacle too far away and
                    # then heatmap does not represent any important info
                    distance_to_goal = np.clip(distance_to_goal, a_min=None, a_max=0.8)
                data[i, j] = distance_to_goal

        ax = plt.gca()
        im = ax.imshow(data, cmap='viridis', interpolation='nearest')
        # Create colorbar
        cbar_kw = {}
        cbarlabel = ""
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.title('heatmap')
        plt.savefig('log/heatmaps/distance_map_at_timestep_{}.png'.format(timestep))
        plt.clf()



if __name__ == '__main__':

    # Getting arguments from command line + defaults
    args = get_args()
    # creates copy of args for the real coordinates
    # this class compares space generated by neuralt network (in this case Bbox) with real coordinates)
    if args.vae_dist_help:
        load_vaes(args)

    # since some extensions of the envs use the distestimator this load is used with the interval wrapper#todo use other?
    load_field_parameters(args)
    assert args.dist_estimator_type is not None
    temp_env = make_temp_env(args)
    load_dist_estimator(args, temp_env)
    del temp_env

    env = make_env(args)


    #LoggerMock = namedtuple('Logger', ['my_log_dir'])#todo needed?
    #args.logger = LoggerMock(my_log_dir='log/space_tests/')

    create_heatmap_distances(args=args, env=env)