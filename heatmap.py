import numpy as np
import time
from common import get_args, experiment_setup, load_vaes, make_env, load_field_parameters, load_dist_estimator
from vae_env_inter import take_env_image, take_image_objects, latents_from_images
import copy
from vae.latent_space_transformations import interval_map_function
import matplotlib.pyplot as plt
from play import Player
from algorithm.replay_buffer import goal_based_process
from utils.image_util import create_rollout_video
from collections import namedtuple
from PIL import Image
import seaborn as sns
from gym.envs.robotics import rotations, robot_env, utils


def _get_obs_gripper_at(env, position):
    gripper_target = np.array([position[0], position[1], position[2]])

    # positions
    grip_pos = env.env.env.sim.data.get_site_xpos('robot0:grip')
    dt = env.env.env.sim.nsubsteps * env.env.env.sim.model.opt.timestep
    grip_velp = env.env.env.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(env.env.env.sim)
    if env.env.env.has_object:
        object_pos = env.env.env.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(env.env.env.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = env.env.env.sim.data.get_site_xvelp('object0') * dt
        object_velr = env.env.env.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
    else:
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not env.env.env.has_object:
        achieved_goal = grip_pos.copy()
    else:
        achieved_goal = np.squeeze(object_pos.copy())

    obs = np.concatenate([
        grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    ])

    # env.env.env._get_image('obs.png')

    return {
        'observation': obs.copy(),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': env.env.env.goal.copy(),
    }

def create_heatmap_qvalues(args, env, player, player_imaginary):
    nx = 80#todo at least 100
    ny = 80

    pos_x = np.linspace(args.real_field_center[0] - args.real_field_size[0], args.real_field_center[0] + args.real_field_size[0], num=nx, endpoint=True)
    pos_y = np.linspace(args.real_field_center[1] - args.real_field_size[1], args.real_field_center[1] + args.real_field_size[1], num=ny, endpoint=True)

    first_ob = env.reset()
    puck_first_pos = first_ob['achieved_goal']
    grip_pos = env.env.env.sim.data.get_site_xpos('robot0:grip')
    dt = env.env.env.sim.nsubsteps * env.env.env.sim.model.opt.timestep
    grip_velp = env.env.env.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(env.env.env.sim)
    for timestep in range(17):
        obs_orginal = []
        obs = []
        data_v_vals = np.zeros(shape=(nx, ny))
        data_v_vals_imaginary = np.zeros(shape=(nx, ny))

        #pass some steps so that moving obstacles are in other part
        #move puck and robot back to beginning
        #show image

        do_steps = 3
        while do_steps > 0:
            do_steps -= 1
            action = [0., 0., 0., 0.]
            env.step(action)

        env_image = take_env_image(env, 500)
        im_current = Image.fromarray(env_image.astype(np.uint8))
        im_current.save('log/{}_env.png'.format(args.env))

        for i, px in enumerate(pos_x):
            for j, py in enumerate(pos_y):
                env.env.env._move_gripper_to(position=[px, py + 0.1, 0.61])
                '''try:
                    env.env.env._move_gripper_to(position=[px, py + 0.15, 0.52])
                except Exception:
                    print('no work with x={}, y={} i={}, j={}'.format(px,py,i, j) )
                    exit()'''
                env.env.env._move_object(position=[px, py, 0.425])#todo use height of env



                # move puck and robot back to beginning
                # show image
                env_image = take_env_image(env, 500)
                im_current = Image.fromarray(env_image.astype(np.uint8))
                #im_current.save('log/{}_env.png'.format(args.env))
                if (i > 0) and (i % 10 == 0) and (j % 10 == 0):
                    im_current.save('log/{}_env_from_{}_{}.png'.format(args.env, i, j))
                    print('d')
                o = env.get_obs()
                obs_orginal.append(o)
                obs.append(goal_based_process(o))
                q_val = player.get_q_pi([goal_based_process(o)])[0]
                q_val_imaginary = player_imaginary.get_q_pi([goal_based_process(o)])[0]

                data_v_vals[i, j] = q_val
                data_v_vals_imaginary[i, j] = q_val_imaginary
            fwef = 4234
        rq4r4 = 43231

        ax = plt.gca()
        im = ax.imshow(data_v_vals, cmap='cool', interpolation='nearest')
        # Create colorbar
        cbar_kw = {}
        cbarlabel = ""
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.title('heatmap')
        plt.savefig('log/{}_valuemap_{}.png'.format(args.env, timestep))
        plt.clf()
        plt.close()

        #for imaginary values
        ax = plt.gca()
        im = ax.imshow(data_v_vals_imaginary, cmap='cool', interpolation='nearest')
        # Create colorbar
        cbar_kw = {}
        cbarlabel = ""
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        plt.title('heatmap with imaginary')
        plt.savefig('log/{}_valuemap_{}_imaginary.png'.format(args.env, timestep))
        plt.clf()
        plt.close()



if __name__ == '__main__':

    # Getting arguments from command line + defaults
    args = get_args()
    # creates copy of args for the real coordinates
    # this class compares space generated by neuralt network (in this case Bbox) with real coordinates)
    env = make_env(args)
    load_field_parameters(args)
    assert args.play_path is not None
    player = Player(args)
    player_imaginary = Player(args, direct_playpath=args.play_path_im_h)

    create_heatmap_qvalues(args=args, env=env, player=player, player_imaginary=player_imaginary)