import numpy as np
import torch
import os
import copy
from utils.os_utils import make_dir
from common import get_args, load_vaes, load_field_parameters
import gym
from vae_env_inter import take_image_objects


def extract_info(images, args):
    with torch.no_grad():
        args.vae_model.eval()
        images = torch.from_numpy(images).float().to(args.device)
        images /= 255.
        images = images.permute([0, 3, 1, 2])

        # (B, K, D)
        z_pres, z_depth, z_scale, z_pos = args.vae_model.encode(images)
        z_pres, z_depth, z_scale, z_pos = z_pres.detach().cpu().numpy(), z_depth.detach().cpu().numpy(), \
                                          z_scale.detach().cpu().numpy(), z_pos.detach().cpu().numpy()


    return z_pres, z_depth, z_scale, z_pos

if __name__ == "__main__":
    args = get_args()
    # create data folder if it does not exist, corresponding folders, and files where to store data
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    base_data_dir = this_file_dir + 'data/'
    env_data_dir = base_data_dir + args.env + '/'
    make_dir(env_data_dir, clear=False)

    assert args.vae_dist_help
    load_vaes(args, doing_inference=True)
    load_field_parameters(args)
    env = gym.make(args.env)

    pres = []
    scale = []
    pos = []
    for rs in range(10):

        env.reset()
        image = take_image_objects(None, args.img_size, direct_env=env.env)
        #im_current = Image.fromarray(image.astype(np.uint8))
        #im_current.save('env_image_for_vae.png')
        z_pres, z_depth, z_scale, z_pos = extract_info(np.array([image]), args)
        pres.append(z_pres[0])
        scale.append(z_scale[0])
        pos.append(z_pos[0])


        for timestep in range(50):
            # get action from the ddpg policy
            action = env.action_space.sample()
            no_action = np.zeros_like(action)
            obs, _, _, _ = env.step(no_action)
            image = take_image_objects(None, args.img_size, direct_env=env.env)
            #im_current = Image.fromarray(image.astype(np.uint8))
            #im_current.save('env_image_for_vae.png')
            z_pres, z_depth, z_scale, z_pos = extract_info(np.array([image]), args)
            pres.append(z_pres[0])
            scale.append(z_scale[0])
            pos.append(z_pos[0])
    stacked_pres = np.stack(pres, axis=0)
    pres_mean = np.mean(stacked_pres, axis=0)
    stacked_pos = np.stack(pos, axis=0)
    pos_mean = np.mean(stacked_pos, axis=0)
    pos_std = np.std(stacked_pos, axis=0)
    stacked_scale = np.stack(scale, axis=0)
    scale_mean = np.mean(stacked_scale, axis=0)
    scale_std = np.std(stacked_scale, axis=0)

    indices = np.arange(len(pres_mean))
    present = pres_mean[:, 0] > 0.7
    indices = indices[present]
    pssss = pos_std[indices]

    scale2 = []
    pos2 = []
    for rs in range(10):

        env.reset()
        image = take_image_objects(None, args.img_size, direct_env=env.env)
        z_pres, z_depth, z_scale, z_pos = extract_info(np.array([image]), args)
        scale2.append(z_scale[0])
        pos2.append(z_pos[0])

        for timestep in range(50):
            # get action from the ddpg policy
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            image = take_image_objects(None, args.img_size, direct_env=env.env)
            z_pres, z_depth, z_scale, z_pos = extract_info(np.array([image]), args)
            scale2.append(z_scale[0])
            pos2.append(z_pos[0])
    stacked_pos2 = np.stack(pos2, axis=0)
    pos_mean2 = np.mean(stacked_pos2, axis=0)
    pos_std2 = np.std(stacked_pos2, axis=0)
    stacked_scale2 = np.stack(scale2, axis=0)
    scale_mean2 = np.mean(stacked_scale2, axis=0)
    scale_std2 = np.std(stacked_scale2, axis=0)

    pos_std_dif = np.square(np.subtract(pos_std[indices], pos_std2[indices]))
    pos_std_mean_error = np.mean(pos_std_dif, axis=1)
    pos_mean_dif = np.square(np.subtract(pos_mean[indices], pos_mean2[indices]))
    pos_mean_mean_error = np.mean(pos_mean_dif, axis=1)

    scale_std_dif = np.square(np.subtract(scale_std[indices], scale_std2[indices]))
    scale_std_mean_error = np.mean(scale_std_dif, axis=1)
    scale_mean_dif = np.square(np.subtract(scale_mean[indices], scale_mean2[indices]))
    scale_mean_mean_error = np.mean(scale_mean_dif, axis=1)

    object_index = np.argmax(pos_mean_mean_error)
    object_index = indices[object_index]
    obstacle_indices = np.setdiff1d(indices, np.array([object_index]))
    np.save(env_data_dir+args.vae_type+'_obj_i', object_index)
    np.save(env_data_dir+args.vae_type+'_obstacles_indices', obstacle_indices)
    print('obj index is {}'.format(object_index))
    print('obstacle indices are {}'.format(obstacle_indices))
    print('finish')