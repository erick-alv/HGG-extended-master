import numpy as np
import copy
import torch
from j_vae.train_vae_sb import VAE_SB
from j_vae.train_vae import VAE
from j_vae.train_monet import Monet_VAE

from j_vae.latent_space_transformations import torch_goal_transformation, torch_obstacle_transformation, \
    torch_get_size_in_space


'''def setup_env_sizes(env, object_size=None, size_of_obstacle=None):
    if object_size is not None:
        env.env.env._set_size(names_list=['object0'], size=[object_size, 0.035, 0.0])
    else:
        env.env.env._set_size(names_list=['object0'], size=[puck_size, 0.035, 0.0])
    if size_of_obstacle is not None:
        env.env.env._set_size(names_list=['obstacle'], size=[size_of_obstacle, 0.035, 0.0])
    else:
        env.env.env._set_size(names_list=['obstacle'], size=[obstacle_size, 0.035, 0.0])'''


# todo set to false once is trained with table
def take_obstacle_image(env, img_size, make_table_invisible=True):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=0.0)
    if not make_table_invisible:
        env.env.env._set_visibility(names_list=['table0'], alpha_val=1.0)
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array

def take_goal_image(env, img_size, make_table_invisible=True, make_walls_invisible=True):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=1.0)
    if not make_table_invisible:
        env.env.env._set_visibility(names_list=['table0'], alpha_val=1.0)
    if not make_walls_invisible:
        if 'wall1' in env.env.env.sim.model.body_names:
            env.env.env._set_visibility(names_list=['wall1', 'wall2', 'wall3', 'wall4'], alpha_val=1.0)
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array

def take_env_image(env, img_size):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['table0'], alpha_val=1.0)
    for id in  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17, 18, 21]:
        env.env.env._set_visibility_with_id(id, alpha_val=0.2)
    #just to activate in case viewer is not intialized
    if not hasattr(env.env.env.viewer, 'cam'):
        np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    #env.env.env.viewer.cam.distance += 0.3
    #env.env.env.viewer.cam.elevation += 15
    #self.viewer.cam.azimuth = 180.
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    #env.env.env.viewer.cam.distance -= 0.3
    #env.env.env.viewer.cam.elevation -= 15
    return rgb_array

def take_objects_image_training(env, img_size):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['rectangle'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['cube'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['cylinder'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['table0'], alpha_val=0.0)
    # just to activate in case viewer is not intialized
    if not hasattr(env.env.env.viewer, 'cam'):
        np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array

def take_image_objects(env, img_size):
    env.env.env._set_arm_visible(visible=False)
    env.env.env._set_visibility(names_list=['object0'], alpha_val=1.0)
    env.env.env._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    # just to activate in case viewer is not intialized
    if not hasattr(env.env.env.viewer, 'cam'):
        np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
    return rgb_array


def transform_image_to_latent_batch_torch(im_batch, vae, img_size, device):
    image = torch.from_numpy(im_batch).float().to(device)
    image /= 255
    image = image.permute([0, 3, 1, 2])
    with torch.no_grad():
        if isinstance(vae, VAE):
            mu, logvar = vae.encode(image.reshape(-1, img_size * img_size * 3))
            return mu, logvar
        elif isinstance(vae, VAE_SB):
            mu, logvar = vae.encode(image)
            return mu, logvar
        elif isinstance(vae, Monet_VAE):
            mu_s, logvar_s, masks = vae.encode(image)
            return mu_s, logvar_s, masks



def goal_latent_from_images(goals_images, args):
    assert goals_images.ndim == 4#always send as batch, even of 1
    # first transform them in latent_representation
    lg, lg_var = transform_image_to_latent_batch_torch(goals_images.copy(), args.vae_model_goal,
                                                       args.img_size, args.device)
    del lg_var
    lg = torch_goal_transformation(lg, args.device, ind_1=args.goal_ind_1, ind_2=args.goal_ind_2)
    lg = lg.detach().cpu().numpy()
    return lg

def obstacle_latent_from_images(obstacles_images, args):
    assert obstacles_images.ndim == 4  # always send as batch, even of 1
    lo, lo_var = transform_image_to_latent_batch_torch(obstacles_images.copy(), args.vae_model_obstacle,
                                                       args.img_size, args.device)
    del lo_var
    if args.use_mixed_with_size:
        obstacle_latent = torch_obstacle_transformation(lo, args.device, ind_1=args.obstacle_ind_1, ind_2=args.obstacle_ind_2)
        obstacle_latent = obstacle_latent.detach().cpu().numpy()
        if args.size_ind_2 is not None:
            obstacle_size_latent_1, obstacle_size_latent_2 = torch_get_size_in_space(lo, args.device, args.size_ind, ind_2=args.size_ind_2)
            obstacle_size_latent_1 = obstacle_size_latent_1.detach().cpu().numpy()
            obstacle_size_latent_2 = obstacle_size_latent_2.detach().cpu().numpy()
            obstacle_size_latent = np.concatenate([
                np.expand_dims(obstacle_size_latent_1, axis=1),
                np.expand_dims(obstacle_size_latent_2, axis=1),
            ], axis=1)
            return obstacle_latent, obstacle_size_latent
        else:
            obstacle_size_latent = torch_get_size_in_space(lo, args.device, args.size_ind)
            obstacle_size_latent = obstacle_size_latent.detach().cpu().numpy()
            return obstacle_latent, obstacle_size_latent
    else:
        lo = torch_obstacle_transformation(lo, args.device, ind_1=args.obstacle_ind_1, ind_2=args.obstacle_ind_2)
        lo = lo.detach().cpu().numpy()
        lo_s, lo_s_var = transform_image_to_latent_batch_torch(obstacles_images.copy(), args.vae_model_size,
                                                               args.img_size, args.device)
        del lo_s_var
        lo_s = torch_get_size_in_space(lo_s, args.device, args.size_ind)
        lo_s = lo_s.detach().cpu().numpy()

        return lo, lo_s

def latents_from_images(images, args):
    assert images.ndim == 4


    if args.vae_type == 'space':
        with torch.no_grad():
            args.vae_model.eval()
            images = torch.from_numpy(images).float().to(args.device)
            images /= 255.
            images = images.permute([0, 3, 1, 2])

            z_pres, z_depth, z_scale, z_shift, z_where, z_pres_logits, z_depth_post, \
            z_scale_post, z_shift_post, z_what, z_what_post = args.vae_model.encode(images)
            z_p = z_pres.detach().cpu().numpy()
            z_sc = z_scale.detach().cpu().numpy()
            z_sc = np.flip(z_sc, axis=2)
            z_sh = z_shift.detach().cpu().numpy()
            z_sh = np.flip(z_sh, axis=2)
            z_wh = z_what.detach().cpu().numpy()
            indices = z_p > 0.98
            # coordinates = z_sh[indices]
            # sizes = z_sc[indices]
        goal_pos = []
        goal_size = []
        obstacles_pos = []
        obstacles_size = []
        for i in range(len(images)):
            this_im_indices = np.squeeze(indices[i])
            im_coord = z_sh[i][this_im_indices]
            im_desc = z_wh[i][this_im_indices]
            im_scale = z_sc[i][this_im_indices]
            g_idx, o_idx = get_indices_goal_obstacle(im_desc)
            if len(g_idx) == 0:
                #the goal is not visible must be outside
                goal_pos.append(np.array([5., 5.]))
                goal_size.append(np.array([0., 0.]))
            else:
                goal_pos.append(im_coord[g_idx[0]])
                goal_size.append(im_scale[g_idx[0]])

            if len(o_idx) == 0:
                obstacles_pos.append([])
                obstacles_size.append([])
            else:
                obstacles_pos.append(im_coord[o_idx])
                obstacles_size.append(im_scale[o_idx])

        return np.array(goal_pos), np.array(goal_size), np.array(obstacles_pos), np.array(obstacles_size)
    else:
        # first transform them in latent_representation
        mu_s, logvar_s, masks = transform_image_to_latent_batch_torch(images.copy(), args.vae_model_goal,
                                                           args.img_size, args.device)
        del logvar_s
        del masks
        #with monet no transformation required + indices of postion and size should be same
        lg = mu_s[args.goal_slot][:, [args.goal_ind_1,args.goal_ind_2]]
        lo = mu_s[args.obstacle_slot][:, [args.goal_ind_1,args.goal_ind_2]]
        lo_s = mu_s[args.obstacle_slot][:, args.size_ind]

        return lg.detach().cpu().numpy(), lo.detach().cpu().numpy(), lo_s.detach().cpu().numpy()


#todo this must be inferred
def get_indices_goal_obstacle(z_char, goal_val=-1.19, goal_val2=3.8, obstacle_val=1.3, obstacle_val2=3.2):
    goal_indices = z_char[:, 8] <= goal_val
    obstacle_indices = np.logical_not(goal_indices)
    #goal_indices = z_char[:, 9] <= goal_val
    #goal_indices_2 = z_char[:, 9] >= goal_val2
    #goal_indices = np.logical_or(goal_indices, goal_indices_2)
    #obstacle_indices = z_char[:, 9] >= obstacle_val
    #obstacle_indices2 = z_char[:, 9] <= obstacle_val2
    #obstacle_indices = np.logical_and(obstacle_indices, obstacle_indices2)
    return np.squeeze(np.argwhere(goal_indices), axis=1), np.squeeze(np.argwhere(obstacle_indices), axis=1)
