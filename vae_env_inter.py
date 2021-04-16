import numpy as np
import torch
from vae.train_monet import Monet_VAE

def take_goal_image(env, img_size, make_table_invisible=True, direct_env=None):
    if direct_env is not None:
        env_to_use = direct_env
    else:
        env_to_use = env.env.env
    env_to_use._set_arm_visible(visible=False)
    env_to_use._set_visibility(names_list=['object0'], alpha_val=1.0)
    if not make_table_invisible:
        env_to_use._set_visibility(names_list=['table0'], alpha_val=1.0)
    rgb_array = np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    return rgb_array

def take_env_image(env, img_size, direct_env=None):
    if direct_env is not None:
        env_to_use = direct_env
    else:
        env_to_use = env.env.env
    env_to_use._set_arm_visible(visible=False)
    env_to_use._set_visibility(names_list=['object0'], alpha_val=1.0)
    env_to_use._set_visibility(names_list=['table0'], alpha_val=1.0)
    # todo think more effective way of doing this
    for name in ['obstacle', 'obstacle2', 'obstacle3']:
        try:
            env_to_use._set_visibility(names_list=[name], alpha_val=1.0)
        except:
            pass
    for id in  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16, 17, 18, 21]:
        env_to_use._set_visibility_with_id(id, alpha_val=0.7)
    #just to activate in case viewer is not intialized
    if not hasattr(env_to_use.viewer, 'cam'):
        np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    #env_to_use.viewer.cam.distance += 0.3
    #env_to_use.viewer.cam.elevation += 15
    #self.viewer.cam.azimuth = 180.
    rgb_array = np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    #env_to_use.viewer.cam.distance -= 0.3
    #env_to_use.viewer.cam.elevation -= 15
    return rgb_array

def take_objects_image_training(env, img_size, direct_env=None):
    if direct_env is not None:
        env_to_use = direct_env
    else:
        env_to_use = env.env.env
    env_to_use._set_arm_visible(visible=False)
    env_to_use._set_visibility(names_list=['rectangle'], alpha_val=1.0)
    env_to_use._set_visibility(names_list=['cube'], alpha_val=1.0)
    env_to_use._set_visibility(names_list=['cylinder'], alpha_val=1.0)
    env_to_use._set_visibility(names_list=['table0'], alpha_val=0.0)
    try:
        env_to_use._set_visibility(names_list=['rectangle1'], alpha_val=1.0)
    except:
        pass
    try:
        env_to_use._set_visibility(names_list=['rectangle2'], alpha_val=1.0)
    except:
        pass
    try:
        env_to_use._set_visibility(names_list=['rectangle3'], alpha_val=1.0)
    except:
        pass
    try:
        env_to_use._set_visibility(names_list=['cube1'], alpha_val=1.0)
    except:
        pass
    try:
        env_to_use._set_visibility(names_list=['cylinder1'], alpha_val=1.0)
    except:
        pass

    # just to activate in case viewer is not initialized
    if not hasattr(env_to_use.viewer, 'cam'):
        np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    rgb_array = np.array(env_to_use.render( mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    return rgb_array

def take_image_objects(env, img_size, direct_env=None):
    if direct_env is not None:
        env_to_use = direct_env
    else:
        env_to_use = env.env.env
    env_to_use._set_arm_visible(visible=False)
    env_to_use._set_visibility(names_list=['object0'], alpha_val=1.0)
    env_to_use._set_visibility(names_list=[], alpha_val=1.0)
    for name in ['obstacle', 'obstacle2', 'obstacle3']:
        try:
            env_to_use._set_visibility(names_list=[name], alpha_val=1.0)
        except:
            pass
    # just to activate in case viewer is not intialized
    if not hasattr(env_to_use.viewer, 'cam'):
        #using camera name does not call viewer setup
        np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    rgb_array = np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    return rgb_array

def transform_image_to_latent_batch_torch(im_batch, vae, img_size, device):
    image = torch.from_numpy(im_batch).float().to(device)
    image /= 255
    image = image.permute([0, 3, 1, 2])
    with torch.no_grad():
        if isinstance(vae, Monet_VAE):
            mu_s, logvar_s, masks = vae.encode(image)
            return mu_s, logvar_s, masks

def latents_from_images(images, args):
    assert images.ndim == 4
    if args.vae_type == 'bbox':
        with torch.no_grad():
            args.vae_model.eval()
            images = torch.from_numpy(images).float().to(args.device)
            images /= 255.
            images = images.permute([0, 3, 1, 2])

            goal_index = args.obj_index
            obstacle_idx = args.obstacles_indices
            #(B, K, D)
            z_pres, z_depth, z_scale, z_pos = args.vae_model.encode(images)
            z_pres, z_depth, z_scale, z_pos = z_pres.detach().cpu().numpy(), z_depth.detach().cpu().numpy(), \
                                              z_scale.detach().cpu().numpy(), z_pos.detach().cpu().numpy()

            
        goal_pos = z_pos[:, goal_index, :]
        goal_size = z_scale[:, goal_index, :]
        obstacles_pos = z_pos[:, obstacle_idx, :]
        obstacles_size = z_scale[:, obstacle_idx, :]
        indices_goal_present = z_pres[:, goal_index, 0] > 0.8
        indices_goal_not_present = np.logical_not(indices_goal_present)
        #set those goals far away
        goal_pos[indices_goal_not_present] = np.array([100., 100.])
        goal_size[indices_goal_not_present] = np.array([0., 0.])

        #print('obj pos is {}'.format(goal_pos))
        #print('obj size is {}'.format(goal_size))
        #print('obstacle pos are {}'.format(obstacles_pos))
        #print('obstacle sizes are {}'.format(obstacles_size))

        return goal_pos, goal_size, obstacles_pos, obstacles_size
        #return np.round(goal_pos, 3), np.round(goal_size, 3), np.round(obstacles_pos, 3), np.round(obstacles_size, 3)
        #return np.round(goal_pos, 2), np.round(goal_size, 2), np.round(obstacles_pos, 2), np.round(obstacles_size, 2)

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
