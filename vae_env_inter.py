import numpy as np
import copy
import torch
from vae.train_vae_sb import VAE_SB
from vae.train_vae import VAE
from vae.train_monet import Monet_VAE
from vae.latent_space_transformations import torch_goal_transformation, torch_obstacle_transformation, \
    torch_get_size_in_space
import torchvision
from PIL import Image


# todo set to false once is trained with table
def take_obstacle_image(env, img_size, make_table_invisible=True, direct_env=None):
    if direct_env is not None:
        env_to_use = direct_env
    else:
        env_to_use = env.env.env
    env_to_use._set_arm_visible(visible=False)
    env_to_use._set_visibility(names_list=['obstacle'], alpha_val=1.0)
    env_to_use._set_visibility(names_list=['object0'], alpha_val=0.0)
    if not make_table_invisible:
        env_to_use._set_visibility(names_list=['table0'], alpha_val=1.0)
    rgb_array = np.array(env_to_use.render(mode='rgb_array', width=img_size, height=img_size, camera_name='cam_top'))
    return rgb_array

def take_goal_image(env, img_size, make_table_invisible=True, make_walls_invisible=True, direct_env=None):
    if direct_env is not None:
        env_to_use = direct_env
    else:
        env_to_use = env.env.env
    env_to_use._set_arm_visible(visible=False)
    env_to_use._set_visibility(names_list=['object0'], alpha_val=1.0)
    if not make_table_invisible:
        env_to_use._set_visibility(names_list=['table0'], alpha_val=1.0)
    if not make_walls_invisible:
        if 'wall1' in env_to_use.sim.model.body_names:
            env_to_use._set_visibility(names_list=['wall1', 'wall2', 'wall3', 'wall4'], alpha_val=1.0)
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
    elif args.vae_type == 'bbox':
        with torch.no_grad():
            args.vae_model.eval()

            #a = np.random.randint(0, 10)
            #im_current = Image.fromarray(images[0].astype(np.uint8))
            #im_current.save('{}env_image_for_vae_{}.png'.format(args.logger.my_log_dir, a))

            images = torch.from_numpy(images).float().to(args.device)
            images /= 255.
            images = images.permute([0, 3, 1, 2])

            goal_index = args.obj_index
            obstacle_idx = args.obstacles_indices
            #print('obj index is {}'.format(goal_index))
            #print('obstacle indices are {}'.format(obstacle_idx))



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

        print('obj pos is {}'.format(goal_pos))
        print('obj size is {}'.format(goal_size))
        print('obstacle pos are {}'.format(obstacles_pos))
        print('obstacle sizes are {}'.format(obstacles_size))

        return goal_pos, goal_size, obstacles_pos, obstacles_size
        #return np.round(goal_pos, 3), np.round(goal_size, 3), np.round(obstacles_pos, 3), np.round(obstacles_size, 3)
        #return np.round(goal_pos, 2), np.round(goal_size, 2), np.round(obstacles_pos, 2), np.round(obstacles_size, 2)

    elif args.vae_type=='faster_rcnn':
        with torch.no_grad():
            args.vae_model.eval()
            images = torch.from_numpy(images).float().to(args.device)
            images /= 255.
            images = images.permute([0, 3, 1, 2])
            output = args.vae_model(images)
            boxes = []
            labels = []
            scores = []
            for i in range(len(output)):
                keep = torchvision.ops.nms(boxes=output[i]['boxes'], scores=output[i]['scores'], iou_threshold=0.3)
                b = output[i]['boxes'][keep].detach().cpu().numpy()
                #boxes to env format
                x_center = (b[:, 2] + b[:, 0]) / 2.
                x_dim = np.abs(b[:, 2] - b[:, 0]) / 2.
                y_center = (b[:, 3] + b[:, 1]) / 2.
                y_dim = np.abs(b[:, 3] - b[:, 1]) / 2.
                b[:, 0] = x_center
                b[:, 1] = y_center
                b[:, 2] = x_dim
                b[:, 3] = y_dim
                boxes.append(b)
                labels.append(output[i]['labels'][keep].detach().cpu().numpy())
                scores.append(output[i]['scores'][keep].detach().cpu().numpy())


        boxes_goal = []
        boxes_obstacles = []
        for i in range(len(boxes)):
            index_goal = labels[i] == 2
            index_obstacles = labels[i] == 1
            if np.all(index_goal == False):
                #the goal is not present
                boxes_goal.append(np.array([100., 100., 0., 0.]))
            else:
                max_ind = np.argmax(scores[i][index_goal])
                boxes_goal.append(boxes[i][index_goal][max_ind])
            ob_idx_over_tr = scores[i][index_obstacles] >= 0.8
            boxes_obstacles.append(boxes[i][index_obstacles][ob_idx_over_tr])
        boxes_goal = np.array(boxes_goal)
        boxes_obstacles = np.array(boxes_obstacles)
        goal_pos = boxes_goal[:, 0:2]
        goal_size = boxes_goal[:, 2:4]
        obstacles_pos = boxes_obstacles[:, :, 0:2]
        obstacles_size = boxes_obstacles[:, :, 2:4]

        return goal_pos, goal_size, obstacles_pos, obstacles_size

        
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


def get_indices_goal_obstacle(z_char, goal_val=0.8, goal_val2=3.8, obstacle_val=1.3, obstacle_val2=3.2):#goal_val=-1.19, goal_val2=3.8, obstacle_val=1.3, obstacle_val2=3.2):
    #mean = 0.5 * z_char[:, 2] + 0.5 * z_char[:, 3]
    #goal_indices = mean <= -0.65#-0.9
    goal_indices = z_char[:, 2] <= -0.5
    obstacle_indices = np.logical_not(goal_indices)
    '''#goal_indices = z_char[:, 2] <= goal_val
    goal_indices = z_char[:, 2] >= goal_val
    obstacle_indices = np.logical_not(goal_indices)
    #goal_indices = z_char[:, 9] <= goal_val
    #goal_indices_2 = z_char[:, 9] >= goal_val2
    #goal_indices = np.logical_or(goal_indices, goal_indices_2)
    #obstacle_indices = z_char[:, 9] >= obstacle_val
    #obstacle_indices2 = z_char[:, 9] <= obstacle_val2
    #obstacle_indices = np.logical_and(obstacle_indices, obstacle_indices2)'''
    return np.squeeze(np.argwhere(goal_indices), axis=1), np.squeeze(np.argwhere(obstacle_indices), axis=1)
