from PIL import Image
import numpy as np

from j_vae.generate_vae_data import random_pos_inside, size_file,random_size_at,generate_points
from j_vae.common_data import  min_obstacle_size, max_obstacle_size, range_x, range_y, obstacle_size, \
    puck_size, z_table_height, center_obstacle, train_file_name, vae_sb_weights_file_name, file_corners_name
import torch
import matplotlib.pyplot as plt
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from envs import make_env
from vae_env_inter import take_goal_image, take_obstacle_image, take_objects_image_training
from j_vae.train_vae import load_Vae
from j_vae.train_monet import load_Vae as load_Monet
import copy

from j_vae.latent_space_transformations import create_rotation_matrix, rotate_list_of_points, angle_obstacle, angle_goal,\
    get_size_in_space
from vae_env_inter import get_indices_goal_obstacle
from SPACE.main_space import load_space_model


def visualization_grid_points(env, model, size_to_use, img_size, n, enc_type, ind_1, ind_2,
                              using_sb=True, use_d=False, fig_file_name=None):
    if use_d:
        d = 0.12#0.32
        points = generate_points(range_x=[range_x[0] - d, range_x[1] + d], range_y=[range_y[0] - d, range_y[1] + d],
                                 z=z_table_height, total=n,
                                 object_x_y_size=[size_to_use, size_to_use])
    else:
        points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=n,
                                 object_x_y_size=[size_to_use, size_to_use])


    n_labels = np.arange(len(points))

    points = np.array(points)
    #print_max_and_min(points)

    xs = points[:, 0]
    ys = points[:, 1]
    plt.figure(1)
    plt.subplot(211, )
    plt.scatter(xs,ys)
    plt.title('real')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (xs[i], ys[i]))


    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    #move other objects to plaecs they do not disturb
    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
        pass
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._move_object(position=[2.,2.,0.4])
    else:
        raise Exception('Not supported enc type')
    for i,p in enumerate(points):
        if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
            env.env.env._move_object(position=p)
            data_set[i] = take_goal_image(env, img_size, make_table_invisible=True)
        elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
            env.env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env, img_size)
        else:
            raise Exception('Not supported enc type')
    all_array = None
    t = 0
    for r in range(n):
        row = None
        for c in range(n):
            rcim = data_set[t].copy()
            t += 1
            if row is None:
                row = rcim
            else:
                row = np.concatenate([row.copy(), rcim], axis=1)
        if all_array is None:
            all_array = row.copy()
        else:
            all_array = np.concatenate([all_array.copy(), row], axis=0)
    all_ims = Image.fromarray(all_array.astype(np.uint8))
    if fig_file_name is not None:
        all_ims.save('{}_ims.png'.format(fig_file_name))
        from hindsight_goals_visualizer import show_points
        show_points(points, '{}_vis'.format(fig_file_name),'real')
    else:
        all_ims.show()
    all_ims.close()
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    model.eval()
    if not using_sb:
        mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
    else:
        mu, logvar = model.encode(data)
    mu = mu.detach().cpu().numpy()

    assert ind_1 != ind_2
    mu = np.concatenate([np.expand_dims(mu[:, ind_1], axis=1),
                         np.expand_dims(mu[:, ind_2], axis=1)], axis=1)


    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        rm = create_rotation_matrix(angle_goal)
        mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, goal_map_x, goal_map_y)
        pass
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        #for i, p in enumerate(mu):
        #    mu[i] = reflect_obstacle_transformation(p)
        rm = create_rotation_matrix(angle_obstacle)
        mu = rotate_list_of_points(mu, rm)
        #mu = map_points(mu, obstacle_map_x, obstacle_map_y)
        pass
    else:
        raise Exception('Not supported enc type')
    print_max_and_min(mu)

    lxs = mu[:, 0]
    lys = mu[:, 1]
    plt.subplot(212)
    plt.scatter(lxs, lys)
    plt.title('latent')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (lxs[i], lys[i]))

    if fig_file_name is not None:
        plt.savefig(fig_file_name)
    else:
        plt.show()
    plt.close()

def visualization_grid_points_all(env, model, size_to_use, img_size, n, enc_type, ind_1, ind_2, fig_file_name):
    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=n, object_x_y_size=[size_to_use, size_to_use])

    #points = points[:n]#just take those thar vary in y
    #points = [points[i*n+3] for i in range(n)]  # just take those that vary in x
    n_labels = np.arange(len(points))

    points = np.array(points)
    points_2 = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=n, object_x_y_size=[size_to_use, size_to_use])
    points_2 = np.array(points_2)
    np.random.shuffle(points_2)
    #print_max_and_min(points)

    xs = points[:, 0]
    ys = points[:, 1]
    plt.figure(1)
    plt.scatter(xs, ys)
    plt.title('real')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (xs[i], ys[i]))
    plt.savefig("{}_real".format(fig_file_name))
    plt.close()


    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")


    if True:#todo delete this condition
        # move other objects to plaecs they do not disturb
        env.env.env._set_position(names_list=['obstacle'], position=[10., 10., 10.])
        env.env.env._move_object(position=[-10., -10., -10.])
        # position object
        p1 = np.array([-20., 20., 20.])
        p2 = np.array([-20., -20., 20.])
        d1 = 0.12
        d2 = 0.03
        env.env.env._set_size(names_list=['rectangle'], size=[d1, d2, 0.035])
        env.env.env._change_color(['rectangle'], 0., 0., 1.)
        pos = [1.3, 1., 0.4 + 0.035]
        env.env.env._set_position(names_list=['rectangle'], position=pos)

        r = 0.04
        env.env.env._set_size(names_list=['cylinder'], size=[r, 0.025, 0.])
        env.env.env._change_color(['cylinder'], 1., 0., 0.)
        pos = [1.15, 0.65, 0.4 + 0.025]
        env.env.env._set_position(names_list=['cylinder'], position=pos)
        # env.env.env._set_position(names_list=['cylinder'], position=p1)

        s = 0.03
        env.env.env._set_size(names_list=['cube'], size=[s, s, s])
        env.env.env._change_color(['cube'], 0.5, 0., 0.5)
        pos = [1.3, 0.6, 0.4 + s]
        env.env.env._set_position(names_list=['cube'], position=pos)
        # env.env.env._set_position(names_list=['cube'], position=p2)

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    for i,p in enumerate(points):
        env.env.env._set_position(names_list=['cylinder'], position=[p[0], p[1], 0.4 + 0.025])
        #env.env.env._set_position(names_list=['rectangle'], position=[points_2[i][0], points_2[i][1] , 0.4 + 0.035])
        #env.env.env._set_position(names_list=['rectangle'], position=[p[0], p[1], 0.4 + 0.035])
        data_set[i]  = take_objects_image_training(env, img_size)
    '''all_array = None
    t = 0
    for r in range(n):
        row = None
        for c in range(n):
            rcim = data_set[t].copy()
            t += 1
            if row is None:
                row = rcim
            else:
                row = np.concatenate([row.copy(), rcim], axis=1)
        if all_array is None:
            all_array = row.copy()
        else:
            all_array = np.concatenate([all_array.copy(), row], axis=0)
    all_ims = Image.fromarray(all_array.astype(np.uint8))
    if fig_file_name is not None:
        all_ims.save('{}_ims.png'.format(fig_file_name))
        from hindsight_goals_visualizer import show_points
        show_points(points, '{}_vis'.format(fig_file_name),'real')
    else:
        all_ims.show()
    all_ims.close()'''
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    with torch.no_grad():
        model.eval()
        mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s = model(data)
        from j_vae.train_monet import visualize_masks, numpify
        visualize_masks(imgs=numpify(data), masks=numpify(torch.cat(masks, dim=1)), recons=numpify(full_reconstruction),
                        file_name=fig_file_name + "_recons.png")
        mu_s, logvar_s, masks = model.encode(data)
    #for slot_ind, mu in enumerate(mu_s):
    slot_ind = 1
    mu = mu_s[slot_ind]
    for _ in range(1):
        mu = mu.detach().cpu().numpy()
        for latent_ind_1 in range(mu.shape[1]-1):
            for latent_ind_2 in range(latent_ind_1+1, mu.shape[1]):
                vals1 = mu[:, latent_ind_1]
                vals2 = mu[:, latent_ind_2]
                plt.scatter(vals1, vals2)
                plt.title('latent')
                for i, en in enumerate(n_labels):
                    plt.annotate(en, (vals1[i], vals2[i]))

                plt.savefig("{}_slot_{}_latent1_{}_latent2_{}".format(fig_file_name, slot_ind,
                                                                      latent_ind_1, latent_ind_2))
                plt.close()

def visualization_grid_points_space(env, model, size_to_use, img_size, n, enc_type, ind_1, ind_2, fig_file_name):
    points = generate_points(range_x=[range_x[0], range_x[1]], range_y=[range_y[0],range_y[1]], z=z_table_height, total=n, object_x_y_size=[size_to_use, size_to_use])
    n_labels = np.arange(len(points))

    points = np.array(points)
    points_2 = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=n, object_x_y_size=[size_to_use, size_to_use])
    points_2 = np.array(points_2)
    np.random.shuffle(points_2)
    #print_max_and_min(points)

    xs = points[:, 0]
    ys = points[:, 1]
    plt.figure(1)
    plt.scatter(xs, ys)
    plt.title('real')
    for i, en in enumerate(n_labels):
        plt.annotate(en, (xs[i], ys[i]))
    plt.savefig("{}_real".format(fig_file_name))
    plt.close()

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")
    if True:#todo delete this condition
        # move other objects to plaecs they do not disturb
        env.env.env._set_position(names_list=['obstacle'], position=[10., 10., 10.])
        env.env.env._move_object(position=[-10., -10., -10.])
        # position object
        p1 = np.array([-20., 20., 20.])
        p2 = np.array([-20., -20., 20.])
        d1 = 0.12
        d2 = 0.03
        #env.env.env._set_size(names_list=['rectangle'], size=[d1, d2, 0.035])
        env.env.env._set_size(names_list=['rectangle'], size=[0.09, 0.03, 0.035])
        env.env.env._change_color(['rectangle'], 0., 0., 1.)
        #pos = [1.3, 1., 0.4 + 0.035]
        pos = [1.3, 0.75, 0.4 + 0.035]
        env.env.env._set_position(names_list=['rectangle'], position=pos)

        r = 0.04
        env.env.env._set_size(names_list=['cylinder'], size=[r, 0.025, 0.])
        env.env.env._change_color(['cylinder'], 1., 0., 0.)
        pos = [1.15, 0.65, 0.4 + 0.025]
        env.env.env._set_position(names_list=['cylinder'], position=pos)
        #env.env.env._change_color(['cylinder'], 0., 0., 1.)
        # env.env.env._set_position(names_list=['cylinder'], position=p1)

        s = 0.03
        env.env.env._set_size(names_list=['cube'], size=[s, s, s])
        env.env.env._change_color(['cube'], 0.5, 0., 0.5)
        pos = [1.3, 0.6, 0.4 + s]
        env.env.env._set_position(names_list=['cube'], position=p2)
        # env.env.env._set_position(names_list=['cube'], position=p2)

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    for i,p in enumerate(points):
        env.env.env._set_position(names_list=['cylinder'], position=[p[0], p[1], 0.4 + 0.025])
        #env.env.env._set_position(names_list=['rectangle'], position=[points_2[i][0], points_2[i][1] , 0.4 + 0.035])
        #env.env.env._set_position(names_list=['rectangle'], position=[p[0], p[1], 0.4 + 0.035])
        data_set[i]  = take_objects_image_training(env, img_size)
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    char = []
    with torch.no_grad():
        model.eval()

        z_pres, z_depth, z_scale, z_shift, z_where, z_pres_logits, z_depth_post, z_scale_post, z_shift_post, z_what, z_what_post = model.encode(data)
        z_p = z_pres.detach().cpu().numpy()
        z_sc = z_scale.detach().cpu().numpy()
        z_sc = np.flip(z_sc, axis=2)
        z_sh = z_shift.detach().cpu().numpy()
        z_sh = np.flip(z_sh, axis=2)
        z_wh = z_what.detach().cpu().numpy()
        indices = z_p > 0.98
        #coordinates = z_sh[indices]
        #sizes = z_sc[indices]
    for i in range(len(points)):

        this_im_indices = np.squeeze(indices[i])
        c_im = z_sh[i][this_im_indices]
        c_w = z_wh[i][this_im_indices]
        c_sc = z_sc[i][this_im_indices]
        print("i: {}".format(i))
        if len(c_im) > 1:
            print('len bigger than 1')
        print("char: {}".format(c_w[:, 8]))
        for c in c_w[:, 8]:
            char.append(c)



        g_idx, o_idx = get_indices_goal_obstacle(c_w)
        if i in [0,3,6,9, 14, 17, 20, 26, 31,35, 38, 40, 42,45,48]:
            a = 1
        if len(g_idx) == 0:
            print('oh')
        else:
            print(c_im[o_idx])
            c_im = c_im[g_idx[0]]

            #s_im = z_sc[i][this_im_indices]
            plt.scatter(c_im[0], c_im[1], c='blue')
            plt.annotate(n_labels[i], (c_im[0], c_im[1]))
    char = np.array(char)
    print("the min is {} \n the max is{} \n the mean is {}".format(
        np.min(char),np.max(char), np.mean(char)))
    plt.title('latent')
    plt.savefig("{}.png".format(fig_file_name))
    plt.close()


def traversal(env, model, img_size,  latent_size, n, enc_type, using_sb=True, fig_file_name=None):
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")
    dist = 0.8

    data_set = np.empty([n * latent_size, img_size, img_size, 3])
    # move other objects to plaecs they do not disturb
    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._move_object(position=[2., 2., 0.4])
    elif enc_type == 'all':
        env.env.env._set_position(names_list=['obstacle'], position=[10., 10., 10.])
        env.env.env._move_object(position=[-10., -10., -10.])
        obj = 'cube'
        p1 = np.array([-20., 20., 20.])
        p2 = np.array([-20., -20., 20.])
        env.env.env._set_position(names_list=['rectangle'], position=p1)
        env.env.env._set_position(names_list=['cylinder'], position=p2)
        s = 0.06
        env.env.env._set_size(names_list=['cube'], size=[s, s, s])
        pos = [1.3, 0.75, 0.4+s]
        env.env.env._set_position(names_list=[obj], position=pos)
    else:
        raise Exception('Not supported enc type')

    # sample image  central
    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._move_object(position=[1.3, 0.75, 0.4])
        central_im = take_goal_image(env, img_size, make_table_invisible=True)
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._set_position(names_list=['obstacle'], position=[1.3, 0.75, 0.4])
        # env.env.env._set_size(names_list=['obstacle'], size=np.array([0.15, 0.035, 0.]))
        central_im = take_obstacle_image(env, img_size)
    elif enc_type == 'all':
        central_im = take_objects_image_training(env, img_size)
    else:
        raise Exception('Not supported enc type')

    #trasform to latent
    data = np.expand_dims(central_im.copy(), axis=0)
    data = torch.from_numpy(data).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    model.eval()
    if not using_sb:
        mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
    else:
        mu, logvar = model.encode(data)

    mid = int(n / 2)
    for l in range(latent_size):
        for t in range(n):
            if t == mid:
                data_set[n*l+t] = central_im.copy()
            else:
                v=torch.zeros(latent_size)
                v[l] = 1.
                if t < mid:
                    v = v*-(mid-t)*dist
                else:
                    v = v*(t-mid)*dist
                v = v.to(device)
                z = mu+v

                im = model.decode(z)
                im = im.view(3, img_size, img_size)
                im = im.permute([1, 2, 0])
                im *= 255.
                im = im.type(torch.uint8)
                im = im.detach().cpu().numpy()
                data_set[n * l + t] = im.copy()

    all_array = None
    t = 0
    for r in range(latent_size):
        row = None
        for c in range(n):
            rcim = data_set[t].copy()
            t += 1
            if row is None:
                row = rcim
            else:
                row = np.concatenate([row.copy(), rcim], axis=1)
        if all_array is None:
            all_array = row.copy()
        else:
            all_array = np.concatenate([all_array.copy(), row], axis=0)
    all_ims = Image.fromarray(all_array.astype(np.uint8))
    if fig_file_name is not None:
        all_ims.save('{}_ims.png'.format(fig_file_name))
    else:
        all_ims.show()
    all_ims.close()


def traversal_all(env, model, img_size,  latent_size, n, fig_file_name):
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")
    dist = 1.

    # move other objects to plaecs they do not disturb
    env.env.env._set_position(names_list=['obstacle'], position=[10., 10., 10.])
    env.env.env._move_object(position=[-10., -10., -10.])
    # position object
    p1 = np.array([-20., 20., 20.])
    p2 = np.array([-20., -20., 20.])
    d1 = 0.12
    d2 = 0.03
    env.env.env._set_size(names_list=['rectangle'], size=[d1, d2, 0.035])
    env.env.env._change_color(['rectangle'], 0., 0., 1.)
    pos = [1.3, 1., 0.4 + 0.035]
    env.env.env._set_position(names_list=['rectangle'], position=pos)

    r = 0.04
    env.env.env._set_size(names_list=['cylinder'], size=[r, 0.025, 0.])
    env.env.env._change_color(['cylinder'], 1., 0., 0.)
    pos = [1.15, 0.65, 0.4 + 0.025]
    env.env.env._set_position(names_list=['cylinder'], position=pos)
    #env.env.env._set_position(names_list=['cylinder'], position=p1)

    s = 0.03
    env.env.env._set_size(names_list=['cube'], size=[s, s, s])
    env.env.env._change_color(['cube'], 0.5, 0., 0.5)
    pos = [1.3, 0.6, 0.4+s]
    env.env.env._set_position(names_list=['cube'], position=pos)
    #env.env.env._set_position(names_list=['cube'], position=p2)

    # sample image  central
    central_im = take_objects_image_training(env, img_size)

    #trasform to latent
    data = np.expand_dims(central_im.copy(), axis=0)
    data = torch.from_numpy(data).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s = model(data)
    #model.eval() Todo!!!!!!! WITH model eval it does not work
    from j_vae.train_monet import visualize_masks, numpify
    visualize_masks(imgs=numpify(data), masks=numpify(torch.cat(masks, dim=1)), recons=numpify(full_reconstruction),
                    file_name=fig_file_name+"_recons.png")
    mu_s, logvar_s, masks = model.encode(data)

    for slot_idx in range(len(mu_s)):
        mu = mu_s[slot_idx]
        data_set = np.empty([n * latent_size, img_size, img_size, 3])
        mid = int(n / 2)
        for l in range(latent_size):
            for t in range(n):
                if t == mid:
                    data_set[n*l+t] = central_im.copy()
                else:
                    v=torch.zeros(latent_size)
                    v[l] = 1.
                    if t < mid:
                        v = v*-(mid-t)*dist
                    else:
                        v = v*(t-mid)*dist
                    v = v.to(device)
                    z = mu+v

                    #z_s = copy.copy(mu_s)
                    #z_s[slot_idx] = z
                    #im, _, _ = model.decode(z_s, masks)
                    im, rec_mask = model._decoder_step(z)
                    rec_mask_mod = torch.sigmoid(rec_mask)
                    rec_mask_mod = torch.unsqueeze(rec_mask_mod, dim=0)
                    z_s = copy.copy(mu_s)
                    z_s[slot_idx] = z
                    new_masks = copy.copy(masks)
                    new_masks[slot_idx] = rec_mask_mod
                    im, _, _ = model.decode(z_s, new_masks)

                    im = im.view(3, img_size, img_size)
                    im = im.permute([1, 2, 0])
                    im *= 255.
                    im = im.type(torch.uint8)
                    im = im.detach().cpu().numpy()
                    data_set[n * l + t] = im.copy()

        all_array = None
        t = 0
        spacer_r = np.zeros(shape=(img_size, 5, 3))
        spacer_c = np.zeros(shape=(5, img_size*n+5*(n-1), 3))
        for r in range(latent_size):
            row = None
            for c in range(n):
                rcim = data_set[t].copy()
                t += 1
                if row is None:
                    row = rcim
                else:
                    row = np.concatenate([row.copy(), spacer_r, rcim], axis=1)
            if all_array is None:
                all_array = row.copy()
            else:
                all_array = np.concatenate([all_array.copy(), spacer_c, row], axis=0)
        all_ims = Image.fromarray(all_array.astype(np.uint8))
        all_ims.save('{}_ims_slot_{}.png'.format(fig_file_name, slot_idx))
        all_ims.close()


def print_max_and_min(points):
    assert isinstance(points, np.ndarray)
    xs = points[:, 0]
    ys = points[:, 1]


    print('max x: {}'.format(xs.max()))
    print('min x: {}'.format(xs.min()))
    print('max y: {}'.format(ys.max()))
    print('min y: {}'.format(ys.min()))


def save_corners(env, size_to_use, file_corners, img_size, enc_type):
    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=2,
                             object_x_y_size=[size_to_use, size_to_use])

    # sample images
    data_set = np.empty([len(points), img_size, img_size, 3])
    # move other objects to plaecs they do not disturb
    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._move_object(position=[2., 2., 0.4])
    else:
        raise Exception('Not supported enc_type')
    for i, p in enumerate(points):
        if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
            env.env.env._move_object(position=p)
            data_set[i] = take_goal_image(env, img_size)
        elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
            env.env.env._set_position(names_list=['obstacle'], position=p)
            data_set[i] = take_obstacle_image(env, img_size)
        else:
            raise Exception('Not supported enc_type')
    np.save(file_corners, data_set)
    all_array = None
    t = 0
    for r in range(len(points)):
        rcim = data_set[t].copy()
        t += 1
        if all_array is None:
            all_array = rcim
        else:
            all_array = np.concatenate([all_array.copy(), rcim], axis=1)
    all_ims = Image.fromarray(all_array.astype(np.uint8))
    all_ims.show()
    all_ims.close()


def save_center(env, size_to_use, file_corners, img_size, enc_type):
    points = generate_points(range_x=range_x, range_y=range_y, z=z_table_height, total=3,
                             object_x_y_size=[size_to_use, size_to_use])

    # sample images
    data_set = np.empty([1, img_size, img_size, 3])
    # move other objects to plaecs they do not disturb
    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._move_object(position=[2., 2., 0.4])
    else:
        raise Exception('Not supported enc_type')

    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._move_object(position=points[4])
        data_set[0] = take_goal_image(env, img_size, make_table_invisible=False)
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._set_position(names_list=['obstacle'], position=points[4])
        data_set[0] = take_obstacle_image(env, img_size)
    else:
        raise Exception('Not supported enc_type')
    np.save(file_corners, data_set)


def visualization_sizes_obstacle(env, env_name, model, enc_type, img_size, n,ind_1, ind_2=None, using_sb=True):
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    if enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
        env.env.env._set_position(names_list=['obstacle'], position=[2., 2., 0.4])
        pass
    elif enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
        env.env.env._move_object(position=[2.,2.,0.4])
    else:
        raise Exception('Not supported enc type')

    if env_name =='FetchPushMovingObstacleEnv-v1':
        assert ind_2 is not None
        min_s = min_obstacle_size['FetchPushMovingObstacleEnv-v1']
        max_s = max_obstacle_size['FetchPushMovingObstacleEnv-v1']
        sizes_x = np.linspace(min_s, max_s, num=n)
        sizes_y = np.linspace(min_s, max_s, num=n)
        plt.figure(1)
        plt.subplot(211)
        data_set = np.empty([len(sizes_x) * len(sizes_x), img_size, img_size, 3])
        for i, s_x in enumerate(sizes_x):
            for j, s_y in enumerate(sizes_y):
                plt.scatter(s_x, s_y)
                plt.annotate(i * len(sizes_y) + j, (s_x, s_y))
                #take image
                env.env.env._set_size(names_list=['obstacle'], size=np.array([s_x, s_y, 0.035]))
                data_set[i*len(sizes_y)+j] = take_obstacle_image(env, img_size)

        data = torch.from_numpy(data_set).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        if not using_sb:
            mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
        else:
            mu, logvar = model.encode(data)
        mu = mu.detach().cpu().numpy()


        assert ind_1 != ind_2
        mu = np.concatenate([np.expand_dims(mu[:, ind_1], axis=1),
                             np.expand_dims(mu[:, ind_2], axis=1)], axis=1)
        mtx, mty = get_size_in_space(mu[:, 0], v2=mu[:, 1])
        mu = np.concatenate([np.expand_dims(mtx, axis=1),
                             np.expand_dims(mty, axis=1)], axis=1)
        plt.subplot(212)
        plt.title('latent')

        for i, m in enumerate(mu):
            plt.scatter(m[0], m[1])
            plt.annotate(i, (m[0], m[1]))


    else:
        sizes = np.linspace(min_obstacle_size, max_obstacle_size, num=n)
        #sizes = np.linspace(obstacle_size, obstacle_size, num=n)
        n_labels = np.arange(len(sizes))
        sizes = np.array(sizes)

        xs = np.repeat(1, len(sizes))
        ys = sizes
        plt.figure(1)
        plt.subplot(211)
        plt.scatter(xs, ys)
        plt.title('real')
        for i, en in enumerate(n_labels):
            plt.annotate(en, (xs[i], ys[i]))

        # sample images
        data_set = np.empty([len(sizes), img_size, img_size, 3])
        for i, p in enumerate(sizes):
            env.env.env._set_position(names_list=['obstacle'], position=center_obstacle)
            env.env.env._set_size(names_list=['obstacle'], size=np.array([sizes[i], 0.035, 0.0]))
            data_set[i] = take_obstacle_image(env, img_size)
        data = torch.from_numpy(data_set).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        if not using_sb:
            mu, logvar = model.encode(data.reshape(-1, img_size * img_size * 3))
        else:
            mu, logvar = model.encode(data)
        mu = mu.detach().cpu().numpy()


        mu = mu[:, ind_1]

        for i, p in enumerate(mu):
            mu[i] = get_size_in_space(mu[i])
        #    mu[i] = map_size_space(mu[i])
        #    mu[i] = get_size_in_space(map_size_space(mu[i]))
            pass

        lxs = np.repeat(1, len(sizes))
        lys = mu

        plt.subplot(212)
        plt.scatter(lxs, lys)
        plt.title('latent')
        for i, en in enumerate(n_labels):
            plt.annotate(en, (lxs[i], lys[i]))
        print(mu)


    plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1', required=True)
    parser.add_argument('--task', help='the type of attribute that we want to generate/encode', type=str,
                        default='show_space', choices=['show_space', 'save_corners', 'save_center',
                                                       'show_size','show_traversal'], required=True)
    args, _ = parser.parse_known_args()
    if args.env == 'HandReach-v0':
        parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                            choices=['vanilla', 'reach'])
    else:
        parser.add_argument('--goal', help='method of goal generation', type=str, default='interval',
                            choices=['vanilla', 'fixobj', 'interval', 'custom'])
        if args.env[:5] == 'Fetch':
            parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32,
                                default=1.0)
        elif args.env[:4] == 'Hand':
            parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32,
                                default=0.25)

    parser.add_argument('--ind_1', help='first index to extract from latent vector', type=np.int32)
    parser.add_argument('--ind_2', help='second index to extract from latent vector', type=np.int32)

    parser.add_argument('--enc_type', help='the type of attribute that we want to generate/encode', type=str,
                        default='goal', choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes', 'mixed', 'all', 'space'])
    parser.add_argument('--mix_h', help='if the representation should de done with goals or obstacles', type=str,
                        default='goal', choices=['goal', 'obstacle'])

    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)
    parser.add_argument('--latent_size', help='latent size to train the VAE', type=np.int32, default=5)


    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    base_data_dir = this_file_dir + '../data/'
    data_dir = base_data_dir + args.env + '/'
    train_file = data_dir + train_file_name[args.enc_type]
    weights_path = data_dir + vae_sb_weights_file_name[args.enc_type]

    # load environment
    env = make_env(args)

    #other arguments for the algorithms
    if args.enc_type == 'goal':
        #size_to_use = puck_size
        size_to_use = 0
    elif args.enc_type == 'obstacle':
        #size_to_use = obstacle_size
        size_to_use = 0
    elif args.enc_type == 'mixed':
        size_to_use = (obstacle_size + puck_size) /2
    elif args.enc_type == 'all':
        size_to_use = 0.06
    elif args.enc_type == 'space':
        size_to_use = 0.0

    if args.task == 'show_space' or args.task == 'show_size' or args.task == 'show_traversal':
        # load the latent_size and model
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if args.enc_type == 'goal' or args.enc_type == 'obstacle' or args.enc_type == 'mixed':
            model = load_Vae_SB(weights_path, args.img_size, args.latent_size)
        elif args.enc_type == 'all':
            model = load_Monet(weights_path, args.img_size, args.latent_size, num_slots=4)
        elif args.enc_type == 'space':
            model = load_space_model(checkpoint_path='../data/FetchGenerativeEnv-v1/',
                                     check_name='../data/FetchGenerativeEnv-v1/model_000030001.pth', device='cuda:0')
            model.eval()
        else:
            model = load_Vae(weights_path, args.imgsize, args.latent_size)

        if args.task == 'show_space':

            if args.enc_type == 'goal' or (args.enc_type == 'mixed' and args.mix_h == 'goal'):
                fig_name = 'vis_grid_g'
            elif args.enc_type == 'obstacle' or (args.enc_type == 'mixed' and args.mix_h == 'obstacle'):
                fig_name = 'vis_grid_o'

            if args.enc_type == 'all':
                visualization_grid_points_all(n=7, env=env, model=model,size_to_use=size_to_use, img_size=args.img_size,
                                              enc_type=args.enc_type, ind_1=args.ind_1, ind_2=args.ind_2,
                                              fig_file_name='all_fig_cylinder')
            elif args.enc_type == 'space':
                visualization_grid_points_space(n=9, env=env, model=model,size_to_use=size_to_use, img_size=args.img_size,
                                              enc_type=args.enc_type, ind_1=args.ind_1, ind_2=args.ind_2,
                                              fig_file_name='space_fig_cylinder')
            else:
                visualization_grid_points(n=7, env=env, model=model,size_to_use=size_to_use, img_size=args.img_size,
                                          enc_type=args.enc_type, ind_1=args.ind_1, ind_2=args.ind_2,
                                          fig_file_name=fig_name)
        elif args.task == 'show_size':
            if hasattr(args, 'ind_2'):
                ind_2 = args.ind_2
            else:
                ind_2 = None
            visualization_sizes_obstacle(env=env, env_name=args.env, model=model, enc_type=args.enc_type,
                                         img_size=args.img_size,n=5, ind_1=args.ind_1, ind_2=ind_2)
        elif args.task == 'show_traversal':
            if args.enc_type == 'all':
                traversal_all(env=env, model=model, img_size=args.img_size, latent_size=args.latent_size, n=7,
                              fig_file_name='traversal')
            else:
                traversal(env, model, img_size=args.img_size, latent_size=args.latent_size, n=7,
                          enc_type=args.enc_type, fig_file_name='traversal')


    else:
        if args.task == 'save_corners':
            file_corners = data_dir + file_corners_name[args.enc_type]
            save_corners(env, size_to_use, file_corners, args.img_size, args.enc_type)
        '''elif args.task == 'save_center':
            assert args.enc_type == 'goal' or args.enc_type == 'obstacle'
            file_center = data_dir + file_center_name[args.enc_type]
            save_center(env, size_to_use, file_center, args.img_size, args.enc_type)'''



