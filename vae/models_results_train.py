import numpy as np
from utils.image_util import store_image_array_at, stack_images_column, stack_images_row, make_video
from vae.conv_vae_keras import CVAE_Keras
from envs import make_env
import pickle
from custom_start import get_args_and_initialize
import torch
from vae.vae_torch import setup_vae_just_images
from vae.latent_visualization import get_pca_reduction, visualize_reduced_state
import pandas as pd
import csv
import os

def make_batches(data, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        data: np array to data to be splitted in batches
        batch_size: Integer, batch size.

    # Returns
        A array with batches.
    """
    size = len(data)
    num_batches = (size + batch_size - 1) // batch_size  # round up
    permuted_indices = np.random.permutation(size)#indexes used since wholedata is to
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min(size, (i + 1) * batch_size)
        indices_to_use = permuted_indices[batch_start:batch_end]
        yield np.take(data, indices_to_use, axis=0)

def play_with_trained(args):
    # LOADING THE DATA
    with open(args.datapath, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['next_obs'].shape[0]
    # shaping as image 64x64x3
    dataset['obs'] = dataset['obs'].astype(np.float32)
    dataset['next_obs'] = dataset['obs'].astype(np.float32)
    # Normalizing the images to the range of [0., 1.]
    dataset['obs'] /= 255.
    dataset['next_obs'] /= 255.

    restore_path = args.dirpath + "weights_dir_complete/cvae_weights_0"
    #conv_vae = CVAE(lr=0.001, input_channels=3, representation_size=16, imsize=64, restore_path=restore_path)
    conv_vae = CVAE_Keras(lr=0.001, input_channels=3, representation_size=16, imsize=64, restore_path=restore_path)
    img_orig = dataset['next_obs'][3000]
    z, mu, log_sigma = conv_vae.encode(img_orig)
    print('this is z:', z)
    img_restored = conv_vae.decode(z)
    random_z = np.random.uniform(low=-1.5, high=2, size=16)
    img_random = conv_vae.decode(random_z)
    store_image_array_at(img_orig, args.dirpath, 'image_original', force_remap_to_255=True)
    store_image_array_at(img_restored, args.dirpath, 'image_restored', force_remap_to_255=True)
    store_image_array_at(img_random, args.dirpath, 'image_random')

def create_vae_result_images(args, recover_filename):
    #we do not use trainer any more, neihter the test loader
    vae_model, _, train_loader, _ = setup_vae_just_images(args, recover_filename=recover_filename)
    vae_model.eval()
    #loads model samples from latent space to create a plot from this one
    samples_size = 45000
    latent_samples = None
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            data = data.to(args.device)
            mu, logvar = vae_model.encode(data.view(-1, vae_model.flattened_dim))
            z = vae_model.reparameterize(mu, logvar)
            if latent_samples is None:
                latent_samples = z.cpu()
            else:
                latent_samples = torch.cat([latent_samples, z.cpu()])
            if len(latent_samples) >= samples_size:
                break
    pca, projected_latents = get_pca_reduction(latent_samples)

    #creates the environment and starts loop for random agent
    env = make_env(args)
    steps = 100
    columns_per_frame = [[] for _ in range(steps)]

    def np_image_to_torch(single_image_array):
        single_image_tensor = torch.from_numpy(single_image_array.copy()).to(args.device)
        if single_image_tensor.dtype == torch.uint8:
            single_image_tensor = single_image_tensor.float().div(255)
        single_image_tensor = single_image_tensor.permute(2,0,1)
        single_image_tensor = single_image_tensor.reshape(1, vae_model.flattened_dim)
        return single_image_tensor

    def torch_to_np_image(single_image_tensor, denormalize=False):
        if denormalize:
            single_image_tensor = single_image_tensor.mul(255).type(torch.uint8)
        single_image_tensor = single_image_tensor.reshape(vae_model.flattened_dim)
        single_image_tensor = single_image_tensor.view(vae_model.channels_dim, vae_model.x_y_dim, vae_model.x_y_dim)
        single_image_tensor = single_image_tensor.permute(1, 2, 0)
        return single_image_tensor.cpu().numpy()

    with torch.no_grad():
        for _ in range(3):
            env.reset()
            image_goal = env.env.env.sample_goal_info(64, 64)
            image_goal = np_image_to_torch(image_goal)#pytorch uses NCHW
            mu, logvar = vae_model.encode(image_goal)
            latent_goal = vae_model.reparameterize(mu, logvar).cpu()
            #back to numpy
            image_goal = torch_to_np_image(image_goal, denormalize=True)
            for i in range(steps):
                env.step(env.action_space.sample())
                image_state = env.render(mode='rgb_array', width=64, height=64)
                image_state = np_image_to_torch(image_state)
                mu, logvar = vae_model.encode(image_state)
                latent_state = vae_model.reparameterize(mu, logvar)
                rec = vae_model.decode(latent_state)
                latent_state = latent_state.cpu()
                rec = torch_to_np_image(rec, denormalize=True)
                image_state = torch_to_np_image(image_state, denormalize=True)
                latent_image = visualize_reduced_state(pca, projected_latents, latent_state, latent_goal)
                stacked = stack_images_column([image_state, rec, latent_image], indices_env_ims=[0, 1], others_first=2)
                columns_per_frame[i].append(stacked)
    for i in range(steps):
        frame = stack_images_row(columns_per_frame[i])
        store_image_array_at(frame, args.dirpath + 'images_for_video/', 'frame_'+str(i))

'''
def calculate_mean_dist_and_std(dataset, cvae):
    batches = make_batches(dataset['next_obs'], batch_size=batch_size)
    assert isinstance(cvae, CVAE_Keras)
    zs = []
    for batch_index, im_batch in enumerate(batches):
        z_batch, mu_batch, log_sigma_batch = cvae.encode_batch(im_batch)
        for m in mu_batch:
            zs.append(m)
    zs = np.array(zs)
    dist_mu = zs.mean(axis=0)
    dist_std = zs.std(axis=0)
    return dist_mu, dist_std

#once for each histpgram
def process_histogram(dist_mu, dist_std, imsize, given_histogram, noisy, latent_state, latent_goal):
    vmin = 0
    vmax = 5
    histogram = given_histogram[0][1]
    if noisy:
        use_true_prior = True
    else:
        use_true_prior = False

    return get_image_latent_plt(dist_std, dist_mu, np.transpose(histogram), imsize, vmin, vmax, latent_state,
                                latent_goal, extent=[-2.5, 2.5, -2.5, 2.5], use_true_prior=use_true_prior)

def calculate_latent_histogram(dist_mu, dist_std, images_dataset, cvae):
    num_active_dims = 0
    for std in dist_std:
        if std > 0.15:
            num_active_dims += 1

    #active_dims = dist_std.argsort()[-num_active_dims:][::-1]
    #inactive_dims = dist_std.argsort()[:-num_active_dims][::-1]
    sampled_idx = np.random.choice(images_dataset.shape[0], 10000)
    imgs = images_dataset[sampled_idx]

    num_local_batch = 2500
    latents_mu, latents_z = None, None
    for i in range(0, 10000, num_local_batch):
        z_batch, mu_batch, log_sigma_batch = cvae.encode_batch(imgs[i:i + num_local_batch])
        if latents_mu is None:
            latents_mu = mu_batch
            latents_z = z_batch
        else:
            latents_mu = np.concatenate((latents_mu, mu_batch), axis=0)
            latents_z = np.concatenate((latents_z, z_batch), axis=0)
    num_dims = min(len(dist_std), 10)
    latent_histogram_mu = [[] for _ in range(num_dims)]
    latent_histogram_z = [[] for _ in range(num_dims)]
    nx, ny = 50, 50

    sorted_dims = dist_std.argsort()[::-1]
    for i in range(num_dims):
        for j in range(num_dims):
            if i < j:
                histogram = np.zeros((nx, ny))
                dims = [sorted_dims[i], sorted_dims[j]]
                latents = latents_mu[:, dims]
                mu = dist_mu[dims]
                std = dist_std[dims]
                latents = (latents - mu) / std
                lower_bounds = np.array([-2.5, -2.5])
                upper_bounds = np.array([2.5, 2.5])
                diffs = (upper_bounds - lower_bounds) / (nx, ny)
                latent_indices = ((latents - lower_bounds) / diffs).astype(int)

                for idx in latent_indices:
                    if np.any(idx < 0) or np.any(idx > (nx - 1, ny - 1)):
                        continue
                    histogram[tuple(idx)] += 1
            else:
                histogram = np.array([[0.0]])
            latent_histogram_mu[i].append(histogram)

    for i in range(num_dims):
        for j in range(num_dims):
            if i < j:
                histogram = np.zeros((nx, ny))
                dims = [sorted_dims[i], sorted_dims[j]]
                latents = latents_z[:, dims]
                mu = np.zeros(len(dims))
                std = np.ones(len(dims))
                latents = (latents - mu) / std

                lower_bounds = np.array([-2.5, -2.5])
                upper_bounds = np.array([2.5, 2.5])
                diffs = (upper_bounds - lower_bounds) / (nx, ny)
                latent_indices = ((latents - lower_bounds) / diffs).astype(int)

                for idx in latent_indices:
                    if np.any(idx < 0) or np.any(idx > (nx - 1, ny - 1)):
                        continue
                    histogram[tuple(idx)] += 1
            else:
                histogram = np.array([[0.0]])
            latent_histogram_z[i].append(histogram)

    return latent_histogram_mu, latent_histogram_z'''

from gym.envs.robotics.fetch.push_labyrinth2 import extract_gripper_pos_from_observation
import io
def pass_dataset_to_images(args):
    with open(args.datapath, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['next_state_image'].shape[0]

    #Create csv file
    csv_file_path = args.dirpath + args.vae_training_images_folder + args.training_csv_file
    fieldnames = ['image','gripper_pos', 'object_pos']
    if os.path.isfile(csv_file_path):
        raise Exception('This csv file already exists')

    with open(csv_file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            store_image_array_at(dataset['next_state_image'][i], args.dirpath+args.vae_training_images_folder,
                                 args.training_images_prefix+str(i))
            gripper_pos = extract_gripper_pos_from_observation(dataset['next_state'][i])
            out = io.StringIO()
            np.savetxt(out, gripper_pos)
            object_pos = dataset['next_achieved_goal'][i]
            outobj = io.StringIO()
            np.savetxt(outobj, object_pos)
            row = {'image': args.training_images_prefix+str(i), 'gripper_pos': out.getvalue(),
                   'object_pos': outobj.getvalue()}
            writer.writerow(row)

        if args.data_has_goal_ims:
            step = args.interaction_steps
            for t in range(0, N, step):
                i += 1
                store_image_array_at(dataset['goal_image'][t], args.dirpath + args.vae_training_images_folder,
                                     args.training_images_prefix + str(i))
                gripper_pos = extract_gripper_pos_from_observation(dataset['goal_state'][t])
                out = io.StringIO()
                np.savetxt(out, gripper_pos)
                object_pos = dataset['goal_pos'][t]
                outobj = io.StringIO()
                np.savetxt(outobj, object_pos)
                row = {'image': args.training_images_prefix + str(i), 'gripper_pos': out.getvalue(),
                       'object_pos': outobj.getvalue()}
                writer.writerow(row)

if __name__=='__main__':
    args = get_args_and_initialize()
    pass_dataset_to_images(args)
    #create_vae_result_images(args, 'vae_weights_950')
    #make_video(args.dirpath+'images_for_video/', '.png')
