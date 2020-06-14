from PIL import Image
import numpy as np
import argparse
import os
import fnmatch
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--npy_save_path', type=str)
    parser.add_argument('--npy_path', type=str)
    parser.add_argument('--image_save_path', type=str)
    return parser.parse_args()
def image_to_npy(image_path,npy_save_path, show=False):
    image = Image.open(image_path)
    image_array = np.array(image)
    np.save(npy_save_path, image_array)
    if show:
        print(image_array)

def save_and_show_npy_to_image(npy_path, image_save_path, display=False):
    def process_image(single_image_array, image_path):
        image = Image.fromarray(single_image_array, 'RGB')
        image.save(image_path)
        if display:
            image.show()

    npy_as_array = np.load(npy_path)
    if npy_as_array.ndim > 3:
        print('more than one image')
        max_times = min(10, npy_as_array.shape[0])
        name = None
        if image_save_path.endswith('.png'):
            name = image_save_path[:-4]
        else:
            name = image_save_path
        for i in range(max_times):
            process_image(npy_as_array[i], name+'_'+str(i)+'.png')
    else:
        process_image(npy_as_array, image_save_path)


def store_image_array_at(single_image_array, path_to_folder, img_name, force_remap_to_255=False ):
    '''
    :param dict:
    :param dict_key: key for the dict, where the images should be
    :return:
    '''
    '''files = [f for f in os.listdir(path_to_folder) if fnmatch.fnmatch(f, '*_id****.png')]
    print(files)'''
    if single_image_array.max() <= 1.0 or force_remap_to_255:
        #assume is normalized in [0,1]
        single_image_array *= 255.0

    image = Image.fromarray(single_image_array.astype(np.uint8), 'RGB')
    image.save(path_to_folder+img_name+'.png')

def make_random_imarray():
    return np.array([
        np.array([
            np.random.uniform(low=0, high=1, size=3)
            for _ in range(4)])
    for _ in range(4)])

def rgb_array_to_image(single_image_array):
    if single_image_array.max() <= 1.0:
        #assume is normalized in [0,1]
        single_image_array *= 255.0
    image = Image.fromarray(single_image_array.astype(np.uint8), 'RGB')
    return image

def denormalize(single_image_array):
    return single_image_array * 255.0

def make_video(path_to_folder, ext_end):
    img_array = []
    for filename in glob.glob(path_to_folder+'*'+ext_end):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(path_to_folder+'rollout_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def stack_images_row(images):
    w = images[0].shape[0]
    spacer = np.zeros(shape=(w, 10, 3))
    s = spacer.copy()
    for im in images:
        s = np.hstack((s, im))
        s = np.hstack((s, spacer))
    return s

def stack_images_column(images, indices_env_ims, others_first):
    if images[indices_env_ims[0]].shape[1] < images[others_first].shape[1]:
        width_envs = images[indices_env_ims[0]].shape[1]
        #first reshape others to half:
        width_others = images[others_first].shape[1]
        if width_others/3 > width_envs:
            nw = width_others//3
            nh = images[others_first].shape[0]//3
            for i in range(others_first, len(images)):
                images[i] = cv2.resize(images[i], dsize=(nw, nh), interpolation=cv2.INTER_AREA)
            width_others = nw
        new_width = 0
        while new_width + width_envs < width_others:
            new_width += width_envs
        dif_left = int(np.floor((width_others - new_width)/2))
        dif_right = int(np.ceil((width_others - new_width)/2))
        l = np.zeros(shape=(new_width, dif_left, 3))
        r = np.zeros(shape=(new_width, dif_right, 3))
        for i in indices_env_ims:
            images[i] = cv2.resize(images[i], dsize=(new_width, new_width), interpolation=cv2.INTER_CUBIC)
            images[i] = np.hstack((l, images[i]))
            images[i] = np.hstack((images[i], r))
    w = images[0].shape[1]
    spacer = np.zeros(shape=(10, w, 3))
    s = spacer.copy()
    for im in images:
        s = np.vstack((s, im))
        s = np.vstack((s, spacer))
    return s

def get_image_latent_plt(dist_std, dist_mu, vals, imsize, vmin, vmax, latent_state=None, latent_goal=None,
                         extent=[-2.5, 2.5, -2.5, 2.5], cmap='Greys',
                         top_dim_axis=None,draw_subgoals=False, use_true_prior=False):
    fig, ax = plt.subplots()
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_ylim(ax.get_ylim()[::-1])
    DPI = fig.get_dpi()
    fig.set_size_inches(imsize / float(DPI), imsize / float(DPI))

    if top_dim_axis is None:
        top_latent_indices = dist_std.argsort()[-2:][::-1]
    else:
        sorted_dims = dist_std.argsort()[::-1]
        top_latent_indices = sorted_dims[np.array(top_dim_axis)]

    if use_true_prior:
        mu = np.zeros(len(top_latent_indices))
        std = np.ones(len(top_latent_indices))
    else:
        mu = dist_mu[top_latent_indices]
        std = dist_std[top_latent_indices]

    if latent_state is not None:
        obs_pos = latent_state[top_latent_indices]
        obs_pos = (obs_pos - mu) / std

        ball = plt.Circle(obs_pos, 0.12, color='dodgerblue')
        ax.add_artist(ball)
    if latent_goal is not None:
        goal_pos = latent_goal[top_latent_indices]
        goal_pos = (goal_pos - mu) / std

        goal = plt.Circle(goal_pos, 0.12, color='lime')
        ax.add_artist(goal)
    '''if draw_subgoals and vae_env.latent_subgoals is not None:
        if use_true_prior:
            latent_subgoals = vae_env.latent_subgoals[:, top_latent_indices]
        else:
            latent_subgoals = vae_env.latent_subgoals_reproj[:, top_latent_indices]
        for subgoal in latent_subgoals:
            latent_pos = (subgoal - mu) / std
            sg = plt.Circle(latent_pos, 0.12, color='red')
            ax.add_artist(sg)'''

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    ax.axis('off')

    ax.imshow(
        vals,
        extent=extent,
        cmap=plt.get_cmap(cmap),
        interpolation='nearest',
        vmax=vmax,
        vmin=vmin,
        origin='bottom',
    )

    return plt_to_numpy(fig)

def plt_to_numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
