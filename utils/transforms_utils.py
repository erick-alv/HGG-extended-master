from torchvision import transforms
import torch
from collections import namedtuple
import numpy as np

BA = namedtuple('BA', 'buffer_type buffer_size')

image_to_tensor_funtion = transforms.ToTensor()

def transform_np_image_to_tensor(image, args):
    return image_to_tensor_funtion(image.copy()).to(args.device)

def np_to_tensor(np_vector, args):
    return torch.from_numpy(np_vector.copy()).float().to(args.device)

def extend_obs(obs, args, ae=None):
    '''
    :param ae: An Autoencoder or Varational autoencoder
    :param obs: the disctionary received by the environment
    :param args: args of the experiment
    :return:
    '''
    #TODO concat per default
    if ae is not None and (args.observation_type == 'latent' or args.observation_type == 'concat'):
        f = transforms.ToTensor()
        with torch.no_grad():
            img_s = f(obs['state_image'].copy()).to(args.device)
            obs['state_latent'] = ae.encode(img_s).cpu().data.numpy()[0]
            img_g = f(obs['goal_image'].copy()).to(args.device)
            obs['goal_latent'] = ae.encode(img_g).cpu().data.numpy()[0]

    return obs