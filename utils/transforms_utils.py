from torchvision import transforms
import torch

image_to_tensor_funtion = transforms.ToTensor()

def transform_np_image_to_tensor(image, args):
    return image_to_tensor_funtion(image.copy()).to(args.device)

def np_to_tensor(np_vector, args):
    return torch.from_numpy(np_vector.copy()).float().to(args.device)

def extend_obs(ae, obs, args):
    '''
    :param ae: An Autoencoder or Varational autoencoder
    :param obs: the disctionary received by the environment
    :param args: args of the experiment
    :return:
    '''
    #TODO concat per default
    if args.observation_type == 'latent' or args.observation_type == 'concat':
        f = transforms.ToTensor()
        with torch.no_grad():
            img_s = f(obs['state_image'].copy()).to(args.device)
            obs['state_latent'] = ae.encode(img_s).cpu().data.numpy()[0]
            img_g = f(obs['goal_image'].copy()).to(args.device)
            obs['goal_latent'] = ae.encode(img_g).cpu().data.numpy()[0]
        '''state_decs = ae.decode(obs['state_latent'])
        st_rec = state_decs[0]
        goal_decs = ae.decode(obs['goal_latent'])
        g_rec = goal_decs[0]
        save_image(img_s.cpu(), args.dirpath + 'temp/'+ 'real_state.png')
        save_image(torch.reshape(st_rec, (args.img_channels, args.img_dim, args.img_dim)).cpu(), args.dirpath + 'temp/' + 'rec_state.png')
        save_image(img_g.cpu(), args.dirpath + 'temp/' + 'real_goal.png')
        save_image(torch.reshape(g_rec, (args.img_channels, args.img_dim, args.img_dim)).cpu(), args.dirpath + 'temp/' + 'rec_goal.png')'''
        return obs
    else:
        return obs