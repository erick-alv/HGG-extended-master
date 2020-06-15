from gym.wrappers.pixel_observation import PixelObservationWrapper
from vae.vae_torch import VAE, load_vae
import torch
import numpy as np

class VAEWrapper(PixelObservationWrapper):
    def __init__(self, args, *oargs, **kwargs):
        super(VAEWrapper, self).__init__(*oargs, **kwargs)
        self.vae_model = VAE(args.img_dim, 3, args.latent_dim).to('cpu')#'we want to charge the VAE to CPU so other networks can'
        #device
        path = args.dirpath + 'weights_dir/' + args.vae_wrap_filename
        save_dict = torch.load(path, map_location=torch.device('cpu'))
        self.vae_model.load_state_dict(save_dict['model_state_dict'])
        self.vae_model.eval()

    def np_image_to_torch(self, single_image_array):
        single_image_tensor = torch.from_numpy(single_image_array.copy())
        if single_image_tensor.dtype == torch.uint8:
            single_image_tensor = single_image_tensor.float().div(255)
        single_image_tensor = single_image_tensor.permute(2, 0, 1)
        single_image_tensor = single_image_tensor.reshape(1, self.vae_model.flattened_dim)
        return single_image_tensor

    def torch_to_np_image(self, single_image_tensor, denormalize=False):
        if denormalize:
            single_image_tensor = single_image_tensor.mul(255).type(torch.uint8)
        single_image_tensor = single_image_tensor.reshape(self.vae_model.flattened_dim)
        single_image_tensor = single_image_tensor.view(self.vae_model.channels_dim, self.vae_model.x_y_dim,
                                                       self.vae_model.x_y_dim)
        single_image_tensor = single_image_tensor.permute(1, 2, 0)
        return single_image_tensor.numpy()

    def _get_obs(self):
        obs = self.env.env._get_obs()
        obs = super().observation(obs)
        goal = self.env.env.goal_image
        with torch.no_grad():
            g_mu, g_logvar = self.vae_model.encode(self.np_image_to_torch(goal))
            #g_std = g_logvar.mul(0.5).exp_()
            #if noisy:
            #    eps = ptu.Variable(std.data.new(std.size()).normal_())
            #    goal_latent =eps.mul(std).add_(mu)
            goal_latent = g_mu[0].numpy()
            s_mu, s_logvar = self.vae_model.encode(self.np_image_to_torch(obs['state_image']))
            state_latent = s_mu[0].numpy()
        extra_obs = {
            'goal_image': goal,
            'goal_latent': goal_latent,
            'state_latent': state_latent
        }
        obs.update(extra_obs)
        return obs