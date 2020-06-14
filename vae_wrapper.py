from gym.wrappers.pixel_observation import PixelObservationWrapper
from vae.vae_torch import VAE, load_vae

class VAEWrapper(PixelObservationWrapper):
    def __init__(self, args, *oargs, **kwargs):
        super(VAEWrapper, self).__init__(*oargs, **kwargs)
        vae_model = VAE(84, 3, 20).to(args.device)
        load_vae(args, args.vae_wrap_filename, vae_model, trainer=None)

    def _get_obs(self):
        obs = self.env.env._get_obs()
        obs = super().observation(obs)
        goal = self.env.env.goal_image
        extra_obs = {
            'image_goal':goal
        }
        obs.update(extra_obs)
        return obs