from gym.wrappers.pixel_observation import PixelObservationWrapper
import torch
class PixelAndGoalWrapper(PixelObservationWrapper):
    def __init__(self, args, *oargs, **kwargs):
        super(PixelAndGoalWrapper, self).__init__(*oargs, **kwargs)
        # charge the VAE to CPU so other networks can use device

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
        goal_image = self.env.env.goal_image
        goal_state = self.env.env.goal_state
        extra_obs = {
            'goal_image': goal_image,
            'goal_state': goal_state
        }
        obs.update(extra_obs)
        return obs