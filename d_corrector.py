import torch
import numpy as np
from vae_env_inter import transform_image_to_latent_batch_torch
from j_vae.latent_space_transformations import torch_goal_transformation
def correct(goal_list, args):
    #todo make normal RL with the goal list

    #todo check if Success to the goalList and pass it

    pass

def correct_representation(rollout_ims, args):#[r1, r2] = [[step_1_im, step2_im, ..., step_last_im], [step_1_im, step2_im, ..., step_last_im]]
    all_ims = np.concatenate(rollout_ims, axis=0)
    args.vae_model_goal.train()
    all_mu_torch, all_logvar_torch = transform_image_to_latent_batch_torch(all_ims, args.vae_model_goal, args.img_size, args.device)
    pass