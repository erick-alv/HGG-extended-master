import vae.train_obs_vae as VAE_OBS
import vae.train_goal_vae as VAE_GOAL
import numpy as np

vae_obs = VAE_OBS.load_Vae(path='data/FetchPush/vae_model_obs')
vae_goal = VAE_GOAL.load_Vae(path='data/FetchPush/vae_model_goal')

goal_set = np.load('data/FetchPush/goal_set.npy')

