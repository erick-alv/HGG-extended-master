
import vae.train_goal_vae as VAE_GOAL
import numpy as np
import os
this_file_dir = os.path.dirname(os.path.abspath(__file__))
#vae_obs = VAE_OBS.load_Vae(path='data/FetchPush/vae_model_obs')
vae_goal = VAE_GOAL.load_Vae(path='{}/../data/FetchPush/vae_model_goal'.format(this_file_dir))

goal_set = np.load('{}/../data/FetchPush/goal_set.npy'.format(this_file_dir))

