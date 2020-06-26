import numpy as np
def conc_latent(raw_obs, latent):
    return np.concatenate((raw_obs, latent))

def rem_latent(state, latent_dim):
    raw_obs = state[:-latent_dim]
    latent = state[-latent_dim:]
    return raw_obs, latent

def from_raw_get_achieved(raw_obs):
    return raw_obs[3:6]