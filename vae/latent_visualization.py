#adapted from
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import cv2


def get_pca_reduction(latent_samples):
    pca = PCA(n_components=2)
    #calculates the reduction and applies it
    projected_latents = pca.fit_transform(latent_samples)
    return pca, projected_latents

def visualize_reduced_state(reducer, projected_latents, state_z, goal_z = None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.scatter(projected_latents[:, 0], projected_latents[:, 1], edgecolors='none', alpha=0.05, c='black')
    assert isinstance(reducer, PCA)
    #plot state
    state_z = state_z.reshape(1, -1)#assumes just one sample given
    state_pr = reducer.transform(state_z)
    ax.scatter(state_pr[:, 0], state_pr[:, 1], alpha=1., c='blue', s=100)
    if goal_z is not None:
        goal_z = goal_z.reshape(1,-1)
        goal_pr = reducer.transform(goal_z)
        ax.scatter(goal_pr[:, 0], goal_pr[:, 1], alpha=1., c='red', s=100)
    assert isinstance(fig, plt.Figure)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data