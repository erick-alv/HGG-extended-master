import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

def show_points(points_list, save_file, space_of):
    if space_of == 'real':
        support_points = np.array([[1.05, 0.5, 0.43], [1.05, 1.0, 0.43], [1.55, 0.5, 0.43], [1.55, 1.0, 0.43]])
    elif space_of == 'latent':
        support_points = np.array([[-1., -1., 0.], [-1., 1, 0.], [1, -1., 0.], [1, 1, 0.]])
    
    if points_list.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_list[:, 0], points_list[:, 1], points_list[:, 2])
        ax.scatter(support_points[:, 0], support_points[:, 1], support_points[:, 2], c='red')
        plt.savefig('{}_3D'.format(save_file))
        plt.close()
        show_points(points_list[:, :2], save_file+'_2D', space_of)
    elif points_list.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(points_list[:, 0], points_list[:, 1])
        ax.scatter(support_points[:, 0], support_points[:, 1], c='red')
        plt.savefig(save_file)
        plt.close()
    else:
        print('cannot create visualization of shape {}'.format(points_list.shape))
