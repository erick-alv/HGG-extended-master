lower_limits=[1.05, 0.5]
upper_limits=[1.55, 1.]
center_obstacle = [1.3, 0.75, 0.435]
#todo use range insead of lower and upper limit
range_x=[1.05, 1.55]
range_y=[0.5, 1.0]
obstacle_size = 0.13
min_obstacle_size = 0.001
max_obstacle_size = (range_x[1] - range_x[0])/2.
puck_size = 0.045
z_table_height=0.43


### configuration information ####
train_file_name = {'goal': 'goal_set.npy',
                   'obstacle':'obstacle_set.npy',
                   'obstacle_sizes':'obstacle_sizes_set.npy',
                   'goal_sizes':'goal_sizes_set.npy'
                   }
file_corners_name = {'goal': 'goal_corners.npy',
                     'obstacle':'obstacle_corners.npy',
                     }
file_center_name = {'goal': 'goal_center.npy',
                     'obstacle':'obstacle_center.npy',
                     }

vae_sb_weights_file_name = {'goal': 'vae_sb_model_goal',
                            'obstacle':'vae_sb_model_obstacle',
                            }

vae_weights_file_name = {
                   'obstacle_sizes':'vae_model_obstacle_sizes',
                   'goal_sizes':'vae_model_goal_sizes'
                   }
