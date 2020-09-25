lower_limits=[1.05, 0.5]
upper_limits=[1.55, 1.]
center_obstacle = [1.3, 0.75, 0.435]
#todo use range insead of lower and upper limit
range_x=[1.05, 1.55]
range_y=[0.5, 1.0]
obstacle_size = 0.048#0.13#TODO!!!!! make dependant from env
min_obstacle_size = 0.05
max_obstacle_size = (range_x[1] - range_x[0])/2.
puck_size = 0.045
z_table_height=0.43
z_table_height_obstacle = 0.435
z_table_height_goal = 0.425


### configuration information ####
train_file_name = {'goal': 'goal_set.npy',
                   'obstacle':'obstacle_set.npy',
                   'obstacle_sizes':'obstacle_sizes_set.npy',
                   'goal_sizes':'goal_sizes_set.npy',
                   'mixed': 'mixed.npy'
                   }
file_corners_name = {'goal': 'goal_corners.npy',
                     'obstacle':'obstacle_corners.npy',
                     'mixed':'mixed_corners.npy'
                     }




vae_sb_weights_file_name = {'goal': 'vae_sb_model_goal',
                            'obstacle':'vae_sb_model_obstacle',
                            'mixed':'vae_sb_model_mixed'
                            }

vae_weights_file_name = {
                   'obstacle_sizes':'vae_model_obstacle_sizes',
                   'goal_sizes':'vae_model_goal_sizes'
                   }
