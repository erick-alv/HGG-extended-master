
center_obstacle = [1.3, 0.75, 0.435]
range_x=[1.05, 1.55]
range_y=[0.5, 1.0]
obstacle_size = {'FetchPushObstacleFetchEnv-v1': 0.13,
                 'FetchPushMovingObstacleEnv-v1': (0.1, 0.03),
                 }

min_obstacle_size = {'FetchPushObstacleFetchEnv-v1': 0.05,
                     'FetchPushMovingObstacleEnv-v1': 0.025,
                     }

max_obstacle_size = {'FetchPushObstacleFetchEnv-v1': (range_x[1] - range_x[0])/2.,
                     'FetchPushMovingObstacleEnv-v1': 0.25,
                     }


puck_size = 0.045
z_table_height=0.43
z_table_height_obstacle = 0.435
z_table_height_goal = 0.425


### configuration information ####
train_file_name = {'goal': 'goal_set.npy',
                   'obstacle':'obstacle_set.npy',
                   'obstacle_sizes':'obstacle_sizes_set.npy',
                   'goal_sizes':'goal_sizes_set.npy',
                   'mixed': 'mixed.npy',
                   'all':'all_set.npy',
                   'space':'n.npy',
                   }
file_corners_name = {'goal': 'goal_corners.npy',
                     'obstacle':'obstacle_corners.npy',
                     'mixed':'mixed_corners.npy',

                     }




vae_sb_weights_file_name = {'goal': 'vae_sb_model_goal',
                            'obstacle':'vae_sb_model_obstacle',
                            'mixed':'vae_sb_model_mixed',
                            'all':'all_sb_model',
                            'space':'../data/FetchGenerativeEnv-v1/model_000030001.pth'
                            }

vae_weights_file_name = {
                   'obstacle_sizes':'vae_model_obstacle_sizes',
                   'goal_sizes':'vae_model_goal_sizes'
                   }
