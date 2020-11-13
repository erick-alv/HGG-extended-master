import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir
from common import get_args, load_vaes
import tensorflow as tf
from vae_env_inter import take_goal_image, take_obstacle_image, take_env_image, transform_image_to_latent_batch_torch
from utils.image_util import create_rollout_video
import matplotlib.pyplot as plt

class Player:
    def __init__(self, args):
        # initialize environment
        self.args = args
        if args.vae_dist_help:
            load_vaes(args)
        self.env = make_env(args)
        self.args.timesteps = self.env.env.env.spec.max_episode_steps
        self.env_test = make_env(args)
        self.info = []
        self.test_rollouts = 1

        # get current policy from path (restore tf session + graph)
        self.play_dir = args.play_path
        self.play_epoch = args.play_epoch
        self.meta_path = "{}saved_policy-{}.meta".format(self.play_dir, self.play_epoch)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(self.meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.play_dir))
        graph = tf.get_default_graph()
        self.raw_obs_ph = graph.get_tensor_by_name("raw_obs_ph:0")
        self.pi = graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")


    def my_step_batch(self, obs):
        # compute actions from obs based on current policy by running tf session initialized before
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions

    def play(self):
        # play policy on env
        env = self.env
        acc_sum, obs = 0.0, []
        #acs = [np.array([0., 1., 0., 0.]) for _ in range(25)] + [np.array([0., 0., 0., 0.]) for _ in range(110)]
        for t in range(self.test_rollouts):
            ob = env.reset()
            #env.env.env._move_object(position=[1.13, 0.75, 0.425])

            obs.append(goal_based_process(ob))
            trajectory_goals = [ob['achieved_goal'].copy()]
            #trajectory_goals_latents = [ob['achieved_goal_latent'].copy()]

            trajectory_obstacles = [ob['real_obstacle_info'].copy()]
            #trajectory_obstacles_latents = [ob['obstacle_latent'].copy()]
            #trajectory_obstacles_latents_sizes = [ob['obstacle_size_latent'].copy()]

            tr_env_images = [take_env_image(self.env, args.img_size)]




            for timestep in range(self.args.timesteps):
                actions = self.my_step_batch(obs)
                #actions = [env.action_space.sample() for _ in range(len(obs))]
                #actions = [acs[timestep]]
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                trajectory_goals.append(ob['achieved_goal'].copy())
                #trajectory_goals_latents.append(ob['achieved_goal_latent'].copy())

                trajectory_obstacles.append(ob['real_obstacle_info'].copy())
                #trajectory_obstacles_latents.append(ob['obstacle_latent'].copy())
                #trajectory_obstacles_latents_sizes.append(ob['obstacle_size_latent'].copy())

                tr_env_images.append(take_env_image(self.env, args.img_size))

                infos.append(info)


            if t % 5 == 0 or t == self.test_rollouts -1:
                steps = np.arange(len(tr_env_images))
                #latent_ind_x = 1
                #latent_ind_y = 0
                latent_ind_x = 0
                latent_ind_y = 1

                plt.plot(steps, np.array(trajectory_goals)[:, 0], label='real')
                #plt.plot(steps, map_x_table(np.array(trajectory_goals_latents)[:, latent_ind_x]), label='latent')
                plt.title('positions_x_goals')
                plt.legend(loc=4)
                plt.savefig("{}it_{}_positions_x_goals.png".format(args.logger.my_log_dir, t))
                plt.close()

                plt.plot(steps, np.array(trajectory_goals)[:, 1], label='real')
                #plt.plot(steps, map_y_table(np.array(trajectory_goals_latents)[:, latent_ind_y]), label='latent')
                plt.title('positions_y_goals')
                plt.legend(loc=4)
                plt.savefig("{}it_{}_positions_y_goals.png".format(args.logger.my_log_dir, t))
                plt.close()

                '''plt.plot(steps, np.array(trajectory_obstacles)[:, 0], label='real')
                plt.plot(steps, map_x_table(np.array(trajectory_obstacles_latents)[:, 0, latent_ind_x]), label='latent')
                plt.title('positions_x_obstacles')
                plt.legend(loc=4)
                plt.savefig("{}it_{}_positions_x_obstacles.png".format(args.logger.my_log_dir, i))
                plt.close()



                plt.plot(steps, np.array(trajectory_obstacles)[:, 1], label='real')
                plt.plot(steps, map_y_table(np.array(trajectory_obstacles_latents)[:, 0, latent_ind_y]), label='latent')
                plt.title('positions_y_obstacles')
                plt.legend(loc=4)
                plt.savefig("{}it_{}_positions_y_obstacles.png".format(args.logger.my_log_dir, i))
                plt.close()

                plt.plot(steps, np.array(trajectory_obstacles)[:, 3], label='real')
                plt.plot(steps, map_y_table(np.array(trajectory_obstacles_latents_sizes)[:, 0, latent_ind_x]), label='latent')
                plt.title('sizes_x_obstacles')
                plt.legend(loc=4)
                plt.savefig("{}it_{}_sizes_x_obstacles.png".format(args.logger.my_log_dir, i))
                plt.close()

                plt.plot(steps, np.array(trajectory_obstacles)[:, 4], label='real')
                plt.plot(steps, map_y_table(np.array(trajectory_obstacles_latents_sizes)[:, 0, latent_ind_y]), label='latent')
                plt.title('sizes_y_obstacles')
                plt.legend(loc=4)
                plt.savefig("{}it_{}_sizes_y_obstacles.png".format(args.logger.my_log_dir, i))
                plt.close()'''

                create_rollout_video(tr_env_images, args=self.args, filename='play_it_{}'.format(t))

def interval_map_function(a,b,c, d):
    def map(x):
        return c + (d - c)/ (b-a) * (x-a)
    return map

map_x_table = interval_map_function(-1., 1., 1.05, 1.55)
map_y_table = interval_map_function(-1, 1, 0.5, 1.)


if __name__ == "__main__":
    # Call play.py in order to see current policy progress
    args = get_args()
    player = Player(args)
    player.play()