import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir
from common import get_args, load_vaes
import tensorflow as tf
from vae_env_inter import take_goal_image, take_obstacle_image, take_env_image, transform_image_to_latent_batch_torch
from utils.image_util import create_rollout_video
import matplotlib.pyplot as plt
from mujoco_py import functions

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
        self.test_rollouts = 12

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

        for t in range(self.test_rollouts):
            rand_steps_wait = np.random.randint(low=12, high=18)
            acs = [np.array([0., 0., 0., 0.]) for _ in range(rand_steps_wait)] + [
                np.array([-1., 0., 0., 0.]) for _ in range(3)] + [np.array([0., -1., 0., 0.]) for _ in range(7)] + [
                np.array([1., 1., 0., 0.]) for _ in range(100)]
            '''rand_steps_wait = np.random.randint(low=6, high=12)
            acs = [np.array([0., 0., -.5, 0.]) for _ in range(3)] + [np.array([0.672, -0.8, 0, 0.]) for _ in range(3)] + \
                  [np.array([0., 0., -.4, 0.]) for _ in range(2)]+ \
                  [np.array([0., 0.5, 0., 0.]) for _ in range(10)] + [np.array([-0.1, 0., 0., 0.]) for _ in range(2)]+ \
                  [np.array([0.2, -0.8, 0., 0.]) for _ in range(3)] + [np.array([0.4, 0., 0., 0.]) for _ in range(3)] + \
                  [np.array([0., 1., 0., 0.]) for _ in range(4)] + [np.array([-0.6, -0.2, 0., 0.]) for _ in range(4)] + \
                  [np.array([0., 0.89, 0., 0.]) for _ in range(4)] + [np.array([0., 0., 0., 0.]) for _ in range(rand_steps_wait)] + \
                  [np.array([1., 0., 0., 0.]) for _ in range(5)] + [np.array([0.5, -0.5, 0., 0.]) for _ in range(8)] + \
                  [np.array([0., 0., 0., 0.]) for _ in range(110)]'''
            '''acs = [np.array([1., 0., 0., 0.]) for _ in range(3)] + \
                  [np.array([0., -.5, 0., 0.]) for _ in range(6)] + \
                  [np.array([0., 0., -.6, 0.]) for _ in range(2)] + \
                  [np.array([-0.9, 0., 0, 0.]) for _ in range(10)] + \
                  [np.array([0., 1., 0., 0.]) for _ in range(4)] + \
                  [np.array([-0.8, 0., 0., 0.]) for _ in range(2)] + \
                  [np.array([0., 0., 0., 0.]) for _ in range(rand_steps_wait)] +\
                  [np.array([0., -1., 0., 0.]) for _ in range(15)]+ \
                  [np.array([0., 0., 0., 0.]) for _ in range(110)]'''
            #acs = [np.array([0., 0., 0., 0.]) for _ in range(100)]
            #env.env.env._move_object(position=[1.13, 0.75, 0.425])
            ob = env.reset()
            obs.append(goal_based_process(ob))
            trajectory_goals = [ob['achieved_goal'].copy()]
            #trajectory_goals_latents = [ob['achieved_goal_latent'].copy()]

            trajectory_obstacles = [ob['real_obstacle_info'].copy()]
            #trajectory_obstacles_latents = [ob['obstacle_latent'].copy()]
            #trajectory_obstacles_latents_sizes = [ob['obstacle_size_latent'].copy()]

            tr_env_images = [take_env_image(self.env, args.img_vid_size)]

            for timestep in range(self.args.timesteps):
                #actions = self.my_step_batch(obs)
                #actions = [env.action_space.sample() for _ in range(len(obs))]
                actions = [acs[timestep]]
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                trajectory_goals.append(ob['achieved_goal'].copy())
                #trajectory_goals_latents.append(ob['achieved_goal_latent'].copy())

                trajectory_obstacles.append(ob['real_obstacle_info'].copy())
                #trajectory_obstacles_latents.append(ob['obstacle_latent'].copy())
                #trajectory_obstacles_latents_sizes.append(ob['obstacle_size_latent'].copy())

                tr_env_images.append(take_env_image(self.env, args.img_vid_size))
                infos.append(info)
                sim = env.sim
                '''print('________ contacts at step {} -------'.format(timestep))
                for i in range(sim.data.ncon):
                    # Note that the contact array has more than `ncon` entries,
                    # so be careful to only read the valid entries.

                    contact = sim.data.contact[i]
                    geom1_name = sim.model.geom_id2name(contact.geom1)
                    geom2_name = sim.model.geom_id2name(contact.geom2)
                    if geom1_name == 'object0' or geom2_name == 'object0':

                        print('contact', i)
                        print('dist', contact.dist)
                        print('geom1', contact.geom1, sim.model.geom_id2name(contact.geom1))
                        print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
                        # There's more stuff in the data structure
                        # See the mujoco documentation for more info!
                        geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
                        print(' Contact force on geom2 body', sim.data.cfrc_ext[geom2_body])
                        print('norm', np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))
                        # Use internal functions to read out mj_contactForce
                        c_array = np.zeros(6, dtype=np.float64)
                        print('c_array', c_array)
                        functions.mj_contactForce(sim.model, sim.data, i, c_array)
                        print('c_array', c_array)

                print('done')'''


            if t % 1 == 0 or t == self.test_rollouts -1:
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
        if hasattr(env.env.env, 'reset_sim_counter'):
            print(env.env.env.reset_sim_counter)

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