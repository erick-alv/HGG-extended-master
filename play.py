import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir
from common import get_args
import tensorflow as tf


class Player:
    def __init__(self, args, direct_playpath=None):
        # initialize environment
        self.args = args
        self.env = make_env(args)
        self.args.timesteps = self.env.env.env.spec.max_episode_steps
        self.env_test = make_env(args)
        self.info = []
        self.test_rollouts = 100

        # get current policy from path (restore tf session + graph)
        if direct_playpath is not None:
            self.play_dir = direct_playpath
        else:
            self.play_dir = args.play_path
        self.play_epoch = args.play_epoch
        self.meta_path = "{}saved_policy-{}.meta".format(self.play_dir, self.play_epoch)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(self.meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.play_dir))
        graph = tf.get_default_graph()
        self.raw_obs_ph = graph.get_tensor_by_name("raw_obs_ph:0")
        self.pi = graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")
        #for q values
        self.acts_ph = graph.get_tensor_by_name("acts_ph:0")
        self.q_pi = graph.get_tensor_by_name("main/value/net/q/BiasAdd:0")


    def my_step_batch(self, obs):
        # compute actions from obs based on current policy by running tf session initialized before
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions


    def step_batch(self, obs):
        return self.my_step_batch(obs)

    def get_q_pi(self, obs):
        actions = self.step_batch(obs)
        feed_dict = {
            self.raw_obs_ph: obs,
            self.acts_ph: actions,
        }
        value = self.sess.run(self.q_pi, feed_dict)[:, 0]  # get the q values at each achieved state
        value = np.clip(value, self.args.clip_return_l, self.args.clip_return_r)
        return value

    def play(self):
        # play policy on env
        env = self.env
        acc_sum, obs = 0.0, []
        for i in range(self.test_rollouts):
            obs.append(goal_based_process(env.reset()))
            for timestep in range(self.args.timesteps):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                env.render()


if __name__ == "__main__":
    # Call play.py in order to see current policy progress
    args = get_args()
    player = Player(args)
    player.play()