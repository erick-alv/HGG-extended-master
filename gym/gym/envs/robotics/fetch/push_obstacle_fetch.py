import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_obstacle_fetch.xml')

class FetchPushObstacleFetchEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        # TODO: configure adaption parameters
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]
        #self.adapt_dict["obstacles"] = [[1.3 - 0.07, 0.75, 0.6 - 0.1, 0.18, 0.04, 0.1]]
        #self.adapt_dict["spaces"] = [30, 30, 10]  # [30, 30, 10]
        #self.adapt_dict["z_penalty"] = 1
        self.target_goal_center = np.array([1.3, 0.57, 0.425])
        self.object_center = np.array([1.3, 0.93, 0.425])


        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.63, 0.4, 1., 0., 0., 0.],  # origin 0.53
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.02, target_range=0.02, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self):
        goal = self.target_goal_center + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        object_xpos = self.object_center[:2] + self.np_random.uniform(-self.obj_range,
                                                                      self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True