import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_moving_obstacle_fetch.xml')

class FetchPushMovingObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        # TODO: configure adaption parameters
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]


        self.target_goal_center = np.array([1.3, 0.57, 0.425])
        self.object_center = np.array([1.3, 0.93, 0.425])
        #for moving
        self.obstacle_vel = 0.055
        self.initial_obstacle_direction = 1
        self.obstacle_direction = 1
        self.obstacle_upper_limit = 1.392
        self.obstacle_lower_limit = 1.208


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

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        # direction is one dimensional see type = 'slide' in mujoco doc
        body_id = self.sim.model.body_name2id('obstacle')
        mov_obst_center = self.sim.data.body_xpos[body_id]
        if mov_obst_center[0] >= self.obstacle_upper_limit and self.obstacle_direction == 1:
            self.obstacle_direction = -1
        elif mov_obst_center[0] <= self.obstacle_lower_limit and self.obstacle_direction == -1:
            self.obstacle_direction = 1

        # self.sim.data.set_joint_qpos('keeper0:joint', obj_qpos + 0.01 * self.obstacle_direction)
        #for name in names_list:
        #    id = self.sim.model.body_name2id(name)
        #    self.sim.model.body_pos[id] = position
        #self.sim.forward()
        self.sim.model.body_pos[body_id][0] += self.obstacle_vel * self.obstacle_direction
        super(FetchPushMovingObstacleEnv, self)._step_callback()

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
        self.obstacle_direction = self.initial_obstacle_direction
        return True