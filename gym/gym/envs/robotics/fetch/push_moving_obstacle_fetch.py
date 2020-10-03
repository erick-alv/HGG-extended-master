import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
import copy

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_moving_obstacle_fetch.xml')

class FetchPushMovingObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        # TODO: configure adaption parameters
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]

        #centers of the interval where goal and initial position will be sampld
        self.target_goal_center = np.array([1.3, 0.57, 0.425])
        self.object_center = np.array([1.3, 0.93, 0.425])
        #for moving
        self.obstacle_vel = 8.
        self.initial_obstacle_direction = 1
        self.obstacle_direction = 1
        #limits are not 100% percent accurate; cahnging the range and margin parameters from the XML help to improve
        #this accuracy
        self.obstacle_upper_limit = 1.35
        self.obstacle_lower_limit = 1.25


        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.3, 0.93, 0.42505, 1., 0., 0., 0.],  # origin 0.53
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.02, target_range=0.02, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.obstacle_slider_idx = self.sim.model.joint_names.index('obstacle:joint')

    # RobotEnv methods
    # ----------------------------

    '''def _step_callback(self):
        # direction is one dimensional see type = 'slide' in mujoco doc
        body_id = self.sim.model.body_name2id('obstacle')
        mov_obst_center = self.sim.data.body_xpos[body_id]
        if mov_obst_center[0] >= self.obstacle_upper_limit and self.obstacle_direction == 1:
            self.obstacle_direction = -1
        elif mov_obst_center[0] <= self.obstacle_lower_limit and self.obstacle_direction == -1:
            self.obstacle_direction = 1
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        if self.obstacle_direction == 1 and \
                self.sim.model.body_pos[body_id][0] + self.obstacle_vel * dt > self.obstacle_upper_limit:
            self.sim.model.body_pos[body_id][0] = self.obstacle_upper_limit
        elif self.obstacle_direction == -1 and \
                self.sim.model.body_pos[body_id][0] - self.obstacle_vel * dt < self.obstacle_lower_limit:
            self.sim.model.body_pos[body_id][0] = self.obstacle_lower_limit
        else:
            self.sim.model.body_pos[body_id][0] += self.obstacle_vel * self.obstacle_direction * dt
        super(FetchPushMovingObstacleEnv, self)._step_callback()'''

    def set_obstacle_slide_pos(self, pos):
        # move obstacle
        qpos = self.sim.data.qpos.flat[:]
        qpos[self.obstacle_slider_idx] = pos
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qpos=qpos)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def set_obstacle_slide_vel(self, vel):
        qvel = self.sim.data.qvel.flat[:]
        qvel[self.obstacle_slider_idx] = vel
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qvel=qvel)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def move_obstacle(self):
        body_id = self.sim.model.body_name2id('obstacle')
        if self.sim.data.body_xpos[body_id][0] >= self.obstacle_upper_limit and self.obstacle_direction == 1:
            self.obstacle_direction = -1
        elif self.sim.data.body_xpos[body_id][0] <= self.obstacle_lower_limit and self.obstacle_direction == -1:
            self.obstacle_direction = 1
        self.set_obstacle_slide_vel(self.obstacle_vel*self.obstacle_direction)

    def step(self, action):
        self.move_obstacle()
        return super(FetchPushMovingObstacleEnv, self).step(action)


    def _sample_goal(self):
        goal = self.target_goal_center + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.obstacle_direction = self.initial_obstacle_direction
        self.set_obstacle_slide_vel(self.obstacle_vel * self.obstacle_direction)

        object_xpos = self.object_center[:2] + self.np_random.uniform(-self.obj_range,
                                                                      self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()

        return True

    def _get_obs(self):
        obs = super(FetchPushMovingObstacleEnv, self)._get_obs()
        body_id = self.sim.model.body_name2id('obstacle')
        obs['real_obstacle_pos'] = self.sim.data.body_xpos[body_id].copy()
        return obs

'''if __name__ == '__main__':
    import time
    from utils.image_util import rgb_array_to_image
    import cv2
    e = FetchPushMovingObstacleEnv()
    im_size = 500
    ims = []
    for episode in range(5):
        ob = e.reset()
        for step in range(100):
            array = e.render(mode='rgb_array')
            image = rgb_array_to_image(array)
            ims.append(image)
            a = e.action_space.sample()
            e.step(a)
            obs = e._get_obs()
    e.close()
    out = cv2.VideoWriter('/home/erick/RL/HGG-extended/HGG-Extended-master/vid.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          10, (im_size, im_size))
    for i in range(len(ims)):
        out.write(np.array(ims[i]))
    out.release()'''