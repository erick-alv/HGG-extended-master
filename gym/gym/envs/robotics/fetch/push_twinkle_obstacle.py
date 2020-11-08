import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
import copy

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_twinkle_obstacle.xml')

class FetchTwinkleObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        # TODO: configure adaption parameters
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]

        #centers of the interval where goal and initial position will be sampled
        self.target_goal_center = np.array([1.3, 0.57, 0.425])
        self.object_center = np.array([1.3, 0.93, 0.425])


        self.is_active = True
        self.active_steps = 0
        self.inactive_steps = 0
        self.active_steps_max = 4
        self.active_steps_min = 2
        self.inactive_steps_max = 3
        self.inactive_steps_min = 1


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
        self.obstacle_id = self.sim.model.body_name2id('obstacle')
        self.obstacle_original_pos = self.sim.data.body_xpos[self.obstacle_id].copy()


    def test_setup(self, active_steps_max=9, active_steps_min=6, inactive_steps_max=3, inactive_steps_min=1):
        '''
        changes the parameter for further tests after training an agent
        '''
        #the default values makes obstacle present for more timesteps thus difficulting to reach goal
        self.active_steps_max = active_steps_max
        self.active_steps_min = active_steps_min
        self.inactive_steps_max = inactive_steps_max
        self.inactive_steps_min = inactive_steps_min

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

    def twinkle_obstacle(self):
        if self.is_active:
            if self.counter_active_steps == 0:
                #sets inactive phase
                self.is_active = False
                self.counter_inactive_steps = self.inactive_steps
                self.set_inactive()
            else:
                self.counter_active_steps -= 1

        else:
            if self.counter_inactive_steps == 0:
                # sets active phase
                self.is_active = True
                self.counter_active_steps = self.active_steps
                self.set_active()
            else:
                self.counter_inactive_steps -= 1

    def set_active(self):
        self.sim.model.body_pos[self.obstacle_id] = self.obstacle_original_pos



    def set_inactive(self):
        #it actually sets the position of the object in another part of the world
        self.sim.model.body_pos[self.obstacle_id] = np.array([10., 10., 10.])


    def step(self, action):
        self.twinkle_obstacle()
        return super(FetchTwinkleObstacleEnv, self).step(action)


    def _sample_goal(self):
        goal = self.target_goal_center + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        return goal.copy()

    def _set_gripper_during_setup(self):
        # Move end effector into position.
        orig_pos = self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = np.array([-0.5, 0.305, -0.306 + self.gripper_extra_height]) + orig_pos
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        #set initial configuration for twinkling
        self.active_steps = np.random.randint(low=self.active_steps_min, high=self.active_steps_max+1)
        self.inactive_steps = np.random.randint(low=self.active_steps_min, high=self.active_steps_max+1)
        self.is_active = True
        self.set_active()
        self.counter_active_steps = self.active_steps


        object_xpos = self.object_center[:2] + self.np_random.uniform(-self.obj_range,
                                                                      self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[3:] = np.array([1., 0., 0., 0.])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()

        return True

    def _get_obs(self):
        obs = super(FetchTwinkleObstacleEnv, self)._get_obs()
        body_id = self.sim.model.body_name2id('obstacle')
        pos = self.sim.data.body_xpos[body_id].copy()
        if self.is_active:
            dims = np.array([0.12, 0.03, 0.025])#todo verify is correct and would be better to read this info
            ob = np.concatenate((pos, dims.copy()))
            obs['real_obstacle_info'] = ob.copy()
        else:
            obs['real_obstacle_info'] = np.array([0., 0., 0., 0.])
            obs['real_size_goal'] = np.array([0.03, 0.03, 0.02])
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