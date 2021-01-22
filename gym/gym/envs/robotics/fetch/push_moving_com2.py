import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
import copy

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push_moving_com2.xml')

class FetchPushMovingComEnv2(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]

        #centers of the interval where goal and initial position will be sampld
        self.target_goal_center = np.array([1.4, 0.7, 0.425])
        self.object_center = np.array([1.1, 0.65, 0.425])

        #for moving
        self.vel_lims = [0.3, 0.5]
        self.current_obstacle_vel = 0.3
        self.obstacle_direction = 1
        #the object must be in the middle from both limits in the xml
        self.obstacle_upper_limit = 1.46
        self.obstacle_lower_limit = 1.33
        self.pos_dif = (self.obstacle_upper_limit - self.obstacle_lower_limit) / 2.


        #!!! set the position of object) :correctly
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.095, 0.68, 0.425, 1., 0., 0., 0.],  # origin 0.53
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.02,
            target_range=0.03, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.obstacle_slider_idx = self.sim.model.joint_names.index('obstacle:joint')
        self.geom_id_object = self.sim.model.geom_name2id('object0')
        self.geom_ids_obstacles = []
        for name in ['o', 'o2', 'o3']:
            self.geom_ids_obstacles.append(self.sim.model.geom_name2id(name))
        self.use_reset_sim = True

    def test_setup(self, new_vel_lims=[1., 1.2]):
        '''
        changes the parameter for further tests after training an agent
        '''
        # the default values makes the obstacle in average faster
        self.vel_lims = new_vel_lims
        self.current_obstacle_vel = new_vel_lims[1]

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
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        qpos = self.sim.data.qpos.flat[:]
        current_qpos = qpos[self.obstacle_slider_idx]

        if self.obstacle_direction == 1:
            if current_qpos >= self.pos_dif:
                new_pos = current_qpos - self.current_obstacle_vel * dt
                self.set_obstacle_slide_pos(new_pos)
                self.obstacle_direction = -1
            else:
                extra_dist = self.current_obstacle_vel * dt
                if current_qpos + extra_dist >= self.pos_dif:
                    new_pos = self.pos_dif
                    self.set_obstacle_slide_pos(new_pos)
                    self.obstacle_direction = -1
                else:
                    new_pos = current_qpos + extra_dist
                    self.set_obstacle_slide_pos(new_pos)

        else:
            if current_qpos <= -self.pos_dif:
                new_pos = current_qpos + self.current_obstacle_vel * dt
                self.set_obstacle_slide_pos(new_pos)
                self.obstacle_direction = 1
            else:
                extra_dist = self.current_obstacle_vel * dt
                if current_qpos - extra_dist <= -self.pos_dif:
                    new_pos = -self.pos_dif
                    self.set_obstacle_slide_pos(new_pos)
                    self.obstacle_direction = 1
                else:
                    new_pos = current_qpos - extra_dist
                    self.set_obstacle_slide_pos(new_pos)


    def step(self, action):
        self.move_obstacle()
        return super(FetchPushMovingComEnv2, self).step(action)


    def _sample_goal(self):
        goal = self.target_goal_center + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        return goal.copy()

    def _set_gripper_during_setup(self):
        # Move end effector into position.
        orig_pos = self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = np.array([-0.83, -0.081, -0.248 + self.gripper_extra_height]) + orig_pos
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.obstacle_direction = np.random.choice([1, -1])

        possible_vels = np.linspace(start=self.vel_lims[0], stop=self.vel_lims[1], num=20, endpoint=True)
        self.current_obstacle_vel = np.random.choice(possible_vels)


        self.set_obstacle_slide_vel(self.current_obstacle_vel * self.obstacle_direction)

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
        obs = super(FetchPushMovingComEnv2, self)._get_obs()
        body_id = self.sim.model.body_name2id('obstacle')
        pos = self.sim.data.body_xpos[body_id].copy()
        dims = np.array([0.09, 0.03, 0.025])
        o1 = np.concatenate((pos, dims.copy()))
        o2 = np.array([1.3, 0.51, 0.44, 0.3, 0.02, 0.04])
        o3 = np.array([1.185, 0.68, 0.44, 0.02, 0.15, 0.04])
        obs['real_obstacle_info'] = np.array([o1, o2, o3])
        obs['real_size_goal'] = np.array([0.04, 0.04, 0.02])
        return obs

