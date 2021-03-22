import os
import numpy as np
import copy

from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide_moving_obstacle.xml')


class FetchSlideMovingObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.11, 0.75, 0.422, 1., 0., 0., 0.],
        }
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]
        # centers of the interval where goal and initial position will be sampled
        self.target_goal_center = np.array([[1.65, 1.12, 0.42], [1.65, 0.38, 0.42]])
        self.object_center = np.array([1.11, 0.75, 0.422])
        # for moving
        #self.vel_lims = [0.6, 0.77]
        #self.current_obstacle_vel = 0.6
        self.vel_lims = [0.9, 1.2]  # 0.6
        self.current_obstacle_vel = 0.9
        self.obstacle_direction = 1
        # the object must be in the middle from both limits in the xml
        self.obstacle_upper_limit = 1.02
        self.obstacle_lower_limit = 0.48
        self.pos_dif = (self.obstacle_upper_limit - self.obstacle_lower_limit) / 2.
        #
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=0.0,
            obj_range=np.array([0.02, 0.05]),#todo axes are inverted
            target_range=np.array([0.08, 0.05]), distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.obstacle1_slider_idx = self.sim.model.joint_names.index('obstacle:joint')
        self.geom_id_object = self.sim.model.geom_name2id('object0')
        self.geom_ids_obstacles = []
        self.use_reset_sim = True
        for name in ['o']:
            self.geom_ids_obstacles.append(self.sim.model.geom_name2id(name))

    def test_setup(self, new_vel_lims=[1., 1.3]):
        '''
        changes the parameter for further tests after training an agent
        '''
        # the default values makes the obstacle in average faster
        self.vel_lims = new_vel_lims
        self.current_obstacle_vel = new_vel_lims[1]

    def set_obstacle_slide_pos(self, pos):
        # move obstacle
        qpos = self.sim.data.qpos.flat[:]
        qpos[self.obstacle1_slider_idx] = pos
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qpos=qpos)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def set_obstacle_slide_vel(self, vel):
        qvel = self.sim.data.qvel.flat[:]
        qvel[self.obstacle1_slider_idx] = vel
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qvel=qvel)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def move_obstacle(self):
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        qpos = self.sim.data.qpos.flat[:]
        current_qpos = qpos[self.obstacle1_slider_idx]

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
        orig_pos = self.sim.data.get_site_xpos('robot0:grip')
        self.move_obstacle()
        return super(FetchSlideMovingObstacleEnv, self).step(action)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        self.obstacle_direction = np.random.choice([1, -1])

        possible_vels = np.linspace(start=self.vel_lims[0], stop=self.vel_lims[1], num=30, endpoint=True)
        self.current_obstacle_vel = np.random.choice(possible_vels)

        self.set_obstacle_slide_vel(self.current_obstacle_vel * self.obstacle_direction)

        obj_offset = self.np_random.uniform(-self.obj_range,self.obj_range)

        object_xpos = self.object_center[:2] + obj_offset
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[3:] = np.array([1., 0., 0., 0.])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        return True

    def _get_obs(self):
        obs = super(FetchSlideMovingObstacleEnv, self)._get_obs()
        body_id = self.sim.model.body_name2id('obstacle')
        pos1 = np.array(self.sim.data.body_xpos[body_id].copy())
        dims = np.array([0.04, 0.17, 0.1])
        ob1 = np.concatenate((pos1, dims.copy()))
        obs['real_obstacle_info'] = np.array([ob1])
        obs['real_size_goal'] = np.array([0.055, 0.055, 0.02])
        return obs




