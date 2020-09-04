import os
import gym
from gym.envs.robotics import fetch_env
import numpy as np
from gym.envs.robotics import rotations, robot_env, utils
from PIL import Image
import copy
from utils.image_util import store_image_array_at

# Ensure we get the path separator correct on windows
#MODEL_XML_PATH = os.path.join('fetch', 'push_labyrinth.xml')
MODEL_XML_PATH = os.path.join('/home/erick/RL/HGG-extended/HGG-Extended-master/gym/gym/envs/robotics/assets/fetch', 'push_labyrinth4.xml')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchPushLabyrinthEnv3(robot_env.RobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type='sparse'):

        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        initial_qpos = {
            'robot0:slide0': 0.405,#position relative x (or y) from main body
            'robot0:slide1': 0.48,#position relative y (or x) from main body
            'robot0:slide2': 0.0,#position relative to axis z from main body
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],#x,y,z,_,rotation_as_quaternion(perhaps)
        }
        model_path = MODEL_XML_PATH
        n_substeps = 20
        self.further= False
        self.gripper_extra_height = 0.0
        self.block_gripper = True
        self.has_object = True
        self.target_in_the_air = False
        self.target_offset = 0.0
        self.obj_range = 0.06 # originally 0.15
        self.target_range_x = 0.06 # entire table: 0.125
        self.target_range_y = 0.06 # entire table: 0.175
        self.distance_threshold = 0.05
        self.reward_type = reward_type

        # TODO: configure adaption parameters
        self.adapt_dict=dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.5, 0.25, 0.35, 0.1]
        self.adapt_dict["obstacles"] = [[1.3, 0.75, 0.5, 0.12, 0.02, 0.1],
                                        [1.3 - 0.23, 0.75, 0.5, 0.02, 0.35, 0.1],
                                        [1.3 + 0.03, 0.75, 0.5, 0.02, 0.2, 0.1]]

        # center 1.3, 0.75, 0.2 geom size 0.25, 0.35, 0.2
        self.table_lims = [[1.3 - 0.25, 1.3 + 0.25], [0.75 - 0.35, 0.75 + 0.35]]
        self.table_lims_target = [[self.table_lims[0][0] + 0.025, self.table_lims[0][1] - 0.025],
                                  [self.table_lims[1][0] + 0.025, self.table_lims[1][1] - 0.025]]

        self.table_lims_object = [[self.table_lims[0][0] + 0.025, self.table_lims[0][1] - 0.025],
                                  [self.table_lims[1][0]+0.025, self.table_lims[1][1] - 0.025]]
        self.visible_gripper =[[0.85, 1.65], [0.38, 1.12]]
        #self.obstacles_lims = [[[1.3 - 0.12, 1.3 + 0.12], [0.75 - 0.02, 0.75 + 0.02]]]
        #self.obstacles_lims = [[[1.3 - 0.12, 1.3 + 0.12], [0.75 - 0.02, 0.75 + 0.02]]]
        self.obstacles_lims = [[[1.3 - 0.25, 1.3 + 0.25], [0.6 - 0.02, 0.6 + 0.02]],
                               [[1.05 - 0.02 , 1.05 + 0.02], [0.85 - 0.25, 0.85 + 0.25]],
                               [[1.55  - 0.02 , 1.55  + 0.02], [0.85 - 0.25, 0.85 + 0.25]]
                               ]
        #self.obstacles_lims = []
        self.target_obst = []
        self.object_obst = []
        self.gripper_obst = []
        for obstacle in self.obstacles_lims:
            l_x, l_y = obstacle
            t_x = [l_x[0]-0.02, l_x[1]+0.02]
            o_x = [l_x[0]-0.025, l_x[1]+0.025]
            g_x = [l_x[0] - 0.045, l_x[1] + 0.045]
            t_y = [l_y[0]-0.02, l_y[1]+0.02]
            o_y = [l_y[0]-0.025, l_y[1]+0.025]
            g_y = [l_y[0] - 0.065, l_y[1] + 0.065]
            tn = [t_x, t_y]
            on = [o_x, o_y]
            gn = [g_x, g_y]
            self.target_obst.append(tn)
            self.object_obst.append(on)
            self.gripper_obst.append(gn)
        self.target_init_region = [[1.25,1.35],[0.65, 0.75]]
        self.gripper_init_region = [[1.2, 1.4], [0.95, 1.0]]

        self.object_width = 0.025*2.0
        self.gripper_finger_width = 0.0385*2.0
        self.img_size = 16
        self.r_low = [1.1, 1.15]
        self.r_high = [1.45, 1.5]
        self.r_rest = [1.15, 1.45]
        self.mem_low = [1.1, 1.15]
        self.mem_high = [1.45, 1.5]
        self.prob_distr = [0.4, 0.4, 0.1, 0.05, 0.05]

        super(FetchPushLabyrinthEnv3, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)

        gym.utils.EzPickle.__init__(self)
    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info): # leave unchanged
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
    """    
    def is_inside_goal_space(self, goal):
        assert goal.shape == (3,)
        if (self.target_center[0] - self.target_range_x <= goal[0] <= self.target_center[0] + self.target_range_x) and 
            (self.target_center[1] - self.target_range_y <= goal[1] <= self.target_center[1] + self.target_range_y) and
            (self.target_center[2] )
    """
    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        # initially close gripper
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def step_obstacle(self):
        # direction is one dimensional see type = 'slide' in mujoco doc
        body_id = self.sim.model.body_name2id('keeper0')
        mov_obst_center = self.sim.data.body_xpos[body_id]
        if mov_obst_center[0] >= 1.38 and self.obstacle_direction == 1:
            self.obstacle_direction = -1
        elif mov_obst_center[0] <= 1.22 and self.obstacle_direction == -1:
            self.obstacle_direction = 1

        # extra for the obstacle
        # body_id = e.sim.model.body_name2id('obstacle_5')
        # e.sim.model.body_pos[body_id][2] = 0.9

        #self.sim.data.set_joint_qpos('keeper0:joint', obj_qpos + 0.01 * self.obstacle_direction)
        self.sim.model.body_pos[body_id][0] += self.obstacle_vel * self.obstacle_direction


    def reset_obstacle(self):
        a = np.random.randint(2)
        if a == 0:
            self.obstacle_direction = -1
        else:
            self.obstacle_direction = 1

        #[1.22;1.38]
        positions = [1.225, 1.3, 1.375]
        start_pos = positions[np.random.randint(0, 3)]
        body_id = self.sim.model.body_name2id('keeper0')
        self.sim.model.body_pos[body_id][0] = start_pos
        vels = [0.00995, 0.01]
        self.obstacle_vel = vels[np.random.randint(0, 2)]
        #obstacle_id = self.sim.model.body_name2id('keeper0')
        #dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        #v = -0.02 / self.sim.nsubsteps
        #self.sim.data.set_joint_qvel('keeper0:joint', v)

        
    def step(self, action):
        self.step_obstacle()
        return super(FetchPushLabyrinthEnv3, self).step(action)

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        #obs = np.concatenate([grip_pos, object_pos.ravel(), object_rel_pos.ravel()])

        gripper_visible = True
        x = grip_pos[0]
        y = grip_pos[1]
        if x < self.visible_gripper[0][0] or x > self.visible_gripper[0][1] or \
                y < self.visible_gripper[1][0] or y > self.visible_gripper[1][1]:
            gripper_visible = False

        object_below_table = object_pos[2] < 0.35

        body_id = self.sim.model.body_name2id('keeper0')
        mov_obst_center = self.sim.data.body_xpos[body_id]
        mov_up = mov_obst_center[:2] + np.array([0.15,0.0])#todo
        mov_down = mov_obst_center[:2] + np.array([-0.15,0.0])
        mov = np.concatenate([mov_up, mov_down])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'gripper_visible':gripper_visible,
            'object_below_table':object_below_table,
            'dyn_pos': mov.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('camlook')
        lookat = self.sim.data.body_xpos[body_id]
        # todo see how to modify to put in wanted position, with external method
        #body_id = self.sim.model.body_name2id('robot0:gripper_link')
        #lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        #self.viewer.cam.lookat[0] += 0.2
        self.viewer.cam.lookat[2] += 0.3
        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = 270#360#

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self, inter_sampling=False, in_range=True):
        self.sim.set_state(self.initial_state)
        self.reset_obstacle()
        if in_range:
            point = self.random_pos_outside_obstacle(self.gripper_obst, self.gripper_init_region)
            self.move_gripper_to(point[0], point[1])
            dx = np.random.uniform(0.05, 0.08)
            dx *= [-1, 1][np.random.randint(0,2)]
            dy = np.random.uniform(0.05, 0.08)
            dy *= -1
            self.move_object_to(point[0]+dx, point[1]+dy)
        elif inter_sampling:
            point = self.random_pos_outside_obstacle(self.gripper_obst, self.table_lims)
            self.move_gripper_to(point[0],point[1])
            if self.has_object:
                # 3 cases: near gripper (this more frequent), expected begin, random position on table
                case = self.np_random.randint(low=0, high=4)
                if case % 3 == 0:
                    object_xpos = self.object_pos_next_to_gripper()
                elif case % 3 == 1:
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,
                                                                                         self.obj_range,
                                                                                         size=2)
                else:
                    object_xpos = self.random_pos_outside_obstacle(self.object_obst, self.table_lims_object)
                self.move_object_to(object_xpos[0], object_xpos[1])
        else:
            # Randomize start position of object.
            if self.has_object:
                object_xpos = self.initial_gripper_xpos[:2]
                # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1: # TODO: next line was in loop
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
                object_qpos = self.sim.data.get_joint_qpos('object0:joint')
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        #extra for the obstacle
        #body_id = e.sim.model.body_name2id('obstacle_5')
        #e.sim.model.body_pos[body_id][2] = 0.9
        return True

    def _sample_goal(self, inter_sampling=False, in_range=True):
        if in_range:
            #self.target_init_region = [[1.25, 1.35], [0.65, 0.75]]
            ranges = [self.r_low, self.r_high, self.r_rest, self.mem_low, self.mem_high]
            idx = np.random.choice([0,1,2,3,4], p=self.prob_distr)
            x_range = ranges[idx]
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(0.65, 0.75)
            goal = np.array([x, y, 0.4])
        elif inter_sampling:
            goal = self.random_pos_outside_obstacle(self.target_obst, self.target_init_region)
        else:
            goal = self.target_center.copy()
            goal[1] += self.np_random.uniform(-self.target_range_y, self.target_range_y)
            goal[0] += self.np_random.uniform(-self.target_range_x, self.target_range_x)
        self.goal = goal.copy()
        self.sample_goal_info(self.img_size, self.img_size)
        return goal.copy()

    def set_goal_img_size(self, size):
        self.img_size = size

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # TODO: initial markers (index 3 nur zufällig, aufpassen!)
        self.target_center = self.sim.data.get_site_xpos('target_center')
        self.init_center = self.sim.data.get_site_xpos('init_center')
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()[3]

        # Move end effector into position. # TODO: changed that to the left
        #gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = self.init_center + self.gripper_extra_height #+ self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        object_xpos = self.initial_gripper_xpos
        object_xpos[2] = 0.4 # table height

        # important part
        site_id = self.sim.model.site_name2id('init_1')
        self.sim.model.site_pos[site_id] = object_xpos + [self.obj_range, self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_2')
        self.sim.model.site_pos[site_id] = object_xpos + [self.obj_range, -self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_3')
        self.sim.model.site_pos[site_id] = object_xpos + [-self.obj_range, self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_4')
        self.sim.model.site_pos[site_id] = object_xpos + [-self.obj_range, -self.obj_range, 0.0] - sites_offset


        site_id = self.sim.model.site_name2id('mark1')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, self.target_range_y, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark2')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, self.target_range_y, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark3')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, -self.target_range_y, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark4')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, -self.target_range_y, 0.0] - sites_offset


        self.sim.step()

        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchPushLabyrinthEnv3, self).render(mode, width, height)

    def chose_region(self, probs):
        random = self.np_random.uniform(0,1)
        acc = 0
        for i, p in enumerate(probs):
            acc += p
            if random < acc:
                return i
        print(acc)

    def _inside_an_obstacle(self, x, y, obstacles_lims):
        def inside(x, y, obstacle):
            if x > obstacle[0][0] and x <  obstacle[0][1] and y > obstacle[1][0] and y <  obstacle[1][1]:
                return True
            return False

        return any([inside(x,y,o) for o in obstacles_lims])

    def random_pos_outside_obstacle(self, obstacles_lims, table_lims):
        while True:
            x = self.np_random.uniform(table_lims[0][0], table_lims[0][1])
            y = self.np_random.uniform(table_lims[1][0], table_lims[1][1])
            if not self._inside_an_obstacle(x, y, obstacles_lims):
                return np.array([x, y, 0.4])

    def gripper_pos_next_object(self):
        r = np.sqrt(2.0 * np.power(self.object_width / 2.0, 2)) + np.sqrt(
            2.0 * np.power(self.gripper_finger_width / 2.0, 2))
        object_pos = self.get_object_current_state()
        x_min = max(object_pos[0] - r, self.table_lims[0][0])
        x_max = min(object_pos[0] + r, self.table_lims[0][1])
        y_min = max(object_pos[1] - r, self.table_lims[1][0])
        y_max = min(object_pos[1] + r, self.table_lims[1][1])
        while True:
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            if not self._inside_an_obstacle(x, y, self.object_obst):
                z = self.np_random.uniform(0.4, 0.5)
                return np.array([x, y, z])

    def object_pos_next_to_gripper(self):
        r = np.sqrt(2.0*np.power(self.object_width/2.0, 2)) + np.sqrt(2.0*np.power(self.gripper_finger_width/2.0, 2))
        grip_pos = self.get_gripper_pos()
        x_min = max(grip_pos[0] - r, self.table_lims_object[0][0])
        x_max = min(grip_pos[0] + r, self.table_lims_object[0][1])
        y_min = max(grip_pos[1] - r, self.table_lims_object[1][0])
        y_max = min(grip_pos[1] + r, self.table_lims_object[1][1])
        while True:
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            if not self._inside_an_obstacle(x, y, self.object_obst):
                return np.array([x, y, 0.4])

    def object_pos_at_goal(self):
        goal = self.goal.copy()
        x_min = max(goal[0] - self.distance_threshold, self.table_lims_object[0][0])
        x_max = min(goal[0] + self.distance_threshold, self.table_lims_object[0][1])
        y_min = max(goal[1] - self.distance_threshold, self.table_lims_object[1][0])
        y_max = min(goal[1] + self.distance_threshold, self.table_lims_object[1][1])
        while True:
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            if not self._inside_an_obstacle(x, y, self.object_obst):
                return np.array([x, y, 0.4])

    def move_goal_to(self, x, y, z=None):
        current = self.goal.copy()
        if z is not None:
            self.goal = np.array([x, y, z])
        else:
            self.goal = np.array([x, y, current[2]].copy())

    def set_goal(self, x,y,z=None):
        self.move_goal_to(x,y,z)
        self.sample_goal_info(self.img_size, self.img_size)


    def move_gripper_to(self, x, y, z=None):
        self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
        self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
        self.sim.forward()
        # first move upside
        object_pos = self.get_gripper_pos()
        up_pos = np.array([object_pos[0], object_pos[1], 1.2])
        self.sim.data.set_mocap_pos('robot0:mocap', up_pos)
        for _ in range(20):
            self.sim.step()
        self.sim.forward()
        # move to wanted x, y position
        gripper_target = [x, y, up_pos[2]]
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        for _ in range(20):
            self.sim.step()
        self.sim.forward()
        # move again to table height or desired height
        if z is not None:
            self.sim.data.set_mocap_pos('robot0:mocap', [x,y,z])
        else:
            self.sim.data.set_mocap_pos('robot0:mocap', [x,y, object_pos[2]])
        for _ in range(20):
            self.sim.step()
        self.sim.forward()

    def move_object_to(self, x, y, z=None):
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[0] = x
        object_qpos[1] = y
        if z is not None:
            object_qpos[2] = z
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()

    def get_object_current_state(self):
        return self.sim.data.get_joint_qpos('object0:joint').copy()

    def reset_object_to_state(self, state):
        self.sim.data.set_joint_qpos('object0:joint', state)

    def get_gripper_pos(self):
        return self.sim.data.get_site_xpos('robot0:grip').copy()

    #just call after a reset
    def sample_goal_info(self, height, width):
        #store state to reset
        original_state = copy.deepcopy(self.sim.get_state())
        # temporally moves to goal position so it is possible to store goal image
        object_goal_pos = self.object_pos_at_goal()
        self.move_object_to(object_goal_pos[0], object_goal_pos[1])
        gripper_goal_pos = self.gripper_pos_next_object()
        self.move_gripper_to(gripper_goal_pos[0], gripper_goal_pos[1])
        # close gripper
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            for _ in range(100):
                self.sim.step()
            self.sim.forward()
        self.goal_image = self.render(mode="rgb_array", width=width, height=height)
        #store_image_array_at(image_goal,
        #                     '/home/erick/RL/HGG-extended/HGG-Extended-master/logsdir/temp/', 'the_goal')
        obs = self._get_obs()
        self.goal_state = obs['observation'].copy()
        self.goal_pos = obs['achieved_goal'].copy()

        #move everything back to original state
        self.sim.set_state(original_state)
        self.sim.set_state(original_state)

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_ranges(self, low, high, rest, mem_low, mem_high, prob_distr=[0.4, 0.4, 0.1, 0.05, 0.05], step_l = 0.01):
        self.r_low = low
        self.r_high = high
        self.r_rest = rest
        self.mem_low = mem_low
        self.mem_high = mem_high
        self.prob_distr = prob_distr
        self.step_l = step_l

    def get_ranges(self):
        return copy.deepcopy(self.r_low), copy.deepcopy(self.r_high), \
               copy.deepcopy(self.r_rest), copy.deepcopy(self.mem_low), copy.deepcopy(self.mem_high), \
               copy.deepcopy(self.prob_distr), copy.deepcopy(self.step_l)

    def reset_ranges(self, mem):
        self.r_low, self.r_high, self.r_rest, self.mem_low, self.mem_high, self.prob_distr, self.step_l = mem

    def print_ranges(self):
        print('low: {}'.format(self.r_low))
        print('high: {}'.format(self.r_high))
        print('rest: {}'.format(self.r_rest))
        print('mem_low: {}'.format(self.mem_low))
        print('mem_high: {}'.format(self.mem_high))
        print('prob_distr: {}'.format(self.prob_distr))
        print('step_l: {}'.format(self.step_l))

    def set_prob_distr(self, p):
        self.prob_distr = p

    def progress_ranges(self):
        if np.abs(self.r_low[0] - self.r_high[0]) <= self.step_l:#ranges close enough that change is not necessary anymore
            pass
        else:
            self.r_high = [self.r_high[0]-self.step_l,self.r_high[1]-self.step_l]
            self.r_low = [self.r_low[0]+self.step_l, self.r_low[1]+self.step_l]
            self.r_rest = [self.r_low[1], self.r_high[0]]
            self.mem_low[1] = self.r_low[0]
            self.mem_high[0] = self.r_high[1]

'''if __name__ == '__main__':
    import time
    from utils.image_util import rgb_array_to_image
    import cv2

    e = FetchPushLabyrinthEnv3()
    im_size = 500
    e.set_goal_img_size(im_size)
    ims = []
    for episode in range(1):
        ob = e.reset()
        for step in range(500):
            array = e.render(mode='rgb_array')
            image = rgb_array_to_image(array)
            ims.append(image)
            e.step(e.action_space.sample())
            obs = e._get_obs()
    e.close()
    out = cv2.VideoWriter('/home/erick/RL/HGG-extended/HGG-Extended-master/vid.avi', cv2.VideoWriter_fourcc(*'DIVX'),
                          20, (im_size, im_size))
    for i in range(len(ims)):
        out.write(np.array(ims[i]))
    out.release()'''