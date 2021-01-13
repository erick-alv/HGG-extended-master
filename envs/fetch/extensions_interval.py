from .interval import IntervalGoalEnv
from abc import ABC, abstractmethod
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

#todo first run jsut the algorithm with the minimizer of collision along side to see what Q values it does create
#A space visualizer V value will be needed that(heat map)

env_dt = 0.02
#becomes obstacles in both steps returns the most current one as 2-dim array
def extract_most_current_obstacles(obstacles_array):
    splitted = np.array(np.split(obstacles_array, len(obstacles_array)/4))
    most_recent = splitted[0:len(splitted):2, :]
    return most_recent


#b_bboxes is expected to be 2 dim array
def check_collisions(a_bbox, b_bboxes):
    # b_min_x - a_max_x
    d1x = (b_bboxes[:, 0] - b_bboxes[:, 2]) - (a_bbox[0]+a_bbox[2])
    d1y = (b_bboxes[:, 1] - b_bboxes[:, 3]) - (a_bbox[1]+a_bbox[3])
    d2x = (a_bbox[0] - a_bbox[2]) - (b_bboxes[:, 0] + b_bboxes[:, 2])
    d2y = (a_bbox[1] - a_bbox[3]) - (b_bboxes[:, 1] + b_bboxes[:, 3])
    d1_bools = np.logical_or(d1x>0., d1y>0.)
    d2_bools = np.logical_or(d2x>0., d2y>0.)
    d_bools = np.logical_or(d1_bools, d2_bools)
    return np.logical_not(d_bools)


def aabbs_max_distances(a_bbox, b_bboxes):
    dcxs = np.abs(b_bboxes[:, 0] - a_bbox[0])
    extra_x_dist = b_bboxes[:, ] + a_bbox[2]
    x_max_dists = dcxs + extra_x_dist

    dcys = np.abs(b_bboxes[:, 1] - a_bbox[1])
    extra_y_dist = b_bboxes[:, 3] + a_bbox[3]
    y_max_dists = dcys + extra_y_dist
    d_maxs = np.sqrt(x_max_dists**2 + y_max_dists**2)
    return d_maxs


def aabbs_min_distances(a_bbox, b_bboxes):
    dcxs = np.abs(b_bboxes[:, 0] - a_bbox[0])
    extra_x_dist = b_bboxes[:, 2] + a_bbox[2]
    zeros_array = np.zeros(shape=extra_x_dist.shape)
    x_min_dists = np.maximum(dcxs - extra_x_dist, zeros_array)

    dcys = np.abs(b_bboxes[:, 1] - a_bbox[1])
    extra_y_dist = b_bboxes[:, 3] + a_bbox[3]
    y_min_dists = np.maximum(dcys - extra_y_dist, zeros_array)
    d_mins = np.sqrt(x_min_dists**2 + y_min_dists**2)
    return d_mins


def calc_vels(bboxes, bboxes_prev, dt):
    pos_dif = bboxes[:, 0:2] - bboxes_prev[:, 0:2]
    vel = pos_dif / dt
    return vel


def calc_angles(a_bbox, b_bboxes):
    if a_bbox[0] == 100. and a_bbox[1] == 100.:
        # use a negative angle so it is different in this case
        angles = np.repeat(-1., repeats=b_bboxes.shape[0])
    else:
        angles = np.arctan2(b_bboxes[:, 1] - a_bbox[1], b_bboxes[:, 0] - a_bbox[0]) * 180 / np.pi  # to degree
        angles = angles % 360.
    angles = np.expand_dims(angles, axis=1)
    return angles


class ObsExtender(ABC):
    def __init__(self, args):
        self.args = args
        self.counter = 0

    @abstractmethod
    def extend_obs(self, obs, env):
        pass


    def step(self):#just here makes sense to increment step
        self.counter += 1

    def reset_ep(self):
        self.counter = 0


#leaves everything as it is, used for test of HGG
class DummyExtender(ObsExtender):
    def __init__(self, args):
        super(DummyExtender, self).__init__(args)

    def extend_obs(self, obs, env):
        return obs



class ObsExtenderBbox(ObsExtender):
    def __init__(self, args):
        super(ObsExtenderBbox, self).__init__(args)

    def extend_obs(self, obs, env):
        if self.args.vae_dist_help:
            extra_goal_state = np.concatenate([obs['achieved_goal_latent'],
                                               obs['achieved_goal_size_latent']])

            obstacle_l = obs['obstacle_latent']
            obstacle_s_l = obs['obstacle_size_latent']
            obstacle_len_shape = len(obstacle_l.shape)
            if len(obstacle_l.shape) > 1:

                extra_obstacle_state = np.concatenate([obstacle_l, obstacle_s_l], axis=1)
            else:
                extra_obstacle_state = np.concatenate([obstacle_l, obstacle_s_l])
        else:
            extra_goal_state = np.concatenate([obs['achieved_goal'][:2],
                                               obs['real_size_goal'][:2]])

            obstacle_info = obs['real_obstacle_info']
            obstacle_len_shape = len(obstacle_info.shape)
            if len(obstacle_info.shape) > 1:

                extra_obstacle_state = np.concatenate([obstacle_info[:, :2], obstacle_info[:, -3:-1]], axis=1)
            else:
                extra_obstacle_state = np.concatenate([obstacle_info[:2], obstacle_info[-3:-1]])

        if self.counter == 0:

            extra_goal_state = np.concatenate([extra_goal_state, extra_goal_state])
            if obstacle_len_shape > 1:
                extra_obstacle_state = np.ravel(np.concatenate([extra_obstacle_state, extra_obstacle_state],
                                                               axis=1)
                                                )
            else:
                extra_obstacle_state = np.concatenate([extra_obstacle_state, extra_obstacle_state])
            self.single_step_extra_goal_state_size = len(extra_goal_state) // 2
            self.single_step_extra_obstacle_state_size = len(extra_obstacle_state) // 2
            self.start_index_extra_observation = len(obs['observation'])
        else:
            # the first entries will always have the more recent representations
            prev_obs = env.last_obs.copy()
            begin_index = self.start_index_extra_observation
            end_index = begin_index + self.single_step_extra_goal_state_size
            prev_goal_state = prev_obs['observation'][begin_index: end_index]

            begin_index = self.start_index_extra_observation + 2 * self.single_step_extra_goal_state_size
            # This one extract until the end since it might exist more tah one obstacle
            end_index = begin_index + self.single_step_extra_obstacle_state_size * 2
            prev_obstacle_state = prev_obs['observation'][begin_index: end_index]
            prev_obstacle_state = extract_most_current_obstacles(prev_obstacle_state)

            # the previous ones are pushed to the back
            extra_goal_state = np.concatenate([extra_goal_state, prev_goal_state])
            if obstacle_len_shape > 1:
                extra_obstacle_state = np.ravel(np.concatenate([extra_obstacle_state, prev_obstacle_state], axis=1))
            else:
                extra_obstacle_state = np.concatenate([extra_obstacle_state, prev_obstacle_state[0]])

        new_state = np.concatenate([obs['observation'], extra_goal_state, extra_obstacle_state])
        obs['observation'] = new_state

        return obs




#basically the same but does not extend the state that is passed to agent. This class will be inherited to extend the
#class in other ways
class ObsExtBboxInfo(ObsExtender):
    def __init__(self, args):
        super(ObsExtBboxInfo, self).__init__(args)

    def extend_obs(self, obs, env):
        if self.args.vae_dist_help:
            extra_goal_state = np.concatenate([obs['achieved_goal_latent'],
                                               obs['achieved_goal_size_latent']])

            obstacle_l = obs['obstacle_latent']
            obstacle_s_l = obs['obstacle_size_latent']
            obstacle_len_shape = len(obstacle_l.shape)
            if len(obstacle_l.shape) > 1:

                extra_obstacle_state = np.concatenate([obstacle_l, obstacle_s_l], axis=1)
            else:
                extra_obstacle_state = np.concatenate([obstacle_l, obstacle_s_l])
                extra_obstacle_state = np.expand_dims(extra_obstacle_state, axis=0)
        else:
            extra_goal_state = np.concatenate([obs['achieved_goal'][:2],
                                               obs['real_size_goal'][:2]])

            obstacle_info = obs['real_obstacle_info']
            if len(obstacle_info.shape) > 1:

                extra_obstacle_state = np.concatenate([obstacle_info[:, :2], obstacle_info[:, -3:-1]], axis=1)
            else:
                extra_obstacle_state = np.concatenate([obstacle_info[:2], obstacle_info[-3:-1]])
                extra_obstacle_state = np.expand_dims(extra_obstacle_state, axis=0)

        if self.counter == 0:
            #It is the first observation. We cannot assume nothing from previous steps and therefore init every
            #field with the same

            #might not need to store all goal state since they will not be used
            obs['goal_st_t'] = extra_goal_state.copy()
            obs['goal_st_t_minus1'] = extra_goal_state.copy()
            obs['goal_st_t_minus2'] = extra_goal_state.copy()

            obs['obstacle_st_t'] = extra_obstacle_state.copy()
            obs['obstacle_st_t_minus1'] = extra_obstacle_state.copy()
            obs['obstacle_st_t_minus2'] = extra_obstacle_state.copy()

        else:
            # the previous ones are pushed to the back
            prev_obs = env.last_obs.copy()
            obs['goal_st_t'] = extra_goal_state.copy()
            obs['goal_st_t_minus1'] = prev_obs['goal_st_t'].copy()
            obs['goal_st_t_minus2'] = prev_obs['goal_st_t_minus1'].copy()

            obs['obstacle_st_t'] = extra_obstacle_state.copy()
            obs['obstacle_st_t_minus1'] = prev_obs['obstacle_st_t'].copy()
            obs['obstacle_st_t_minus2'] = prev_obs['obstacle_st_t_minus2'].copy()

        return obs

    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = copy.deepcopy(obs)
        new_obs['obstacle_st_t'] = new_obstacle_list.copy()
        new_obs['obstacle_st_t_minus1'] = None
        new_obs['obstacle_st_t_minus1'] = None
        return new_obs


class ObsExtBboxColl(ObsExtBboxInfo):
    def __init__(self, args):
        super(ObsExtBboxColl, self).__init__(args)

    def extend_obs(self, obs, env):
        obs = ObsExtBboxInfo.extend_obs(self, obs, env)
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']
        # goal object is not in visible range
        if goal_bbox[0] == 100. and goal_bbox[1] == 100.:
            obs['coll'] = 0.
            obs['coll_bool_ar'] = np.repeat(False, len(obstacle_bboxes))
        else:

            cols = check_collisions(goal_bbox, obstacle_bboxes)
            obs['coll_bool_ar'] = cols.copy()
            ncols = np.sum(cols.astype(np.float))
            obs['coll'] = ncols
        return obs
    
    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = super(ObsExtBboxColl, self)._modify_obs(obs, new_obstacle_list, extra_info, index)
        goal_bbox = new_obs['goal_st_t']
        obstacle_bboxes = new_obs['obstacle_st_t']
        # goal object is not in visible range
        if goal_bbox[0] == 100. and goal_bbox[1] == 100.:
            new_obs['coll'] = 0.
            new_obs['coll_bool_ar'] = np.repeat(False, len(obstacle_bboxes))
        else:

            cols = check_collisions(goal_bbox, obstacle_bboxes)
            new_obs['coll_bool_ar'] = cols.copy()
            ncols = np.sum(cols.astype(np.float))
            new_obs['coll'] = ncols
        return new_obs


class ObsExtMinDist(ObsExtBboxColl):
    def __init__(self, args):
        super(ObsExtMinDist, self).__init__(args)

    def extend_obs(self, obs, env):
        obs = super(ObsExtMinDist, self).extend_obs(obs, env)
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']
        # goal object is not in visible range therefore distance really far away
        if goal_bbox[0] == 100. and goal_bbox[1] == 100.:
            dists = np.repeat(1000., repeats=obstacle_bboxes.shape[0])
        else:
            dists = aabbs_min_distances(goal_bbox, obstacle_bboxes)

        obs['dists'] = dists.copy()
        new_state = np.concatenate([obs['observation'], dists.copy()])
        obs['observation'] = new_state
        return obs
    
    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = super(ObsExtMinDist, self)._modify_obs(obs, new_obstacle_list, extra_info, index)
        goal_bbox = new_obs['goal_st_t']
        obstacle_bboxes = new_obs['obstacle_st_t']
        # goal object is not in visible range therefore distance really far away
        if goal_bbox[0] == 100. and goal_bbox[1] == 100.:
            dists = np.repeat(1000., repeats=obstacle_bboxes.shape[0])
        else:
            dists = aabbs_min_distances(goal_bbox, obstacle_bboxes)

        new_obs['dists'] = dists.copy()
        len_dists = len(dists)
        new_obs['observation'][-len_dists:] = dists
        return new_obs


class ObsExtP(ObsExtMinDist):
    def __init__(self, args):
        super(ObsExtP, self).__init__(args)

    def extend_obs(self, obs, env):
        obs = super(ObsExtP, self).extend_obs(obs, env)
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']

        dists = obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]


        observation_without_dist = obs['observation'][:-len_dists]
        extension = np.concatenate([dists, pos], axis=1)
        new_state = np.concatenate([observation_without_dist, np.ravel(extension)])
        obs['observation'] = new_state
        return obs
    
    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = super(ObsExtP, self)._modify_obs(obs, new_obstacle_list, extra_info, index)
        goal_bbox = new_obs['goal_st_t']
        obstacle_bboxes = new_obs['obstacle_st_t']

        dists = new_obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        extension = np.ravel(np.concatenate([dists, pos], axis=1))
        len_extension = len(extension)
        new_obs['observation'][-len_extension:] = extension
        return new_obs
      


class ObsExtPAV(ObsExtMinDist):
    def __init__(self, args):
        super(ObsExtPAV, self).__init__(args)
        self.length_extension = None
        self.env_dt = None

    def extend_obs(self, obs, env):
        if self.env_dt is None:
            self.env_dt = env.env.env.dt
        obs = super(ObsExtPAV, self).extend_obs(obs, env)
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']
        previous_obstacle_bboxes = obs['obstacle_st_t_minus1']

        dists = obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        dt = env.env.dt
        vel = calc_vels(obstacle_bboxes, previous_obstacle_bboxes, dt)
        angles = calc_angles(goal_bbox, obstacle_bboxes)

        observation_without_dist = obs['observation'][:-len_dists]
        extension = np.ravel(np.concatenate([dists, pos, angles, vel], axis=1))
        if self.length_extension is None:
            self.length_extension = len(extension)
        new_state = np.concatenate([observation_without_dist, extension])
        obs['observation'] = new_state
        return obs
    
    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = super(ObsExtPAV, self)._modify_obs(obs, new_obstacle_list, extra_info, index)
        goal_bbox = new_obs['goal_st_t']
        obstacle_bboxes = new_obs['obstacle_st_t']

        dists = new_obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        len_pos_el = len(pos[0])
        #vel = calc_vels(obstacle_bboxes, previous_obstacle_bboxes, dt)
        angles = calc_angles(goal_bbox, obstacle_bboxes)

        mock_extension = np.ravel(np.concatenate([dists, pos, angles, pos], axis=1))
        len_extension = len(mock_extension)
        unmodified_extension = obs['observation'][-len_extension:].copy()
        unmodified_extension =np.reshape(unmodified_extension, newshape=(-1, 1+2*len_pos_el+1))
        vel = unmodified_extension[:, -2:]

        dir_not_scaled = extra_info['dir_not_scaled']
        if self.env_dt is None:
            raise Exception('this was called before modification of obs')

        vel[index] = dir_not_scaled / self.env_dt

        extension = np.ravel(np.concatenate([dists, pos, angles, vel], axis=1))
        len_extension = len(extension)
        new_obs['observation'][-len_extension:] = extension

        return new_obs

    def visualize(self, obs, file_name):
        extension = obs['observation'][- self.length_extension:].copy()
        extension = np.reshape(extension, (-1, 6))
        dists = extension[:, 0:1]
        pos = extension[:, 1:3]
        angles = extension[:, 3:4]
        vel = extension[:, 4:6]
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']
        fig, ax = plt.subplots()
        if True:
            c_x, c_y = self.args.field_center[0], self.args.field_center[1]
            d_x, d_y = self.args.field_size[0], self.args.field_size[1]
            support_points = np.array([[c_x - d_x, c_y - d_y],
                                       [c_x - d_x, c_y + d_y],
                                       [c_x + d_x, c_y - d_y],
                                       [c_x + d_x, c_y + d_y]])
            ax.scatter(support_points[:, 0], support_points[:, 1], c='black')
        ax.scatter(pos[:, 0], pos[:, 1], c='blue')

        ax.scatter(pos[:, 0], pos[:, 1], s=dists[:, 0], c='blue', alpha=0.8)
        ax.scatter([goal_bbox[0]], [goal_bbox[1]], c='red')
        ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1])

        def plot_point_angle(x, y, a, length, color):
            # find the end point
            endy = y + length * math.sin(math.radians(a))
            endx = x + length * math.cos(math.radians(a))
            # plot the points
            ax.plot([x, endx], [y, endy], '--', c=color)


        colors_extended_area = ['yellow', 'green', 'orange']
        if True:
            for i in range(len(obstacle_bboxes)):
                bb = obstacle_bboxes[i]
                #half of extended area
                ax.add_patch(
                    patches.Rectangle(
                        (bb[0] - bb[2], bb[1] - bb[3]),
                        bb[2] * 2,
                        bb[3] * 2,
                        edgecolor='blue',
                        facecolor='blue',
                        fill=True,
                        alpha = 0.6
                    ))
                #extended area
                e_h = dists[i]
                ax.add_patch(
                    patches.Rectangle(
                        (goal_bbox[0] - goal_bbox[2] - e_h, goal_bbox[1] - goal_bbox[3] - e_h),
                        (goal_bbox[2]+e_h) * 2,
                        (goal_bbox[3]+e_h) * 2,
                        edgecolor=colors_extended_area[i],
                        facecolor=colors_extended_area[i],
                        fill=True,
                        alpha=0.2
                    ))
                #todo visualization for corner cases
                l = np.linalg.norm(pos[i] - goal_bbox[:2])
                plot_point_angle(goal_bbox[0], goal_bbox[1], angles[i], l, colors_extended_area[i])

            #goal
            ax.add_patch(
                patches.Rectangle(
                    (goal_bbox[0] - goal_bbox[2], goal_bbox[1] - goal_bbox[3]),
                    goal_bbox[2] * 2,goal_bbox[3] * 2,
                    edgecolor='red',facecolor='red',
                    fill=True,
                    alpha=0.5
                ))
        plt.savefig(file_name)
        plt.close()


#calculates position realitve to goal object
class ObsExtPRel(ObsExtMinDist):
    def __init__(self, args):
        super(ObsExtPRel, self).__init__(args)

    def extend_obs(self, obs, env):
        obs = super(ObsExtPRel, self).extend_obs(obs, env)
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']

        dists = obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        # here transformation to relative
        pos = pos - goal_bbox[0:2]

        observation_without_dist = obs['observation'][:-len_dists]
        extension = np.concatenate([dists, pos], axis=1)
        new_state = np.concatenate([observation_without_dist, np.ravel(extension)])
        obs['observation'] = new_state
        return obs

    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = super(ObsExtPRel, self)._modify_obs(obs, new_obstacle_list, extra_info, index)
        goal_bbox = new_obs['goal_st_t']
        obstacle_bboxes = new_obs['obstacle_st_t']

        dists = new_obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        # here transformation to relative
        pos = pos - goal_bbox[0:2]
        extension = np.concatenate([dists, pos], axis=1)
        len_extension = len(extension)
        new_obs['observation'][-len_extension:] = extension
        return new_obs


#calculates position realitve to goal object
class ObsExtPAVRel(ObsExtMinDist):
    def __init__(self, args):
        super(ObsExtPAVRel, self).__init__(args)

    def extend_obs(self, obs, env):
        obs = super(ObsExtPAVRel, self).extend_obs(obs, env)
        goal_bbox = obs['goal_st_t']
        obstacle_bboxes = obs['obstacle_st_t']
        previous_obstacle_bboxes = obs['obstacle_st_t_minus1']

        dists = obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        # here transformation to relative
        pos = pos - goal_bbox[0:2]

        dt = env.env.dt
        vel = calc_vels(obstacle_bboxes, previous_obstacle_bboxes, dt)
        angles = calc_angles(goal_bbox, obstacle_bboxes)

        observation_without_dist = obs['observation'][:-len_dists]
        extension = np.concatenate([dists, pos, angles, vel], axis=1)
        new_state = np.concatenate([observation_without_dist, np.ravel(extension)])
        obs['observation'] = new_state
        return obs

    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = super(ObsExtPAVRel, self)._modify_obs(obs, new_obstacle_list, extra_info, index)
        goal_bbox = new_obs['goal_st_t']
        obstacle_bboxes = new_obs['obstacle_st_t']

        dists = new_obs['dists'].copy()
        len_dists = len(dists)
        dists = np.expand_dims(dists, axis=1)
        pos = obstacle_bboxes[:, 0:2]
        pos = pos - goal_bbox[0:2]
        len_pos_el = len(pos[0])
        #vel = calc_vels(obstacle_bboxes, previous_obstacle_bboxes, dt)
        angles = calc_angles(goal_bbox, obstacle_bboxes)

        mock_extension = np.ravel(np.concatenate([dists, pos, angles, pos], axis=1))
        len_extension = len(mock_extension)
        unmodified_extension = obs['observation'][-len_extension:].copy()
        unmodified_extension =np.reshape(unmodified_extension, newshape=(-1, 1+2*len_pos_el+1))
        vel = unmodified_extension[:, -2:]

        dir_not_scaled = extra_info['dir_not_scaled']
        vel[index] = dir_not_scaled / env_dt

        extension = np.ravel(np.concatenate([dists, pos, angles, vel], axis=1))
        len_extension = len(extension)
        new_obs['observation'][-len_extension:] = extension

        return new_obs


#does not change observation state, but observation dict
class ObsExtenderBboxAndColl(ObsExtenderBbox):
    def __init__(self, args):
        ObsExtenderBbox.__init__(self, args)

    def extend_obs(self, obs, env):
        obs = super(ObsExtenderBboxAndColl, self).extend_obs(obs, env)
        begin_index = self.start_index_extra_observation
        end_index = begin_index + self.single_step_extra_goal_state_size
        goal_bbox = obs['observation'][begin_index: end_index]
        #goal object is not in visible range
        if goal_bbox[0] == 100. and goal_bbox[1] == 100.:
            obs['coll'] = 0.
        else:

            begin_index = self.start_index_extra_observation + 2 * self.single_step_extra_goal_state_size
            # This one extract until the end since it might exist more tah one obstacle
            end_index = begin_index + self.single_step_extra_obstacle_state_size * 2
            obstacle_bboxes = obs['observation'][begin_index: end_index]
            obstacle_bboxes = extract_most_current_obstacles(obstacle_bboxes)

            '''fig, ax = plt.subplots()

            ax.plot([-1., 1], [-1, 1.])
            ax.add_patch(
                patches.Rectangle(
                    (goal_bbox[0]-goal_bbox[2], goal_bbox[1]-goal_bbox[2]),
                    goal_bbox[2]*2,
                    goal_bbox[3]*2,
                    edgecolor='red',
                    facecolor='red',
                    fill=True
                ))

            for bb in obstacle_bboxes:
                ax.add_patch(
                    patches.Rectangle(
                        (bb[0]-bb[2], bb[1]-bb[3]),
                        bb[2]*2,
                        bb[3]*2,
                        edgecolor='blue',
                        facecolor='blue',
                        fill=True
                    ))

            plt.show()'''

            cols = check_collisions(goal_bbox, obstacle_bboxes)
            ncols = np.sum(cols.astype(np.float))
            obs['coll'] = ncols
        return obs



#does not change observation state, but observation dict
class ObsExtenderBboxAndCollRegion(ObsExtenderBbox):
    def __init__(self, args):
        ObsExtenderBbox.__init__(self, args)
        self.region_obstacles_bboxes = np.array(args.dist_estimator.obstacles)

    def extend_obs(self, obs, env):
        obs = super(ObsExtenderBboxAndCollRegion, self).extend_obs(obs, env)
        begin_index = self.start_index_extra_observation
        end_index = begin_index + self.single_step_extra_goal_state_size
        goal_bbox = obs['observation'][begin_index: end_index]
        #goal object is not in visible range
        if goal_bbox[0] == 100. and goal_bbox[1] == 100.:
            obs['coll'] = 0.
        else:
            cols = check_collisions(goal_bbox, self.region_obstacles_bboxes)
            ncols = np.sum(cols.astype(np.float))
            obs['coll'] = ncols
        return obs

#all these extend do not change observation state, but observation dict
class OnColl(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def extend_obs(self, obs, env):
        pass

    @abstractmethod
    def compute_reward(self, observation_current, observation_old, goal, env_reward):
        pass


class OnCollRewSub(OnColl):
    def __init__(self, args):
        OnColl.__init__(self, args)

    def extend_obs(self, obs, env):
        return obs

    def compute_reward(self, observation_current, observation_old, goal, env_reward):
        rew = env_reward
        if observation_current['coll'] > 0.:
            rew += -1.0
        return rew


class OnCollRewVec(OnColl):
    def __init__(self, args):
        OnColl.__init__(self, args)

    def extend_obs(self, obs, env):
        return obs

    def compute_reward(self, observation_current, observation_old, goal, env_reward):
        rew = env_reward
        collision_value = -1 if observation_current['coll'] > 0. else 0.
        rew = np.array([rew, collision_value])
        return rew


class OnCollStop(OnColl):
    def __init__(self, args):
        OnColl.__init__(self, args)

    def extend_obs(self, obs, env):
        if obs['coll'] > 0.:
            obs['coll_stop'] = True
        else:
            obs['coll_stop'] = False
        return obs

    def compute_reward(self, observation_current, observation_old, goal, env_reward):
        rew = env_reward
        if observation_current['coll_stop']:
            rew = -1.
        return rew


class OnCollRewMod(OnColl):
    def __init__(self, args):
        OnColl.__init__(self, args)

    def extend_obs(self, obs, env):
        return obs

    def compute_reward(self, observation_current, observation_old, goal, env_reward):
        rew = env_reward
        if observation_current['coll'] > 0.:
            #per default is  -2.0; but leave it this way to experiment with other values
            rew = self.args.rew_mod_val
        return rew


class OnCollStopRewMod(OnCollStop, OnCollRewMod):
    def __init__(self, args):
        super(OnCollStopRewMod, self).__init__(args)

    def extend_obs(self, obs, env):
        return OnCollStop.extend_obs(self, obs, env)

    def compute_reward(self, observation_current, observation_old, goal, env_reward):
        return OnCollRewMod.compute_reward(self, observation_current, observation_old, goal, env_reward)


class TestColl():  # this can be used as well for IntervalSelfCollStop, intervalEnvCollStop, IntervalExt, IntervalColl and interval
    def __init__(self, args):
        pass

    def extend_obs(self, obs, env):
        sim = env.env.sim
        exists_collision = False
        object_id = env.env.env.geom_id_object
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]

            for obstacle_id in env.env.env.geom_ids_obstacles:
                if (contact.geom1 == object_id and contact.geom2 == obstacle_id) or \
                        (contact.geom1 == obstacle_id and contact.geom2 == object_id):
                    exists_collision = True
        obs['collision_check'] = exists_collision
        return obs


#class that combines the compostion
class IntervalWithExtensions(IntervalGoalEnv):
    def __init__(self, args, obs_extender, on_coll_extender=None, test_extender=None):
        #it is important to create these before since some init methods from classes already use reset and other methods
        self.obs_extender = obs_extender
        assert isinstance(self.obs_extender, ObsExtender)
        self.test_extender = test_extender
        self.on_coll_extender = on_coll_extender
        IntervalGoalEnv.__init__(self, args)

    def get_obs(self):
        obs = super(IntervalGoalEnv, self).get_obs()
        obs = self.obs_extender.extend_obs(obs, self)
        if self.test_extender is not None:
            obs = self.test_extender.extend_obs(obs, self)
        if self.on_coll_extender is not None:
            obs = self.on_coll_extender.extend_obs(obs, self)
        return obs

    def step(self, action):
        self.obs_extender.step()
        ret = super(IntervalWithExtensions, self).step(action)
        return ret

    def reset_ep(self):
        self.obs_extender.reset_ep()
        super(IntervalWithExtensions, self).reset_ep()

    def compute_reward(self, observation_current, observation_old, goal):
        rew = super(IntervalWithExtensions, self).compute_reward(observation_current, observation_old, goal)
        if self.on_coll_extender is not None:
            rew = self.on_coll_extender.compute_reward(observation_current, observation_old, goal, rew)
        return rew

    #this should be called just by the imaginary replay_buffer
    def _modify_obs(self, obs, new_obstacle_list, extra_info, index):
        new_obs = self.obs_extender._modify_obs(obs, new_obstacle_list, extra_info, index)
        return new_obs

    def visualize(self, obs, file_name):
        self.obs_extender.visualize(obs, file_name)


class IntervalExt(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBbox(args))


class IntervalColl(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndColl(args))


class IntervalCollRegion(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndCollRegion(args))


class IntervalCollStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndColl(args),
                                       on_coll_extender=OnCollStop(args))


class IntervalCollStopRegion(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndCollRegion(args),
                                        on_coll_extender=OnCollStop(args))


class IntervalRewMod(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndColl(args),
                                        on_coll_extender=OnCollRewMod(args))

class IntervalRewModRegion(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndCollRegion(args),
                                        on_coll_extender=OnCollRewMod(args))


class IntervalRewModStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndColl(args),
                                        on_coll_extender=OnCollStopRewMod(args))


class IntervalRewModRegionStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndCollRegion(args),
                                        on_coll_extender=OnCollStopRewMod(args))


class IntervalRewSub(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBboxAndColl(args),
                                        on_coll_extender=OnCollRewSub(args))


class IntervalRewVec(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args,
                                        obs_extender=ObsExtenderBboxAndColl(args),
                                        on_coll_extender=OnCollRewVec(args))


class IntervalTest(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=DummyExtender(args),
                                        test_extender=TestColl(args))

#with this test every class with a obs extender that inherits ObsExtenderBbox and do not change more observation state
class IntervalTestExtendedBbox(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBbox(args),
                                        test_extender=TestColl(args))


#Using extension of Min dist
class IntervalCollMinDist(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtMinDist(args))

class IntervalMinDistRewMod(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtMinDist(args),
                                        on_coll_extender=OnCollRewMod(args))


class IntervalMinDistRewModStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtMinDist(args),
                                        on_coll_extender=OnCollStopRewMod(args))

#with this test every class with a obs extender that inherits ObsExtMinDist and do not change more observation state
class IntervalTestExtendedMinDist(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtMinDist(args),
                                        test_extender=TestColl(args))

class IntervalP(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtP(args))

class IntervalPRewMod(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtP(args),
                                        on_coll_extender=OnCollRewMod(args))

class IntervalPRewModStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtP(args),
                                        on_coll_extender=OnCollStopRewMod(args))

class IntervalTestExtendedP(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtP(args),
                                        test_extender=TestColl(args))


class IntervalPAV(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAV(args))

class IntervalPAVRewMod(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAV(args),
                                        on_coll_extender=OnCollRewMod(args))

class IntervalPAVRewModStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAV(args),
                                        on_coll_extender=OnCollStopRewMod(args))

class IntervalTestExtendedPAV(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAV(args),
                                        test_extender=TestColl(args))



class IntervalPRel(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPRel(args))


class IntervalPRelRewMod(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPRel(args),
                                        on_coll_extender=OnCollRewMod(args))


class IntervalPRelRewModStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPRel(args),
                                        on_coll_extender=OnCollStopRewMod(args))

class IntervalTestExtendedPRel(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPRel(args),
                                        test_extender=TestColl(args))


class IntervalPAVRel(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAVRel(args))

class IntervalPAVRelRewMod(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAVRel(args),
                                        on_coll_extender=OnCollRewMod(args))

class IntervalPAVRelRewModStop(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAVRel(args),
                                        on_coll_extender=OnCollStopRewMod(args))

class IntervalTestExtendedPAVRel(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtPAVRel(args),
                                        test_extender=TestColl(args))

