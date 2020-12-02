from .interval import IntervalGoalEnv
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#todo first run jsut the algorithm with the minimizer of collision along side to see what Q values it does create
#A space visualizer V value will be needed that(heat map)


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
            rew = -2.0
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
        # todo generalize this for other environments
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if (contact.geom1 == 23 and contact.geom2 == 24) or (contact.geom1 == 24 and contact.geom2 == 23):
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

#with this test every class with a obs extender that inherits ObsExtenderBbox and do not change more observation state
class IntervalTestExtendedBbox(IntervalWithExtensions):
    def __init__(self, args):
        IntervalWithExtensions.__init__(self, args, obs_extender=ObsExtenderBbox(args),
                                        test_extender=TestColl(args))