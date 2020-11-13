from .interval import IntervalGoalEnv
import numpy as np

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

class IntervalExt(IntervalGoalEnv):
    def __init__(self, args):
        IntervalGoalEnv.__init__(self, args)
        self.counter = 0

    def get_obs(self):
        obs = super(IntervalExt, self).get_obs()
        obstacle_len_shape = None
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
            #the first entries will always have the more recent representations
            prev_obs = self.last_obs.copy()
            begin_index = self.start_index_extra_observation
            end_index = begin_index+self.single_step_extra_goal_state_size
            prev_goal_state = prev_obs['observation'][begin_index: end_index]

            begin_index = self.start_index_extra_observation+2*self.single_step_extra_goal_state_size
            #This one extract until the end since it might exist more tah one obstacle
            end_index = begin_index + self.single_step_extra_obstacle_state_size*2
            prev_obstacle_state = prev_obs['observation'][begin_index: end_index]
            prev_obstacle_state = extract_most_current_obstacles(prev_obstacle_state)

            #the previous ones are pushed to the back
            extra_goal_state = np.concatenate([extra_goal_state, prev_goal_state])
            if obstacle_len_shape > 1:
                extra_obstacle_state = np.ravel(np.concatenate([extra_obstacle_state, prev_obstacle_state], axis=1))
            else:
                extra_obstacle_state = np.concatenate([extra_obstacle_state, prev_obstacle_state[0]])

        new_state = np.concatenate([obs['observation'], extra_goal_state, extra_obstacle_state])
        obs['observation'] = new_state

        return obs

    def step(self, action):#just here makes sense to increment step
        self.counter += 1
        ret = super(IntervalExt, self).step(action)
        return ret

    def reset_ep(self):
        self.counter = 0
        super(IntervalExt, self).reset_ep()


import matplotlib.pyplot as plt
import matplotlib.patches as patches

class IntervalColl(IntervalExt):
    def __init__(self, args):
        IntervalExt.__init__(self, args)

    def get_obs(self):
        obs = super(IntervalColl, self).get_obs()
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

class IntervalRewSub(IntervalColl):
    def __init__(self, args):
        IntervalColl.__init__(self, args)

    def compute_reward(self, observation_current, observation_old, goal):
        rew = super(IntervalRewSub, self).compute_reward(observation_current, observation_old, goal)
        if observation_current['coll'] > 0.:
            rew += -0.5#todo is -0.5 ok?; the idea is to create negative reward for collision but different so it can differentiate between negative reward for not reaching goal and negative reward for collision
        return rew

class IntervalRewVec(IntervalColl):
    def __init__(self, args):
        IntervalColl.__init__(self, args)

    def compute_reward(self, observation_current, observation_old, goal):
        rew = super(IntervalRewVec, self).compute_reward(observation_current, observation_old, goal)
        collision_value = -1. if observation_current['coll'] > 0. else 0.
        rew = np.array([rew, collision_value])
        return rew



class IntervalTestColDetRewSub(IntervalRewSub):
    def __init__(self, args):
        IntervalGoalEnv.__init__(self, args)

    def get_obs(self):
        obs = super(IntervalTestColDetRewSub, self).get_obs()
        sim = self.env.env.sim
        exists_collision = False
        #todo generalize this for other environments
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if (contact.geom1 == 23 and contact.geom2 == 24) or (contact.geom1 == 24 and contact.geom2 == 23):
                exists_collision = True

        obs['collision_check'] = exists_collision
        return obs

class IntervalTestColDetRewVec(IntervalRewVec):
    def __init__(self, args):
        IntervalGoalEnv.__init__(self, args)

    def get_obs(self):
        obs = super(IntervalTestColDetRewVec, self).get_obs()
        sim = self.env.env.sim
        exists_collision = False
        #todo generalize this for other environments
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if (contact.geom1 == 23 and contact.geom2 == 24) or (contact.geom1 == 24 and contact.geom2 == 23):
                exists_collision = True

        obs['collision_check'] = exists_collision
        return obs