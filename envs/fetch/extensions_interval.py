from .interval import IntervalGoalEnv
import numpy as np

#todo first run jsut the algorithm with the minimizer of collision along side to see what Q values it does create
#A space visualizer V value will be needed that(heat map)

class IntervalExt(IntervalGoalEnv):
    def __init__(self, args):
        IntervalGoalEnv.__init__(self, args)
        self.counter = 0

    def get_obs(self):
        obs = super(IntervalExt, self).get_obs()
        if self.args.vae_dist_help:
            extra_goal_state = np.concatenate([obs['achieved_goal_latent'],
                                               obs['achieved_goal_size_latent']])

            obstacle_l = obs['obstacle_latent']
            obstacle_s_l = obs['obstacle_size_latent']
            if len(obstacle_l.shape) > 1:
                extra_obstacle_state = np.ravel(np.concatenate([obstacle_l, obstacle_s_l], axis=1))
            else:
                extra_obstacle_state = np.concatenate([obstacle_l, obstacle_s_l])
        else:
            extra_goal_state = np.concatenate([obs['achieved_goal'],
                                               obs['real_size_goal']])

            obstacle_info = obs['real_obstacle_info']
            if len(obstacle_info.shape) > 1:
                extra_obstacle_state = np.ravel(obstacle_info)
            else:
                extra_obstacle_state = obstacle_info

        if self.counter == 0:
            self.single_step_extra_goal_state_size = len(extra_goal_state)
            self.single_step_extra_obstacle_state_size = len(extra_obstacle_state)
            self.start_index_extra_observation = len(obs['observation'])
            extra_goal_state = np.concatenate([extra_goal_state, extra_goal_state])
            extra_obstacle_state = np.concatenate([extra_obstacle_state, extra_obstacle_state])
        else:
            #the first entries will always have the more recent representations
            prev_obs = self.last_obs.copy()
            begin_index = self.start_index_extra_observation
            end_index = self.start_index_extra_observation+self.single_step_extra_goal_state_size
            prev_goal_state = prev_obs['observation'][begin_index: end_index]

            begin_index = self.start_index_extra_observation+2*self.single_step_extra_goal_state_size
            end_index = self.start_index_extra_observation \
                        + 2*self.single_step_extra_goal_state_size \
                        + self.single_step_extra_obstacle_state_size
            prev_obstacle_state = prev_obs['observation'][begin_index: end_index]

            #the previous ones are pushed to the back
            extra_goal_state = np.concatenate([extra_goal_state, prev_goal_state])
            extra_obstacle_state = np.concatenate([extra_obstacle_state, prev_obstacle_state])

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