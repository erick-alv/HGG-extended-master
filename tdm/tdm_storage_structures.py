import numpy as np
from algorithm.replay_buffer import Trajectory, ReplayBuffer_Episodic
from copy import deepcopy
import torch
from temp_func import conc_latent, rem_latent, from_raw_get_achieved

#based in thesame same methods from this
class TDM_Trajectory(Trajectory):
    def __init__(self):
        self.ep = {
            'obs': [],
            '_obs': [],
            'goal': [],
            '_goal':[],
            'rem_steps': [],
            'action': [],
            'next_obs': [],
            '_next_obs': [],
            'reward': [],
            'done': []
        }
        self.length = 0

    def __len__(self):
        return self.length

    def store_step(self, state, goal, rem_steps, action, next_state, reward, done,
                   _state=None, _goal=None, _next_state=None):
        self.ep['obs'].append(deepcopy(state))
        self.ep['goal'].append(deepcopy(goal))
        self.ep['rem_steps'].append(deepcopy(rem_steps))
        self.ep['action'].append(deepcopy(action))
        self.ep['next_obs'].append(deepcopy(next_state))
        self.ep['reward'].append(deepcopy(reward))
        self.ep['done'].append(deepcopy(done))
        if _state is not None:
            self.ep['_obs'].append(deepcopy(_state))
        if _goal is not None:
            self.ep['_goal'].append(deepcopy(_goal))
        if _next_state is not None:
            self.ep['_next_obs'].append(deepcopy(_next_state))
        self.length += 1

#important this super class stores the wholes episode (rollout) on an index, not just one observation
class ReplayBuffer_TDM(ReplayBuffer_Episodic):
    def __init__(self, args, training_env):
        #first_overwrite attribute, so energy sampling is not used
        if not hasattr(args, 'buffer_type'):
            args.buffer_type = 'none'
        super(ReplayBuffer_TDM, self).__init__(args=args)
        #overwrite sample method
        self.sample_batch = self.resampled_random_batch
        self.training_env = training_env

    def resampled_random_batch(self, batch_size, fraction_rollout=0.2, fraction_random=0.2, transform_to_tensor=False):
        batch = {'obs':[], 'goal': [], 'rem_steps': [], 'action': [],'next_obs': [], 'reward': [], 'done': []}

        num_rollout = int(batch_size * fraction_rollout)
        num_random = int(batch_size * fraction_random)
        num_future = batch_size - (num_rollout + num_random)
        for i in range(batch_size):
            ep = np.random.randint(self.length)
            step = np.random.randint(self.steps[ep])
            for key in batch.keys():
                batch[key].append(self.buffer[key][ep][step])
            achieved_g = self.buffer['_next_obs'][ep][step]
            if i < num_rollout:
                pass
                # Nothing to do
            else:
                if i < num_rollout + num_random: #random sampling
                    goal_ep = np.random.randint(self.length)
                    goal_step = np.random.randint(self.steps[goal_ep])
                    if self.args.just_latent:
                        goal = self.buffer['next_obs'][goal_ep][goal_step]
                    else:
                        whole_next = self.buffer['next_obs'][goal_ep][goal_step]
                        raw_obs, latent = rem_latent(whole_next, self.args.latent_dim)
                        goal = conc_latent(from_raw_get_achieved(raw_obs), latent)
                    expected_g = self.buffer['_next_obs'][goal_ep][goal_step]
                    rem_steps = np.random.randint(0, self.args.max_tau + 1)

                else:#future sampling
                    goal_step = np.random.randint(low=step, high=self.steps[ep])
                    if self.args.just_latent:
                        goal = self.buffer['next_obs'][ep][goal_step]
                    else:
                        whole_next = self.buffer['next_obs'][ep][goal_step]
                        raw_obs, latent = rem_latent(whole_next, self.args.latent_dim)
                        goal = conc_latent(from_raw_get_achieved(raw_obs), latent)
                    expected_g = self.buffer['_next_obs'][ep][goal_step]
                    if np.random.randint(2) % 2 == 0:
                        rem_steps = np.random.randint(0, self.args.max_tau + 1)
                    else:
                        rem_steps = batch['rem_steps'][-1]
                '''
                todo SEE IF leave this way appart
                raw_obs, _ = rem_latent(batch['next_obs'][-1], self.args.latent_dim)
                raw_goal, _ = rem_latent(goal, self.args.latent_dim)
                reward = self.training_env.env.env.compute_reward(from_raw_get_achieved(raw_obs),
                                                                  raw_goal, info=None)'''

                reward = self.training_env.env.env.compute_reward(achieved_g, expected_g, info =None)
                done = reward >=0 and rem_steps > 0#todo use latent as well??
                batch['goal'][-1] = goal
                batch['reward'][-1] = reward
                batch['rem_steps'][-1] = rem_steps
                batch['done'][-1] = done
        for key in batch.keys():
            batch[key] = np.array(batch[key])
        if transform_to_tensor:
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key], dtype=torch.float).to(self.args.device)
                if key == 'reward' or key == 'done' or key == 'rem_steps':
                    batch[key] = torch.unsqueeze(batch[key], 1)
        return batch