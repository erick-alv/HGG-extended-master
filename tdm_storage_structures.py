import numpy as np
from algorithm.replay_buffer import Trajectory, ReplayBuffer_Episodic
from copy import deepcopy
import torch
#based in thesame same methods from this
class TDM_Trajectory(Trajectory):#TODO really subclass??
    def __init__(self):
        self.ep = {
            'obs': [],
            'goal': [],
            'rem_steps': [],
            'action': [],
            'next_obs': [],
            'reward': [],
            'done': []
        }
        self.length = 0

    def __len__(self):
        return self.length

    def store_step(self, state, goal, rem_steps, action, next_state, reward, done):
        self.ep['obs'].append(deepcopy(state))
        self.ep['goal'].append(deepcopy(goal))
        self.ep['rem_steps'].append(deepcopy(rem_steps))
        self.ep['action'].append(deepcopy(action))
        self.ep['next_obs'].append(deepcopy(next_state))
        self.ep['reward'].append(deepcopy(reward))
        self.ep['done'].append(deepcopy(done))
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

    def resampled_random_batch(self, batch_size, fraction_rollout=0.2, fraction_random=0.2,
                               obs_latent_concatenated=True, transform_to_tensor=False):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = []
        num_rollout = int(batch_size * fraction_rollout)
        num_random = int(batch_size * fraction_random)
        num_future = batch_size - (num_rollout + num_random)
        for i in range(batch_size):
            ep = np.random.randint(self.length)
            step = np.random.randint(self.steps[ep])
            for key in self.buffer.keys():
                batch[key].append(self.buffer[key][ep][step])
            if i < num_rollout:
                pass
                # Nothing to do
            else:
                if i < num_rollout + num_random: #random sampling
                    goal_ep = np.random.randint(self.length)
                    goal_step = np.random.randint(self.steps[goal_ep])
                    if obs_latent_concatenated:
                        whole_next = self.buffer['next_obs'][goal_ep][goal_step]
                        goal = np.concatenate((whole_next[3:6], whole_next[- self.args.latent_dim:]))#achieved_goal + latent_dim
                    else:
                        goal = self.buffer['next_obs'][goal_ep][goal_step]
                    rem_steps = np.random.randint(0, self.args.tdm_env_steps + 1)

                else:#future sampling
                    goal_step = np.random.randint(low=step, high=self.steps[ep])
                    if obs_latent_concatenated:
                        whole_next = self.buffer['next_obs'][ep][goal_step]
                        goal = np.concatenate((whole_next[3:6], whole_next[- self.args.latent_dim:]))# achieved_goal + latent_dim
                    else:
                        goal = self.buffer['next_obs'][ep][goal_step]
                    if np.random.randint(2) % 2 == 0:
                        rem_steps = np.random.randint(0, self.args.tdm_env_steps + 1)
                    else:
                        rem_steps = batch['rem_steps'][-1]
                reward = self.training_env.env.env.compute_reward(batch['next_obs'][-1][3:6], goal[:3], info=None)
                #todo see how hgg deals with this the parameter done is always False in their case
                done = np.array_equal(batch['next_obs'][-1][3:6], goal[:3]) \
                       and rem_steps > 0#todo use latent as well??; todo verify since remsteps could be just 1
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