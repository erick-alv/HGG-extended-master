import numpy as np
from algorithm.replay_buffer import Trajectory, ReplayBuffer_Episodic
from copy import deepcopy
#based in thesame same methods from this
class TDM_Trajectory(Trajectory):#TODO really subclass??
    def __init__(self):
        self.ep = {
            'obs': [],
            'goal': [],
            'rem_steps': [],
            'action': [],
            'next_obs': [],
            'rewards': [],
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
        self.ep['rewards'].append(deepcopy(reward))
        self.ep['done'].append(deepcopy(done))
        self.length += 1

#important this super class stores the wholes episode (rollout) on an index, not just one observation
class ReplayBuffer_TDM(ReplayBuffer_Episodic):
    def __init__(self, args, training_env):
        #first_overwrite attribute, so energy sampling is not used
        if hash(args, 'buffer_type'):
            args.buffer_type = 'none'
        super(ReplayBuffer_TDM, self).__init__()
        #overwrite sample method
        self.sample_batch = self.resampled_random_batch
        self.training_env = training_env

    def __len__(self):
        return self.length


    def resampled_random_batch(self, batch_size, fraction_rollout=0.2, fraction_random=0.2):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = []
        num_rollout = int(batch_size * fraction_rollout)
        num_random = int(batch_size * fraction_random)
        num_future = batch_size - num_rollout + num_random
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
                    goal = self.buffer['next_obs'][goal_ep][goal_step]
                    rem_steps = np.random(self.args.max_rem_steps)

                else:#future sampling
                    goal_step = np.random.randint(low=step, high=self.steps[ep])
                    goal = self.buffer['next_obs'][ep][goal_step]
                    if np.random.randint(2) % 2 == 0:
                        rem_steps = np.random(0, self.args.max_rem_steps + 1)
                    else:
                        rem_steps = batch['rem_steps'][-1]
                reward = self.training_env.env.env.compute_reward(batch['next_obs'][-1], goal, info=None)
                done = batch['next_obs'][-1] == goal and rem_steps > 0
                batch['goal'][-1] = goal
                batch['reward'] = reward
                batch['rem_steps'] = rem_steps
                batch['done'] = done
        for key in batch.keys():
            batch[key] = np.array(batch[key])
        return batch