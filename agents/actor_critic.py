import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# This implementation is based on https://github.com/sfujim/TD3/blob/master/TD3.py; which is based on the mentioned
# paper. Paper: https://arxiv.org/abs/1802.09477
#Additional modifications, done to make it as leap paper and temporal difference models


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + goal_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state, goal):
        s = torch.cat([state, goal], 1)
        a = F.relu(self.l1(s))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, goal, action):
        sa = torch.cat([state, goal, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            max_action,
            args,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=3e-4
    ):
        max_action = torch.from_numpy(max_action).type(torch.float32).to(args.device)
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(args.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.criticQ1 = Critic(state_dim, goal_dim, action_dim).to(args.device)
        self.criticQ1_target = copy.deepcopy(self.criticQ1)
        self.criticQ1_optimizer = torch.optim.Adam(self.criticQ1.parameters(), lr=lr)

        self.criticQ2 = Critic(state_dim, goal_dim, action_dim).to(args.device)
        self.criticQ2_target = copy.deepcopy(self.criticQ2)
        self.criticQ2_optimizer = torch.optim.Adam(self.criticQ2.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.args = args

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.criticQ1.train()
        self.criticQ1_target.train()
        self.criticQ2.train()
        self.critic2_target.train()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.criticQ1.eval()
        self.criticQ1_target.eval()
        self.criticQ2.eval()
        self.critic2_target.eval()

    def get_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(self.args.device)
        return self.actor(state, goal).cpu().data.numpy().flatten()

    def transform_single_batch_to_tensor(self, batch_single):
        if not isinstance(batch_single, np.ndarray):
            batch_single = np.array(batch_single)
        batch_single = torch.from_numpy(batch_single).float().to(self.args.device)
        return batch_single

    def clip_to(self, to_clip, clip_values):
        a = torch.max(to_clip, -clip_values)
        return torch.min(a, clip_values)

    def train_with_batch(self, batch):
        self.total_it += 1

        # Sample replay buffer 
        state = self.transform_single_batch_to_tensor(batch['obs'])
        next_state = self.transform_single_batch_to_tensor(batch['obs_next'])
        goal = self.transform_single_batch_to_tensor(batch['goals'])
        action = self.transform_single_batch_to_tensor(batch['acts'])
        reward = self.transform_single_batch_to_tensor(batch['rews'])
        done = self.transform_single_batch_to_tensor(batch['done'])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = self.clip_to(self.actor_target(next_state, goal) + noise, self.max_action.unsqueeze(0))

            # Compute the target Q value
            target_Q1 = self.criticQ1_target(next_state, goal, next_action)
            target_Q2 = self.criticQ2_target(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.criticQ1(state, goal, action)
        current_Q2 = self.criticQ2(state, goal, action)

        # Compute critic loss
        criticQ1_loss = F.mse_loss(current_Q1, target_Q)
        criticQ2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.criticQ2_optimizer.zero_grad()
        criticQ1_loss.backward()
        self.criticQ1_optimizer.step()
        self.criticQ2_optimizer.zero_grad()
        criticQ2_loss.backward()
        self.criticQ2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.criticQ1(state, goal, self.actor(state, goal)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.criticQ1.parameters(), self.criticQ1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.criticQ2.parameters(), self.criticQ2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            return float(criticQ1_loss), float(criticQ2_loss), float(actor_loss)

        return float(criticQ1_loss), float(criticQ2_loss), None

    def save(self, filename):
        save_dict = {
            'criticQ1': self.criticQ1.state_dict(),
            'criticQ1_optimizer': self.criticQ1_optimizer.state_dict(),
            'criticQ2': self.criticQ2.state_dict(),
            'criticQ2_optimizer': self.criticQ2_optimizer.state_dict(),
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }
        path = self.args.dirpath + 'weights_dir/' + filename
        torch.save(save_dict, path)

    def save_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.save(filename)

    def load(self, filename):
        path = self.args.dirpath + 'weights_dir/' + filename
        save_dict = torch.load(path)
        self.criticQ1.load_state_dict(save_dict['criticQ1'])
        self.criticQ1_optimizer.load_state_dict(save_dict['criticQ1_optimizer'])
        self.criticQ1_target = copy.deepcopy(self.criticQ1)

        self.criticQ2.load_state_dict(save_dict['criticQ2'])
        self.criticQ2_optimizer.load_state_dict(save_dict['criticQ2_optimizer'])
        self.criticQ2_target = copy.deepcopy(self.criticQ2)

        self.actor.load_state_dict(save_dict['actor'])
        self.actor_optimizer.load_state_dict(save_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)

    def load_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load(filename)


