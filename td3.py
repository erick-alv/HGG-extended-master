import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# This implementation is based on https://github.com/sfujim/TD3/blob/master/TD3.py; which is based on the mentioned
# paper. Additional modifications, done to make it as leap paper and temporal difference models #TODO mention paper
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, rem_steps_dim, max_action, networks_hidden):
        super(Actor, self).__init__()
        index = 0
        self.l_in = nn.Linear(state_dim + goal_dim + rem_steps_dim, networks_hidden[index])
        self.hidden_layers = []
        for _ in range(len(networks_hidden)-1):
            self.hidden_layers.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]))
        self.l_out = nn.Linear(networks_hidden[index], action_dim)
        self.max_action = max_action

    def forward(self, state, goal, rem_steps):
        a = torch.cat([state, goal, rem_steps], 1)
        a = F.relu(self.l_in(a))
        for layer in self.hidden_layers:
            a = F.relu(layer(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, rem_steps_dim, networks_hidden):
        super(Critic, self).__init__()

        # Q1 architecture
        index = 0
        self.l1 = nn.Linear(state_dim + action_dim + goal_dim + rem_steps_dim, networks_hidden[index])
        self.hidden_layers1 = []
        for _ in range(len(networks_hidden)-1):
            self.hidden_layers1.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]))
        self.l3 = nn.Linear(networks_hidden[index], 1)

        # Q2 architecture
        index = 0
        self.l4 = nn.Linear(state_dim + action_dim + goal_dim + rem_steps_dim,  networks_hidden[index])
        self.hidden_layers2 = []
        for _ in range(len(networks_hidden)-1):
            self.hidden_layers2.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]))
        self.l6 = nn.Linear(networks_hidden[index], 1)

    def forward(self, state, action, goal, rem_steps):
        sa = torch.cat([state, action, goal, rem_steps], 1)

        q1 = F.relu(self.l1(sa))
        for layer in self.hidden_layers:
            q1 = F.relu(layer(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        for layer in self.hidden_layers:
            q2 = F.relu(layer(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, goal, rem_steps):
        sa = torch.cat([state, action, goal, rem_steps], 1)

        q1 = F.relu(self.l1(sa))
        for layer in self.hidden_layers:
            q1 = F.relu(layer(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            goal_dim,
            rem_steps_dim,
            max_action,
            args,
            networks_hidden = [400, 300],
            discount=0.99,#TODO see how much
            tau=0.005,
            policy_noise=0.2,#todo this better in wrapper and pass as arguiemnt
            noise_clip=0.5,
            policy_freq=2,
            lr = 0.001,
            ou_process =(0.3, 0.3)#todo this better in wrapper
    ):

        self.actor = Actor(state_dim, action_dim, goal_dim, rem_steps_dim, max_action, networks_hidden).to(args.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, goal_dim, rem_steps_dim, networks_hidden).to(args.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.args = args

    def get_action(self, state, goal, rem_steps):
        state, goal, rem_steps = [torch.FloatTensor(d.reshape(1, -1)).to(self.args.device)
                                  for d in [state, goal, rem_steps]]
        return self.actor(state, goal, rem_steps).cpu().data.numpy().flatten()

    def train(self, batch):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()
        self.total_it += 1

        # Sample replay buffer
        state, goal, rem_steps, action, next_state, reward, done = (batch[k] for k in ['obs', 'goal', 'rem_steps',
                                                                                       'action', 'next_obs', 'reward',
                                                                                       'done'])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state, goal, rem_steps-1) + noise#todo is -1??;what to do if then ==-1
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, goal, rem_steps-1)#todo is -1??;what to do if then ==-1
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, goal, rem_steps)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state, goal, rem_steps), goal, rem_steps).mean()#todo shpu;d not be here rem_steps-1

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        save_dict = {
            'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }
        path = self.args.dirpath + 'weights_dir/'+filename
        torch.save(save_dict, path)

    def save_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.save(filename)

    def load(self, filename):
        path = self.args.dirpath + 'weights_dir/' + filename
        save_dict = torch.load(path)
        self.critic.load_state_dict(save_dict['critic'])
        self.critic_optimizer.load_state_dict(save_dict['critic_optimizer'])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(save_dict['actor'])
        self.actor_optimizer.load_state_dict(save_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)

    def load_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load(filename)