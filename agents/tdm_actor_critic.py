import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorTDM(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, rem_steps_dim, max_action, device, networks_hidden):
        super(ActorTDM, self).__init__()
        index = 0
        self.l_in = nn.Linear(state_dim + goal_dim + rem_steps_dim, networks_hidden[index]).to(device)
        self.hidden_layers = []
        for _ in range(len(networks_hidden) - 1):
            self.hidden_layers.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]).to(device))
            index += 1
        self.l3 = nn.Linear(networks_hidden[index], action_dim).to(device)
        self.max_action = torch.tensor(max_action, dtype=float).to(device)

    def forward(self, state, goal, rem_steps):
        a = torch.cat([state, goal, rem_steps], 1)
        a = F.relu(self.l_in(a))
        for layer in self.hidden_layers:
            a = F.relu(layer(a))
        return self.max_action * torch.tanh(self.l3(a))


class CriticTDM(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, rem_steps_dim, value_dim, device, networks_hidden):
        super(CriticTDM, self).__init__()

        index = 0
        self.l1 = nn.Linear(state_dim + action_dim + goal_dim + rem_steps_dim, networks_hidden[index]).to(device)
        self.hidden_layers1 = []
        for _ in range(len(networks_hidden) - 1):
            self.hidden_layers1.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]).to(device))
            index += 1
        self.l3 = nn.Linear(networks_hidden[index], value_dim).to(device)

    def forward(self, state, action, goal, rem_steps):
        sa = torch.cat([state, action, goal, rem_steps], 1)

        q = F.relu(self.l1(sa))
        for layer in self.hidden_layers1:
            q = F.relu(layer(q))
        q = self.l3(q)

        return q


class TD3TDM(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            goal_dim,
            rem_steps_dim,
            max_action,
            value_dim,
            args,
            networks_hidden=[400, 300],
            discount=1.0,
            tau=0.005,
            policy_noise=0.2,  # todo this better in wrapper and pass as arguiemnt
            noise_clip=0.5,
            policy_freq=2,
            lr=0.001,  # todo this better in wrapper
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.rem_steps_dim = rem_steps_dim
        self.max_action = max_action

        self.actor = ActorTDM(state_dim, action_dim, goal_dim, rem_steps_dim, max_action, args.device,
                              networks_hidden).to(args.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = CriticTDM(state_dim, action_dim, goal_dim, rem_steps_dim, value_dim, args.device,
                                 networks_hidden).to(args.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = CriticTDM(state_dim, action_dim, goal_dim, rem_steps_dim, value_dim, args.device,
                                 networks_hidden).to(args.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.args = args

    def get_action(self, state, goal, rem_steps, return_as_tensor=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if not torch.is_tensor(goal):
            goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if not torch.is_tensor(rem_steps):
            if self.rem_steps_dim == 1:
                rem_steps = torch.tensor([[rem_steps]], dtype=torch.float).to(self.args.device)
            else:
                rem_steps = torch.tensor(rem_steps.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if return_as_tensor:
            return self.actor(state, goal, rem_steps)
        else:
            return self.actor(state, goal, rem_steps).cpu().data.numpy()

    def get_Q_val(self, state, action, goal, rem_steps, return_as_tensor=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if not torch.is_tensor(action):
            action = torch.tensor(action.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if not torch.is_tensor(goal):
            goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if not torch.is_tensor(rem_steps):
            if self.rem_steps_dim == 1:
                rem_steps = torch.tensor([[rem_steps]], dtype=torch.float).to(self.args.device)
            else:
                rem_steps = torch.tensor(rem_steps.reshape(1, -1), dtype=torch.float).to(self.args.device)
        if return_as_tensor:
            return self.critic1(state, action, goal, rem_steps)
        else:
            return self.critic1(state, action, goal, rem_steps).cpu().data.numpy()

    def clip_actions(self, actions):
        # from https://stackoverflow.com/questions/54738045/column-dependent-bounds-in-torch-clamp
        l = torch.tensor([-self.max_action], dtype=torch.float).to(self.args.device)
        u = torch.tensor([self.max_action], dtype=torch.float).to(self.args.device)
        a = torch.max(actions.float(), l)
        return torch.min(a, u)

    def train(self, batch):
        self.actor.train()
        self.actor_target.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
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

            next_action = self.clip_actions(
                self.actor_target(next_state, goal, rem_steps - 1.) + noise
            )

            # Compute the target Q value
            target_Q1 = self.critic1_target(next_state, next_action, goal, rem_steps - 1.)
            target_Q2 = self.critic2_target(next_state, next_action, goal, rem_steps - 1.)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1 = self.critic1(state, action, goal, rem_steps)
        current_Q2 = self.critic2(state, action, goal, rem_steps)

        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losses
            # todo leap paper additionally uses policy_saturation_cost = F.relu(torch.abs(pre_tanh_value) - 20.0)
            # but not used
            actor_loss = -self.critic1(state, self.actor(state, goal, rem_steps).float(), goal, rem_steps).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return float(critic1_loss + critic2_loss), float(actor_loss)

        else:
            return float(critic1_loss + critic2_loss), None

    def evaluate(self, batch):
        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()
        # Sample replay buffer
        state, goal, rem_steps, action, next_state, reward, done = (batch[k] for k in ['obs', 'goal', 'rem_steps',
                                                                                       'action', 'next_obs', 'reward',
                                                                                       'done'])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = self.clip_actions(
                self.actor_target(next_state, goal, rem_steps - 1.) + noise
            )

            # Compute the target Q value
            target_Q1 = self.critic1_target(next_state, next_action, goal, rem_steps - 1.)
            target_Q2 = self.critic2_target(next_state, next_action, goal, rem_steps - 1.)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.discount * target_Q
            # Get current Q estimates
            current_Q1 = self.critic1(state, action, goal, rem_steps)
            current_Q2 = self.critic2(state, action, goal, rem_steps)

            # Compute critic loss
            # todo LEAP paper optimizes them separately
            critic1_loss = F.mse_loss(current_Q1, target_Q)
            critic2_loss = F.mse_loss(current_Q2, target_Q)
            # Compute actor losses
            # todo leap paper additionally uses policy_saturation_cost = F.relu(torch.abs(pre_tanh_value) - 20.0) but not used
            actor_loss = -self.critic1(state, self.actor(state, goal, rem_steps).float(), goal, rem_steps).mean()

            return float(critic1_loss + critic2_loss), float(actor_loss)

    def save(self, filename):
        save_dict = {
            'critic1': self.critic1.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
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
        self.critic1.load_state_dict(save_dict['critic1'])
        self.critic1_optimizer.load_state_dict(save_dict['critic1_optimizer'])
        self.critic1_target = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(save_dict['critic2'])
        self.critic2_optimizer.load_state_dict(save_dict['critic2_optimizer'])
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(save_dict['actor'])
        self.actor_optimizer.load_state_dict(save_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)

    def load_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load(filename)