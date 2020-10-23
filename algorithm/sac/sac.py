#adapted from https://github.com/djbyrne/SAC.git which is based on
# rlkit(https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py) and spinning up

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from algorithm.replay_buffer import goal_based_process
import copy

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400, 300], init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, device, hidden_size=[400, 300],
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetwork, self).__init__()

        self.device = device

        self.epsilon = epsilon

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.mean_linear = nn.Linear(hidden_size[1], num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size[1], num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, deterministic=False):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        log_prob = None

        if deterministic:
            action = torch.tanh(mean)
        else:
            # assumes actions have been normalized to (0,1) #todo verify this in our case I believe is (-1,1)
            normal = Normal(0, 1)
            z = mean + std * normal.sample().requires_grad_()
            action = torch.tanh(z)
            log_prob_term_1 = Normal(mean, std).log_prob(z)
            log_prob_term_1 = log_prob_term_1.sum(-1)
            log_prob_term_2 = torch.log(1 - action * action + self.epsilon)
            log_prob_term_2 = log_prob_term_2.sum(-1)
            log_prob = log_prob_term_1 - log_prob_term_2

        return action, mean, log_std, log_prob, std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _, _, _ = self.forward(state, deterministic)
        act = action.cpu()[0][0]
        return act


class SAC(object):

    def __init__(self, args, hidden_dim=[400, 300], lr=1e-3, auto_alpha=True):
        # Set seeds
        #TODO save seed as parameter
        torch.manual_seed(1)
        np.random.seed(1)
        self.args = args

        # env space
        self.state_dim = self.args.obs_dims[0]
        self.action_dim = self.args.acts_dims[0]
        self.hidden_dim = hidden_dim

        # device
        self.device = args.agent_device

        # init networks

        # Soft Q
        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.target_soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Policy
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.device, self.hidden_dim).to(self.device)

        # Optimizers/Loss
        self.soft_q_criterion = nn.MSELoss()

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.args.q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.args.q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.args.pi_lr)

        # alpha tuning
        self.auto_alpha = auto_alpha

        if self.auto_alpha:
            #self.target_entropy = -np.prod(env.action_space.shape).item()
            self.target_entropy = -np.prod(tuple(self.args.acts_dims)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.discount = self.args.gamma
        self.tau = self.args.polyak

        #added to adapt to logger
        self.train_info = {
            'Pi_q_loss': [],
            'Pi_l2_loss': [],
            'Q_loss': []
        }

    def get_action(self, state, deterministic=False, explore=False):#todo

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #if explore:
        #    return self.env.action_space.sample()
        #else:
        action = self.policy_net.get_action(state, deterministic).detach()
        return action.numpy()

    def update(self, state, action, reward, next_state, done):
        #with torch.autograd.set_detect_anomaly(True):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(np.float32(done)).to(self.device)

        new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy_net(state)

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 0.2  # constant used by OpenAI

        # Update Policy
        q_new_actions = torch.min(
            self.soft_q_net1(state, new_actions),
            self.soft_q_net2(state, new_actions)
        )

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        # Update Soft Q Function
        q1_pred = self.soft_q_net1(state, action)
        q2_pred = self.soft_q_net2(state, action)

        new_next_actions, _, _, new_log_pi, *_ = self.policy_net(next_state)

        target_q_values = torch.min(
            self.target_soft_q_net1(next_state, new_next_actions),
            self.target_soft_q_net2(next_state, new_next_actions),
        ) - alpha * new_log_pi

        q_target = reward + (1 - done) * self.discount * target_q_values
        q1_loss = self.soft_q_criterion(q1_pred, q_target.detach())
        q2_loss = self.soft_q_criterion(q2_pred, q_target.detach())

        # Update Networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.soft_q_optimizer1.zero_grad()
        q1_loss.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q2_loss.backward()
        self.soft_q_optimizer2.step()



        # Soft Updates
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        return {
            'Pi_q_loss': policy_loss.item(),
            'Pi_l2_loss': 0.,#todo use l2 value for pi loss?? In normal SAC this is not used
            'Q_loss': ((q1_loss + q2_loss)/2.).item()
        }

    def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.steps_counter < self.args.warmup):
            return np.random.uniform(-1, 1, size=self.args.acts_dims)
        if self.args.goal_based: obs = goal_based_process(obs)

        # eps-greedy exploration
        if explore and np.random.uniform() <= self.args.eps_act:
            return np.random.uniform(-1, 1, size=self.args.acts_dims)

        action = self.get_action(obs, explore=explore)
        if test_info:
            #action, info = self.sess.run([self.pi, self.step_info], feed_dict)
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_to_execute = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            q_pi = torch.min(
                self.soft_q_net1(state, action_to_execute),
                self.soft_q_net2(state, action_to_execute)
            )
            info = {'Q_average': q_pi.item()}
        action = action[0]

        # uncorrelated gaussian explorarion
        #if explore: action += np.random.normal(0, self.args.std_act, size=self.args.acts_dims) removed sin in theory SAC does not need it
        #action = np.clip(action, -1, 1)
        assert action >= -1
        assert action <= 1

        if test_info:
            return action, info
        return action

    def step_batch(self, obs):
        actions = self.get_action(obs)
        return actions

    def get_q_pi(self, obs):
        action = self.get_action(obs)
        state = torch.FloatTensor(obs).to(self.device)
        action_to_execute = torch.FloatTensor(action).to(self.device)
        q_pi = torch.min(
            self.soft_q_net1(state, action_to_execute),
            self.soft_q_net2(state, action_to_execute)
        )[:, 0]
        q_pi = q_pi.detach().numpy()
        value = np.clip(q_pi, -1.0 / (1.0 - self.args.gamma), 0)#todo THE VALUE was positive ans should be negative see if always is so
        return value

    def get_els_from_batch(self, batch):
        return batch['obs'], batch['acts'], batch['rews'], batch['obs_next'], batch['done']

    def train(self, batch):
        state, action, reward, next_state, done = self.get_els_from_batch(batch)
        info = self.update(state, action, reward, next_state, done)
        return info

    #train pi and train q not implemented, since apparently not used

    def normalizer_update(self, batch):
        #TODO??
        pass

    def target_update(self):
        # TODO??
        pass

    def save(self, filename, global_step = None):
        fname = copy.copy(filename)
        if global_step is not None:
            fname = fname+'_'+global_step

        #not save target network since it should have same values when storing
        save_dict = {}
        {
            'soft_q_net1': self.soft_q_net1.state_dict(),
            'soft_q_net2': self.soft_q_net2.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'soft_q_optimizer1': self.soft_q_optimizer1.state_dict(),
            'soft_q_optimizer2 ': self.soft_q_optimizer2 .state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }
        if self.auto_alpha:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['log_alpha_optimizer'] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, filename)