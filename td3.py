import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.replay_buffer import Trajectory

class TD3_Trajectory:#(Trajectory):
    def __init__(self):
        self.state = []
        self.goal = []
        self.rem_steps = []
        self.action= []
        self.next_state = []
        self.reward = []
        self.not_done = []
        self.length = 0

    def __len__(self):
        return self.length

    def store_step(self, state, goal, rem_steps, action, next_state, reward, not_done):
        self.state.append(state)
        self.goal.append(goal)
        self.rem_steps.append(rem_steps)
        self.action.append(action)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.not_done.append(not_done)
        self.length += 1

	#TODO??
	'''def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
		# from "Energy-Based Hindsight Experience Prioritization"
		if env_id[:5]=='Fetch':
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['achieved_goal'])
			obj = np.array([obj])

			clip_energy = 0.5
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			g, m, delta_t = 9.81, 1, 0.04
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)
		else:
			assert env_id[:4]=='Hand'
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['observation'][-7:])
			obj = np.array([obj])

			clip_energy = 2.5
			g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
			quaternion = obj[:,:,3:].copy()
			angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
			diff_angle = np.diff(angle, axis=1)
			angular_velocity = diff_angle / delta_t
			rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
			rotational_energy = np.sum(rotational_energy, axis=2)
			obj = obj[:,:,:3]
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy + w_rotational*rotational_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)'''


class Replay_Buffer:
    def __init__(self, args):
        self.args = args
        if args.buffer_type == 'energy':
            self.energy = True
            self.energy_sum = 0.0
            self.energy_offset = 0.0
            self.energy_max = 1.0
        else:
            self.energy = False
        self.buffer = {}
        self.steps = []
        self.length = 0
        self.counter = 0
        self.steps_counter = 0
        self.sample_methods = {
            'ddpg': self.sample_batch_ddpg
        }
        self.sample_batch = self.sample_methods[args.alg]

    def store_trajectory(self, trajectory):
        episode = trajectory.ep
        if self.energy:
            energy = trajectory.energy(self.args.env)
            self.energy_sum += energy
        if self.counter == 0:
            for key in episode.keys():
                self.buffer[key] = []
            if self.energy:
                self.buffer_energy = []
                self.buffer_energy_sum = []
        if self.counter < self.args.buffer_size:
            for key in self.buffer.keys():
                self.buffer[key].append(episode[key])
            if self.energy:
                self.buffer_energy.append(copy.deepcopy(energy))
                self.buffer_energy_sum.append(copy.deepcopy(self.energy_sum))
            self.length += 1
            self.steps.append(trajectory.length)
        else:
            idx = self.counter % self.args.buffer_size
            for key in self.buffer.keys():
                self.buffer[key][idx] = episode[key]
            if self.energy:
                self.energy_offset = copy.deepcopy(self.buffer_energy_sum[idx])
                self.buffer_energy[idx] = copy.deepcopy(energy)
                self.buffer_energy_sum[idx] = copy.deepcopy(self.energy_sum)
            self.steps[idx] = trajectory.length
        self.counter += 1
        self.steps_counter += trajectory.length

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
    # todo use relu for activation
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

    def select_action(self, state, goal, rem_steps):
        state, goal, rem_steps = [torch.FloatTensor(d.reshape(1, -1)).to(self.args.device)
                                  for d in [state, goal, rem_steps]]
        return self.actor(state, goal, rem_steps).cpu().data.numpy().flatten()

    def train(self, batch):
        self.total_it += 1

        # Sample replay buffer
        state, goal, rem_steps, action, next_state, reward, not_done = batch

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
            target_Q = reward + not_done * self.discount * target_Q

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
        #todo save in dict
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        # todo save in dict
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)