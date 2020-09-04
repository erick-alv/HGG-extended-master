import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from utils.transforms_utils import transform_np_image_to_tensor, np_to_tensor
import numpy as np
import copy

class EstimatorNet(nn.Module):
    def __init__(self, output_dim, st1_dim, st2_dim, min_range, max_range, device, hidden_dim=256):
        super(EstimatorNet, self).__init__()
        if not isinstance(max_range, torch.Tensor):
            self.min_range = torch.Tensor(min_range)
        else:
            self.min_range = min_range
        if not isinstance(min_range, torch.Tensor):
            self.max_range = torch.Tensor(max_range)
        else:
            self.max_range = max_range
        #values for range transformation
        self.max_range = self.max_range.unsqueeze(0).to(device)
        self.min_range = self.min_range.unsqueeze(0).to(device)
        self.range_dist = self.max_range - self.min_range
        self.device = device

        self.output_dim = output_dim
        self.st1_dim = st1_dim
        self.st2_dim = st2_dim
        self.hidden_dim = hidden_dim

        self.l1 = nn.Linear(self.st1_dim+self.st2_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.output_dim)

    def estimate(self, z1, z2):
        z1_z2 = torch.cat([z1, z2], 1)
        d = F.relu(self.l1(z1_z2))
        d = F.relu(self.l2(d))
        d = self.l3(d)
        #return d * self.range_dist + self.min_range
        return d

    def forward(self, z1, z2):
        return self.estimate(z1, z2)

class DistanceEstimator:
    def __init__(self, output_dim, state_dim, goal_dim, min_range, max_range, observation_type, goal_type,
                 args, lr=3e-4,hidden_dim=256, tau=0.005):
        self.args = args
        self.estimator = EstimatorNet(output_dim, state_dim, goal_dim, min_range,
                                      max_range, args.device, hidden_dim).to(args.device)
        self.estimator_target = copy.deepcopy(self.estimator)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=lr)
        self.total_it = 0
        self.tau = tau
        self.observation_type = observation_type
        self.goal_type = goal_type

        if goal_type == 'latent':
            self.goal_key = 'goal_latent'
            self.goal_her_key = 'state_latent'
        elif goal_type == 'goal_space':
            self.goal_key = 'desired_goal'
            self.goal_her_key = 'achieved_goal'
        elif goal_type == 'state':
            self.goal_key = 'goal_state'
            self.goal_her_key = 'observation'
        elif goal_type == 'concat':
            raise Exception('not implemented yet')
        else:
            raise Exception('goal obs type not valid')
        if observation_type == 'latent':
            self.state_key = 'state_latent'
        elif observation_type == 'real':
            self.state_key = 'observation'
        elif observation_type == 'concat':
            raise Exception('not implemented yet')
        else:
            raise Exception('observation type not valid')

    def estimate_distance(self, z1, z2):
        z1 = torch.FloatTensor(z1.reshape(1, -1)).to(self.args.device)
        z2 = torch.FloatTensor(z2.reshape(1, -1)).to(self.args.device)
        return self.estimator(z1, z2).cpu().data.numpy().flatten()[0]

    def train_with_labels(self, batch, ae, trainer_ae, optimize_ae_too=False):
        #it is possible to read latents states from trajectories, but then is not possible to backpropagate
        #in the encoder if we want to
        for _ in range(3):
            loss = None
            for trajectory in batch:
                l = len(trajectory)
                state_dim = trajectory[0]['observation'].shape[0]
                goal_space_dim = trajectory[0]['achieved_goal'].shape[0]
                expected_distances = [np.arange(0, l - i) for i in range(l)]
                expected_distances = np.concatenate(expected_distances)
                expected_distances = torch.from_numpy(expected_distances).float().to(self.args.device).unsqueeze(1)
                if self.observation_type == 'latent' or self.observation_type == 'concat':
                    ims = [transform_np_image_to_tensor(trajectory[i]['state_image'], self.args) for i in range(l)]
                    ims = torch.stack(ims)
                    zs = ae.encode(ims)
                    del ims

                    if self.observation_type == 'latent':
                        z1s = [zs[i].expand(l - i, self.args.latent_dim) for i in range(l)]
                        z1s = torch.cat(z1s, dim=0)
                    else:
                        z1s = [zs[i].expand(l - i, self.args.latent_dim) for i in range(l)]
                        z1s = torch.cat(z1s, dim=0)
                        states = [np_to_tensor(trajectory[i]['observation'], self.args) for i in range(l)]
                        states = [states[i].unsqueeze(0) for i in range(l)]
                        states = [states[i].expand(l - i, state_dim) for i in range(l)]
                        states = torch.cat(states, dim=0)
                        z1s = torch.cat([states, z1s], dim=1)#IMPORTNAT always first state then latent
                elif self.observation_type == 'real':
                    states = [np_to_tensor(trajectory[i]['observation'], self.args) for i in range(l)]
                    states = [states[i].unsqueeze(0) for i in range(l)]
                    states = [states[i].expand(l - i, state_dim) for i in range(l)]
                    states = torch.cat(states, dim=0)
                    z1s = states
                else:
                    raise Exception('observation type not valid')

                if self.goal_type == 'latent' or self.goal_type == 'concat':
                    z2s = [zs[i:] for i in range(0, l)]
                    z2s = torch.cat(z2s, dim=0)
                    if self.observation_type == 'latent':
                        pass
                    else:
                        states = [np_to_tensor(trajectory[i]['observation'], self.args) for i in range(l)]
                        states = [states[i].unsqueeze(0) for i in range(l)]
                        states = [torch.cat(states[i:], dim=0) for i in range(0, l)]
                        states = torch.cat(states, dim=0)
                        z2s = torch.cat([states, z2s], dim=1)
                elif self.goal_type == 'goal_space' or self.goal_type == 'state':
                    keyword = 'achieved_goal' if self.goal_type == 'goal_space' else 'observation'
                    states = [np_to_tensor(trajectory[i][keyword], self.args) for i in range(l)]
                    states = [states[i].unsqueeze(0) for i in range(l)]
                    states = [torch.cat(states[i:], dim=0) for i in range(0, l)]
                    states = torch.cat(states, dim=0)
                    z2s = states
                else:
                    raise Exception('goal obs type not valid')
                est_dist = self.estimator(z1s, z2s)
                l = nn.MSELoss()(est_dist, expected_distances)
                loss = loss + l if loss is not None else l
            m = 1.0/float(len(batch))
            loss = loss.mul_(0.5*m)
            if optimize_ae_too:
                trainer_ae.optimizer.zero_grad()#todo it will be necessary to calculate it other way
                self.optimizer.zero_grad()
                loss.backward()
                trainer_ae.optimizer.step()
                self.optimizer.step()
            else:
                z1s.detach()
                z2s.detach()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return float(loss)

    def transform_single_batch_to_tensor(self, batch_single):
        if not isinstance(batch_single, np.ndarray):
            batch_single = np.array(batch_single)
        batch_single = torch.from_numpy(batch_single).float().to(self.args.device)
        return batch_single

    def train_rl(self, batch):
        self.total_it += 1

        # Sample replay buffer
        state = self.transform_single_batch_to_tensor(batch['obs'])
        next_state = self.transform_single_batch_to_tensor(batch['obs_next'])
        goal = self.transform_single_batch_to_tensor(batch['goals'])
        dis = self.transform_single_batch_to_tensor(batch['dis'])
        done = self.transform_single_batch_to_tensor(batch['done'])

        with torch.no_grad():
            # Select action according to policy and add clipped noise

            # Compute the target Q value
            next_distance = self.estimator_target(next_state, goal)
            target_d = dis + (1. - done) * 1.0 * next_distance

        # Get current Q estimates
        current_distance = self.estimator(state, goal)

        # Compute critic loss
        loss = F.mse_loss(target_d, current_distance)
        # Optimize the critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Delayed policy updates
        if self.total_it % 3 == 0:

            # Update the frozen target models
            for param, target_param in zip(self.estimator.parameters(), self.estimator_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return float(loss)

    def save(self, filename):
        save_dict = {
            'estimator': self.estimator.state_dict(),
            'optimizer': self.optimizer.state_dict()
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
        self.estimator.load_state_dict(save_dict['estimator'])
        self.optimizer.load_state_dict(save_dict['optimizer'])
        self.estimator_target = copy.deepcopy(self.estimator)

    def load_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load(filename)