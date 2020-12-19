import numpy as np
import copy
from envs.distance_graph import DistanceGraph2D
from dist_train import load_DistNet_model, flat_entries, label_str_to_numpy
from scipy.stats import uniform
import pandas as pd
import torch

class DistMovEst:
    def __init__(self):
        self.max_x = None
        self.min_x = None
        self.max_y = None
        self.min_y = None
        self.x_mid = None
        self.y_mid = None
        self.s_x = None
        self.s_y = None
        self.update_calls = 0
        self.update_complete = False

    #todo use other distribution instead of uniform (for example norm), even better a more complex one
    def update(self, obstacle_latent_list, obstacle_size_latent_list):
        if len(obstacle_latent_list[0].shape) > 1:
            new_list_obs = []
            new_list_s = []
            for i in range(len(obstacle_latent_list)):
                for ob in obstacle_latent_list[i]:
                    new_list_obs.append(ob)
                for ob_s in obstacle_size_latent_list[i]:
                    new_list_s.append(ob_s)
            obstacle_latent_list = new_list_obs
            obstacle_size_latent_list = new_list_s

        obstacle_latent_array = np.array(obstacle_latent_list)
        obstacle_size_latent_array = np.array(obstacle_size_latent_list)
        a_x = np.min(obstacle_latent_array[:, 0])
        b_x = np.max(obstacle_latent_array[:, 0])
        a_y = np.min(obstacle_latent_array[:, 1])
        b_y = np.max(obstacle_latent_array[:, 1])
        s_x = np.mean(obstacle_size_latent_array[:, 0])
        s_y = np.mean(obstacle_size_latent_array[:, 1])
        if self.min_x == None:#assume everything else is
            self.max_x = b_x + s_x
            self.min_x = a_x - s_x
            self.max_y = b_y + s_y
            self.min_y = a_y - s_y
            self.s_x = s_x
            self.s_y = s_y
        else:
            if b_x + s_x > self.max_x:
                self.max_x = b_x + s_x
            if a_x - s_x < self.min_x:
                self.min_x = a_x - s_x
            if b_y + s_y > self.max_y:
                self.max_y = b_y + s_y
            if a_y - s_y < self.min_y:
                self.min_y = a_y - s_y

        self.x_mid = np.mean([self.max_x, self.min_x])
        self.y_mid = np.mean([self.max_y, self.min_y])
        self.update_calls += 1
        if self.update_calls == 3:
            self.update_complete = True

    def initialize_internal_distance_graph(self, field, num_vertices, size_increase):
        obstacles = []
        size_x = self.max_x - self.x_mid
        size_y = self.max_y - self.y_mid
        obstacles.append([self.x_mid, self.y_mid, size_x, size_y])
        self.obstacles = obstacles
        graph = DistanceGraph2D(args=None, field=field, num_vertices=num_vertices,
                                obstacles=obstacles, size_increase=size_increase, use_discrete=False)
        graph.compute_cs_graph()
        graph.compute_dist_matrix()
        self.graph = graph

    def calculate_distance_batch(self, goal_pos, current_pos_batch):
        d, _ = self.graph.get_dist_batch(goal_pos, current_pos_batch)
        indices_inf = d == np.inf
        d[indices_inf] = 9999
        return d


class DistMovEstReal(DistMovEst):
    def __init__(self):
        super(DistMovEstReal, self).__init__()
        self.real = True


    def update(self, obstacle_real_list, obstacle_size_real_list):
        obstacle_real_array = np.array(obstacle_real_list)

        a_x = np.min(obstacle_real_array[:, 0])
        b_x = np.max(obstacle_real_array[:, 0])
        a_y = np.min(obstacle_real_array[:, 1])
        b_y = np.max(obstacle_real_array[:, 1])
        s_x = np.mean(obstacle_real_array[:, 3])
        s_y = np.mean(obstacle_real_array[:, 4])
        if self.min_x == None:  # assume everything else is
            self.max_x = b_x + s_x
            self.min_x = a_x - s_x
            self.max_y = b_y + s_y
            self.min_y = a_y - s_y
            self.s_x = s_x
            self.s_y = s_y
        else:
            if b_x + s_x > self.max_x:
                self.max_x = b_x + s_x
            if a_x - s_x < self.min_x:
                self.min_x = a_x - s_x
            if b_y + s_y > self.max_y:
                self.max_y = b_y + s_y
            if a_y - s_y < self.min_y:
                self.min_y = a_y - s_y

        self.x_mid = np.mean([self.max_x, self.min_x])
        self.y_mid = np.mean([self.max_y, self.min_y])
        self.update_calls += 1
        if self.update_calls == 3:
            self.update_complete = True

    def initialize_internal_distance_graph(self, field, num_vertices, size_increase):
        obstacles = []
        size_x = self.max_x - self.x_mid
        size_y = self.max_y - self.y_mid
        obstacles.append([self.x_mid, self.y_mid, size_x, size_y])
        self.obstacles = obstacles
        graph = DistanceGraph2D(args=None, field=field, num_vertices=num_vertices,
                                obstacles=obstacles, size_increase=size_increase, use_discrete=False)
        graph.compute_cs_graph()
        graph.compute_dist_matrix()
        self.graph = graph
        distances = self.calculate_distance_batch(np.array([1.3, 0.9, 0.4]),
                                                  np.array([[1.3, 0.92, 0.4], [1.4, 1., 0.4]]))
        print(distances)

    def calculate_distance_batch(self, goal_pos, current_pos_batch):
        d, _ = self.graph.get_dist_batch(goal_pos, current_pos_batch)
        indices_inf = d==np.inf
        d[indices_inf] = 9999
        return d


# Works for estimations in 2D plane
class MultipleObstacle(DistMovEst):
    def __init__(self):
        self.max_x = None
        self.min_x = None
        self.max_y = None
        self.min_y = None
        self.x_mid = None
        self.y_mid = None
        self.s_x = None
        self.s_y = None
        self.update_calls = 0
        self.update_complete = False
        self.real = True
        self.none_vals_yet = True

    def initialize_internal_distance_graph(self, field, num_vertices, size_increase):
        obstacles = []
        for i in range(len(self.x_mid)):
            size_x = self.max_x[i] - self.x_mid[i]
            size_y = self.max_y[i] - self.y_mid[i]
            obstacles.append([self.x_mid[i], self.y_mid[i], size_x, size_y])
        self.obstacles = obstacles
        graph = DistanceGraph2D(args=None, field=field, num_vertices=num_vertices,
                                obstacles=obstacles, size_increase=size_increase, use_discrete=False)
        graph.compute_cs_graph()
        graph.compute_dist_matrix()
        self.graph = graph


    def calculate_distance_batch(self, goal_pos, current_pos_batch):
        B = len(current_pos_batch)
        distances = []
        for i in range(B):
            d, _ = self.graph.get_dist(goal_pos, current_pos_batch[i])
            if d == np.inf:
                d = 9999
            distances.append(d)
        distances = np.array(distances)
        return distances


class MultipleDistReal(MultipleObstacle):#in same principle the same but for every single obstacle

    def update(self, obstacle_latent_list, obstacle_size_latent_list):
        #obstacle_size_latent will be actually empty
        if not isinstance(obstacle_size_latent_list, np.ndarray):
            obstacle_array = np.array(obstacle_latent_list)
        else:
            obstacle_array = obstacle_size_latent_list

        #shape of obstacle size latent (B, N, D); B number of samples; N number of obstacles, D dimension of attribute
        #(B, N, D) -> (N, B, D)
        obstacle_array = np.transpose(obstacle_array, axes=[1, 0, 2])
        N, B, D = obstacle_array.shape
        #each object info is organized as [x,y,z,size_x,size_y, size_z]
        a_xs = np.amin(obstacle_array[0:N, :, 0], axis=1)
        b_xs = np.amax(obstacle_array[0:N, :, 0], axis=1)
        a_ys = np.amin(obstacle_array[0:N, :, 1], axis=1)
        b_ys = np.amax(obstacle_array[0:N, :, 1], axis=1)
        s_xs = np.mean(obstacle_array[0:N, :, 3], axis=1)
        s_ys = np.mean(obstacle_array[0:N, :, 4], axis=1)

        if self.none_vals_yet:#assume everything else is
            self.max_x = b_xs + s_xs
            self.min_x = a_xs - s_xs
            self.max_y = b_ys + s_ys
            self.min_y = a_ys - s_ys
            self.s_x = s_xs
            self.s_y = s_ys
            self.none_vals_yet = False
        else:
            indices_to_change = b_xs + s_xs > self.max_x
            self.max_x[indices_to_change] = b_xs[indices_to_change] + s_xs[indices_to_change]
            indices_to_change = a_xs - s_xs < self.min_x
            self.min_x[indices_to_change] = a_xs[indices_to_change] - s_xs[indices_to_change]
            indices_to_change = b_ys + s_ys > self.max_y
            self.max_y[indices_to_change] = b_ys[indices_to_change] + s_ys[indices_to_change]
            indices_to_change = a_ys - s_ys < self.min_y
            self.min_y[indices_to_change] = a_ys[indices_to_change] - s_ys[indices_to_change]

        self.x_mid = (self.max_x + self.min_x) / 2.0
        self.y_mid = (self.max_y + self.min_y) / 2.0


class MultipleDist(MultipleObstacle):
    def update(self, obstacle_latent_list, obstacle_size_latent_list):
        #todo handle case when obstacle not present
        if not isinstance(obstacle_latent_list, np.ndarray):
            obstacle_array = np.array(obstacle_latent_list)
        else:
            obstacle_array = obstacle_latent_list

        if not isinstance(obstacle_size_latent_list, np.ndarray):
            obstacle_array_size = np.array(obstacle_size_latent_list)
        else:
            obstacle_array_size = obstacle_size_latent_list

        # shape of obstacle size latent (B, N, D); B number of samples; N number of obstacles, D dimension of attribute
        # (B, N, D) -> (N, B, D)
        obstacle_array = np.transpose(obstacle_array, axes=[1, 0, 2])
        obstacle_array_size = np.transpose(obstacle_array_size, axes=[1, 0, 2])
        N, B, D = obstacle_array.shape
        # each object info is organized as [x,y,z,size_x,size_y, size_z]
        a_xs = np.amin(obstacle_array[0:N, :, 0], axis=1)
        b_xs = np.amax(obstacle_array[0:N, :, 0], axis=1)
        a_ys = np.amin(obstacle_array[0:N, :, 1], axis=1)
        b_ys = np.amax(obstacle_array[0:N, :, 1], axis=1)
        s_xs = np.mean(obstacle_array_size[0:N, :, 0], axis=1)
        s_ys = np.mean(obstacle_array_size[0:N, :, 1], axis=1)

        if self.none_vals_yet:  # assume everything else is
            self.max_x = b_xs + s_xs
            self.min_x = a_xs - s_xs
            self.max_y = b_ys + s_ys
            self.min_y = a_ys - s_ys
            self.s_x = s_xs
            self.s_y = s_ys
            self.none_vals_yet = False
        else:
            indices_to_change = b_xs + s_xs > self.max_x
            self.max_x[indices_to_change] = b_xs[indices_to_change] + s_xs[indices_to_change]
            indices_to_change = a_xs - s_xs < self.min_x
            self.min_x[indices_to_change] = a_xs[indices_to_change] - s_xs[indices_to_change]
            indices_to_change = b_ys + s_ys > self.max_y
            self.max_y[indices_to_change] = b_ys[indices_to_change] + s_ys[indices_to_change]
            indices_to_change = a_ys - s_ys < self.min_y
            self.min_y[indices_to_change] = a_ys[indices_to_change] - s_ys[indices_to_change]

        self.x_mid = (self.max_x + self.min_x) / 2.0
        self.y_mid = (self.max_y + self.min_y) / 2.0

class Estimator_DistNet:
    def __init__(self, net_weights_path, csv_dist_filepath, device='cuda:0'):
        data = pd.read_csv(csv_dist_filepath)
        self.max = data['max'][0]
        self.min = data['min'][0]
        self.mean_input = torch.from_numpy(
            label_str_to_numpy(data['mean_input'][0])
        ).float().to(device)
        self.std_input = torch.from_numpy(
            label_str_to_numpy(data['std_input'][0])
        ).float().to(device)
        self.mean_output = data['mean_output'][0]
        self.std_output = data['std_output'][0]
        self.input_size = data['input_size'][0]
        self.device=device
        
        self.model = load_DistNet_model(net_weights_path, device, input_size=self.input_size, output_size=1,
                                        val_infinite= self.max + 1.)
        self.model.eval()

    #Net receives normalized input
    def _normalize_input(self, x):
        x = x - self.mean_input
        x = x / self.std_output
        return x

    #Net returns normalized output
    def _denormalize_output(self, x):
        x = x * self.std_output
        x = x + self.mean_output
        return x


    #for points unreachable Net returns max + 1; this must be replaced with 9999; first do denormalization
    def _repos_dist(self, dist):
        difference = torch.abs(dist - (self.max + 1))
        dist[difference <= 1e-12] = 9999
        return dist
    
    #todo manually handle cases outside the region
    def calculate_distance_batch(self, goal_pos, current_pos_batch, bboxes_list_batch):
        B = len(current_pos_batch)
        goal_pos_as_batch = np.stack([goal_pos]*B, axis=0)
        ppairs = np.concatenate([goal_pos_as_batch, current_pos_batch], axis=1)
        bboxes_list_batch_flat = np.reshape(bboxes_list_batch, (B, -1))
        net_input = np.concatenate([bboxes_list_batch_flat, ppairs], axis=1)

        with torch.no_grad():
            net_input = torch.from_numpy(net_input).float().to(self.device)
            net_input = self._normalize_input(net_input)
            d_dict = self.model(net_input)
            d = d_dict['distance']
            d = self._denormalize_output(d)
            d = self._repos_dist(d)
            d = d.cpu().numpy()
        return d