import numpy as np
import copy
from envs.distance_graph import DistanceGraph2D
from scipy.stats import uniform
import pandas as pd
import torch

#It is just simple HGG with euclidean distance. This class is just used to make it work with rest of code.
class NoneTypeEst:
    def __init__(self):
        self.obstacles = []

    def calculate_distance_batch(self, goal_pos, current_pos_batch):
        distances = np.linalg.norm(goal_pos - current_pos_batch, axis=1)
        return distances

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

class SubstractArea(MultipleObstacle):
    def __init__(self):
        self.max_left = None
        self.min_right = None
        self.max_down = None
        self.min_up = None
        self.none_vals_yet = True

    def update(self, obstacle_latent_list, obstacle_size_latent_list):
        max_left, min_right, max_down, min_up = self.get_lims(obstacle_latent_list, obstacle_size_latent_list)
        if self.none_vals_yet:  # assume everything else is
            self.max_left = max_left.copy()
            self.min_right = min_right.copy()
            self.max_down = max_down.copy()
            self.min_up = min_up.copy()
            self.none_vals_yet = False
        else:
            indices_to_change = max_left > self.max_left
            self.max_left[indices_to_change] = max_left[indices_to_change]
            indices_to_change = min_right < self.min_right
            self.min_right[indices_to_change] = min_right[indices_to_change]
            indices_to_change = max_down > self.max_down
            self.max_down[indices_to_change] = max_down[indices_to_change]
            indices_to_change = min_up < self.min_up
            self.min_up[indices_to_change] = min_up[indices_to_change]

    def get_lims(self, obstacle_latent_list, obstacle_size_latent_list):
        pass

    def initialize_internal_distance_graph(self, field, num_vertices, size_increase):
        obstacles = []
        for i in range(len(self.max_left)):
            max_left = self.max_left[i]
            min_right = self.min_right[i]
            max_down = self.max_down[i]
            min_up = self.min_up[i]
            if (min_right > max_left) and (min_up > max_down):
                mid_x = (min_right + max_left) / 2.
                mid_y = (min_up + max_down) / 2.
                size_x = (min_right - max_left) / 2.
                size_y = (min_up - max_down) / 2.
                obstacles.append([mid_x, mid_y, size_x, size_y])
        self.obstacles = obstacles
        graph = DistanceGraph2D(args=None, field=field, num_vertices=num_vertices,
                                obstacles=obstacles, size_increase=size_increase, use_discrete=False)
        graph.compute_cs_graph()
        graph.compute_dist_matrix()
        self.graph = graph


class SubstReal(SubstractArea):
    def get_lims(self, obstacle_latent_list, obstacle_size_latent_list):
        # obstacle_size_latent will be actually empty
        if not isinstance(obstacle_size_latent_list, np.ndarray):
            obstacle_array = np.array(obstacle_latent_list)
        else:
            obstacle_array = obstacle_size_latent_list

        if obstacle_array.ndim == 2:
            obstacle_array = np.expand_dims(obstacle_array, axis=0)

        else:
            # shape of obstacle size latent (B, N, D); B number of samples; N number of obstacles, D dimension of attribute
            # (B, N, D) -> (N, B, D)
            obstacle_array = np.transpose(obstacle_array, axes=[1, 0, 2])
        N, B, D = obstacle_array.shape

        # each object info is organized as [x,y,z,size_x,size_y, size_z]
        s_xs = np.expand_dims(np.mean(obstacle_array[0:N, :, 3], axis=1), axis=1)
        s_ys = np.expand_dims(np.mean(obstacle_array[0:N, :, 4], axis=1), axis=1)
        left_s = obstacle_array[0:N, :, 0] - s_xs
        down_s = obstacle_array[0:N, :, 1] - s_ys
        up_s = obstacle_array[0:N, :, 1] + s_ys
        right_s = obstacle_array[0:N, :, 0] + s_xs

        max_left = np.amax(left_s, axis=1)
        min_right = np.amin(right_s, axis=1)
        max_down = np.amax(down_s, axis=1)
        min_up = np.amin(up_s, axis=1)
        return max_left, min_right, max_down, min_up

class Subst(SubstractArea):
    def get_lims(self, obstacle_latent_list, obstacle_size_latent_list):
        if not isinstance(obstacle_latent_list, np.ndarray):
            obstacle_array = np.array(obstacle_latent_list)
        else:
            obstacle_array = obstacle_latent_list

        if not isinstance(obstacle_size_latent_list, np.ndarray):
            obstacle_array_size = np.array(obstacle_size_latent_list)
        else:
            obstacle_array_size = obstacle_size_latent_list

        if obstacle_array.ndim == 2:
            obstacle_array = np.expand_dims(obstacle_array, axis=0)
            obstacle_array_size = np.expand_dims(obstacle_array_size, axis=0)

        else:
            # shape of obstacle size latent (B, N, D); B number of samples; N number of obstacles, D dimension of attribute
            # (B, N, D) -> (N, B, D)
            obstacle_array = np.transpose(obstacle_array, axes=[1, 0, 2])
            obstacle_array_size = np.transpose(obstacle_array_size, axes=[1, 0, 2])

        N, B, D = obstacle_array.shape
        s_xs = np.expand_dims(np.mean(obstacle_array_size[0:N, :, 0], axis=1), axis=1)
        s_ys = np.expand_dims(np.mean(obstacle_array_size[0:N, :, 1], axis=1), axis=1)
        left_s = obstacle_array[0:N, :, 0] - s_xs
        down_s = obstacle_array[0:N, :, 1] - s_ys
        up_s = obstacle_array[0:N, :, 1] + s_ys
        right_s = obstacle_array[0:N, :, 0] + s_xs

        max_left = np.amax(left_s, axis=1)
        min_right = np.amin(right_s, axis=1)
        max_down = np.amax(down_s, axis=1)
        min_up = np.amin(up_s, axis=1)
        return max_left, min_right, max_down, min_up

