import numpy as np
import copy
'''import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import torch
from common import get_args
from envs import make_env
from j_vae.vae_mix_test import setup_env_sizes, take_goal_image, take_obstacle_image
from j_vae.generate_vae_data import random_pos_inside, size_file, min_obstacle_size, max_obstacle_size, random_size_at, \
    range_x, range_y, obstacle_size, puck_size, generate_points, z_table_height
from j_vae.latent_space_transformations import obstacle_transformation, goal_transformation, get_size_in_space, \
    from_real_pos_to_range, from_real_radius_to_range


vae_model_obstacle = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_obstacle')
vae_model_goal = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_goal')
img_size = 84
vae_model_size = load_Vae(path='../data/FetchPushObstacle/vae_model_sizes', img_size=img_size, latent_size=1)'''


def check_for_collision(g0, g1, o, r_o):
    l = np.linalg.norm(g1 - g0)
    if l == 0:
        return False
    u = np.dot(o-g0, g1-g0)/(l**2)
    e = g0 + np.clip(a=u, a_min=0, a_max=1)*(g1 - g0)
    return np.linalg.norm(e - o) < r_o


def check_for_collision_batch(g0, g1, o, r_o):
    v0 = o-g0
    v1 = g1-g0
    vm = v0 * v1
    dots = np.sum(vm, axis=1)
    ls = np.linalg.norm(g1 - g0, axis=1)
    same_points = ls == 0.0
    ls[same_points] = 1.
    u = dots/(ls**2)
    u[same_points] = 0.0
    clipped = np.expand_dims(np.clip(a=u, a_min=0, a_max=1), axis=1)
    e = g0 + clipped*(g1 - g0)
    b = np.linalg.norm(e - o, axis=1) < r_o
    b[same_points] = False
    return b


def calculate_distance(obstacle_pos, obstacle_radius, current_pos, goal_pos, range_x, range_y, extra_r=0.001):
    # todo see how to measure distance if route out of range
    obstacle_radius = copy.copy(obstacle_radius) + extra_r
    if not check_for_collision(current_pos, goal_pos, obstacle_pos, obstacle_radius):
        return np.linalg.norm(current_pos - goal_pos)

    current_pos = push_check_to_edge(current_pos, obstacle_pos, obstacle_radius)
    goal_pos = push_check_to_edge(goal_pos, obstacle_pos, obstacle_radius)

    a1_curr, a2_curr = get_tangent_points(current_pos, obstacle_pos, obstacle_radius)
    a1_goal, a2_goal = get_tangent_points(goal_pos, obstacle_pos, obstacle_radius)

    if np.linalg.norm(a1_curr - a1_goal) < np.linalg.norm(a1_curr - a2_goal):
        t1_c, t1_g = a1_curr, a1_goal
        t2_c, t2_g = a2_curr, a2_goal
    else:
        t1_c, t1_g = a1_curr, a2_goal
        t2_c, t2_g = a2_curr, a1_goal

    arc1_length = get_arc_length(t1_c, t1_g, obstacle_pos, obstacle_radius)
    arc2_length = get_arc_length(t2_c, t2_g, obstacle_pos, obstacle_radius)

    d1 = np.linalg.norm(current_pos - t1_c) + arc1_length + np.linalg.norm(t1_g - goal_pos)
    d2 = np.linalg.norm(current_pos - t2_c) + arc2_length + np.linalg.norm(t2_g - goal_pos)
    return min(d1, d2)

def calculate_distance_real(obstacle_pos, obstacle_radius, current_pos, goal_pos, range_x, range_y, extra_r=0.001):
    # todo see how to measure distance if route out of range
    obstacle_radius = copy.copy(obstacle_radius) + extra_r
    if not check_for_collision(current_pos, goal_pos, obstacle_pos, obstacle_radius):
        return np.linalg.norm(current_pos - goal_pos)

    current_pos = push_check_to_edge(current_pos, obstacle_pos, obstacle_radius)
    goal_pos = push_check_to_edge(goal_pos, obstacle_pos, obstacle_radius)

    a1_curr, a2_curr = get_tangent_points(current_pos, obstacle_pos, obstacle_radius)
    a1_goal, a2_goal = get_tangent_points(goal_pos, obstacle_pos, obstacle_radius)

    if np.linalg.norm(a1_curr - a1_goal) < np.linalg.norm(a1_curr - a2_goal):
        t1_c, t1_g = a1_curr, a1_goal
        t2_c, t2_g = a2_curr, a2_goal
    else:
        t1_c, t1_g = a1_curr, a2_goal
        t2_c, t2_g = a2_curr, a1_goal

    arc1_length = get_arc_length(t1_c, t1_g, obstacle_pos, obstacle_radius)
    arc2_length = get_arc_length(t2_c, t2_g, obstacle_pos, obstacle_radius)

    d1 = np.linalg.norm(current_pos - t1_c) + arc1_length + np.linalg.norm(t1_g - goal_pos)
    d2 = np.linalg.norm(current_pos - t2_c) + arc2_length + np.linalg.norm(t2_g - goal_pos)
    return min(d1, d2)


# https://doubleroot.in/lessons/circle/tangent-from-an-external-point-2/
def get_tangent_points_batch(p_batch, center, r):
    p_o_v = center - p_batch
    ls = np.linalg.norm(p_o_v, axis=1)
    theta_p = np.arcsin(r / ls)

    dir_a1 = np.concatenate([np.expand_dims(np.cos(theta_p) * p_o_v[:, 0] - np.sin(theta_p) * p_o_v[:, 1],
                                            axis=1),
                             np.expand_dims(np.sin(theta_p) * p_o_v[:, 0] + np.cos(theta_p) * p_o_v[:, 1],
                                            axis=1)],
                            axis=1)
    dir_a1 = dir_a1 / np.expand_dims(np.linalg.norm(dir_a1, axis=1), axis=1)
    dir_a2 = np.concatenate([np.expand_dims(np.cos(-theta_p) * p_o_v[:, 0] - np.sin(-theta_p) * p_o_v[:, 1],
                                            axis=1),
                             np.expand_dims(np.sin(-theta_p) * p_o_v[:, 0] + np.cos(-theta_p) * p_o_v[:, 1],
                                            axis=1)],
                            axis=1)
    dir_a2 = dir_a2 / np.expand_dims(np.linalg.norm(dir_a2, axis=1), axis=1)
    p_a_l = np.expand_dims(np.sqrt(np.power(ls, 2) - r**2), axis=1)

    a1 = p_batch + dir_a1 * p_a_l
    a2 = p_batch + dir_a2 * p_a_l
    return a1, a2


def get_tangent_points(p, center, r):
    p_o_v = center - p
    l = np.linalg.norm(p_o_v)
    theta_p = np.arcsin(r / l)
    theta_p = np.squeeze(theta_p)
    dir_a1 = np.array([np.cos(theta_p) * p_o_v[0] - np.sin(theta_p) * p_o_v[1],
                       np.sin(theta_p) * p_o_v[0] + np.cos(theta_p) * p_o_v[1]])
    dir_a1 = dir_a1 / np.linalg.norm(dir_a1)
    dir_a2 = np.array([np.cos(-theta_p) * p_o_v[0] - np.sin(-theta_p) * p_o_v[1],
                       np.sin(-theta_p) * p_o_v[0] + np.cos(-theta_p) * p_o_v[1]])
    dir_a2 = dir_a2 / np.linalg.norm(dir_a2)
    p_a_l = np.sqrt(l**2 - r**2)
    a1 = p + dir_a1*p_a_l
    a2 = p + dir_a2 * p_a_l
    return a1, a2

def get_arc_length(p1, p2, center, r):
    p1v = p1 - center
    p2v = p2 - center
    angle = np.arccos(np.dot(p1v, p2v) / (np.linalg.norm(p1v) * np.linalg.norm(p2v)))
    return angle * r

#https://tutors.com/math-tutors/geometry-help/how-to-find-arc-measure-formula
#https://en.wikipedia.org/wiki/Dot_product
def get_arc_length_batch(p1_batch, p2, center, r):
    p1v_batch = p1_batch - center
    p2v = p2 - center
    dots = np.dot(p1v_batch, p2v)
    ls = np.linalg.norm(p1v_batch, axis=1) * np.linalg.norm(p2v)
    div = dots/ls
    div = np.clip(div, a_min=-1.0, a_max=1.0)#todo verify if it is really just correcting small cases
    angle = np.arccos(div)
    return angle * r


def push_check_to_edge(p, center, r):
    o_p_v = p - center
    l = np.linalg.norm(o_p_v)
    if l >= r:
        return p
    else:
        dist_to_move = r - l
        # added some distance due to numerically inaccuracy
        dist_to_move += 0.0001
        dir = o_p_v / l
        a = p + dir * dist_to_move
        return a


def push_check_to_edge_batch(p_batch, center, r):
    o_p_v = p_batch - center
    ls = np.linalg.norm(o_p_v, axis=1)
    dist_to_move = r - ls
    already_outside = dist_to_move <= 0
    # added some distance due to numerically inaccuracy
    dist_to_move += 0.0001
    #those outside must not be moved
    dist_to_move[already_outside] = 0.0
    dir = o_p_v / np.expand_dims(ls, axis=1)
    a = p_batch + dir * np.expand_dims(dist_to_move, axis=1)
    return a


def calculate_distance_batch(obstacle_pos, obstacle_radius, current_pos_batch, goal_pos, range_x, range_y,
                             extra_r=0.001):

    #todo see how to measure distance if route out of range
    obstacle_radius = copy.copy(obstacle_radius) + extra_r
    b_v = check_for_collision_batch(current_pos_batch, goal_pos, obstacle_pos, obstacle_radius)

    to_calc_batch = current_pos_batch[b_v]
    to_calc_batch = push_check_to_edge_batch(to_calc_batch, obstacle_pos, obstacle_radius)
    goal_pos = push_check_to_edge(goal_pos, obstacle_pos, obstacle_radius)

    a1_curr, a2_curr = get_tangent_points_batch(to_calc_batch, obstacle_pos, obstacle_radius)
    a1_goal, a2_goal = get_tangent_points(goal_pos, obstacle_pos, obstacle_radius)

    arc1_1_length = get_arc_length_batch(a1_curr, a1_goal, obstacle_pos, obstacle_radius)
    d1_1 = np.linalg.norm(to_calc_batch - a1_curr, axis=1) + arc1_1_length + np.linalg.norm(a1_goal - goal_pos)
    arc2_2_length = get_arc_length_batch(a2_curr, a2_goal, obstacle_pos, obstacle_radius)
    d2_2 = np.linalg.norm(to_calc_batch - a2_curr, axis=1) + arc2_2_length + np.linalg.norm(a2_goal - goal_pos)
    arc1_2_length = get_arc_length_batch(a1_curr, a2_goal, obstacle_pos, obstacle_radius)
    d1_2 = np.linalg.norm(to_calc_batch - a1_curr, axis=1) + arc1_2_length + np.linalg.norm(a2_goal - goal_pos)
    arc2_1_length = get_arc_length_batch(a2_curr, a1_goal, obstacle_pos, obstacle_radius)
    d2_1 = np.linalg.norm(to_calc_batch - a2_curr, axis=1) + arc2_1_length + np.linalg.norm(a1_goal - goal_pos)
    d = np.array([d1_1, d2_2, d1_2, d2_1]).min(axis=0)

    dist = np.zeros(len(current_pos_batch))
    dist[b_v] = d
    not_colliding = current_pos_batch[np.logical_not(b_v)]
    not_colliding_dist = np.linalg.norm(not_colliding - goal_pos, axis=1)
    dist[np.logical_not(b_v)] = not_colliding_dist

    return dist


from scipy.stats import uniform
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
            if b_x + self.s_x > self.max_x:
                self.max_x = b_x + self.s_x
            if a_x - self.s_x < self.min_x:
                self.min_x = a_x - self.s_x
            if b_y + self.s_x > self.max_y:
                self.max_y = b_y + self.s_x
            if a_y - self.s_x < self.min_y:
                self.min_y = a_y - self.s_x

        self.x_mid = np.mean([self.max_x, self.min_x])
        self.y_mid = np.mean([self.max_y, self.min_y])
        self.update_calls += 1
        if self.update_calls == 3:
            self.update_complete = True

    def update_sizes(self, obstacle_latent_list, goal_latent_list):
        obstacle_latents = np.array(obstacle_latent_list)
        goal_latents = np.array(goal_latent_list)

        diffs_x = np.abs(goal_latents[:, 0] - obstacle_latents[:, 0])
        diffs_y = np.abs(goal_latents[:, 1] - obstacle_latents[:, 1])
        x_smaller = diffs_x < self.s_x
        y_smaller = diffs_y < self.s_y
        both = np.logical_and(x_smaller, y_smaller)
        x_correction_candidates = diffs_x[both]
        if len(x_correction_candidates) > 0:
            s_min_x = np.min(x_correction_candidates)
            print('updating x to: {}'.format(s_min_x))
            self.max_x = self.max_x - self.s_x + s_min_x
            self.min_x = self.min_x + self.s_x - s_min_x
            self.x_mid = np.mean([self.max_x, self.min_x])
            self.s_x = s_min_x.copy()
        y_correction_candidates = diffs_y[both]
        if len(y_correction_candidates) > 0:
            s_min_y = np.min(y_correction_candidates)
            print('updating y to: {}'.format(s_min_y))
            self.max_y = self.max_y - self.s_y + s_min_y
            self.min_y = self.min_y + self.s_y - s_min_y
            self.y_mid = np.mean([self.max_y, self.min_y])
            self.s_y = s_min_y.copy()

    def calculate_distance_batch(self, goal_pos, current_pos_batch):
        #todo generalize this even more
        # http://www.jeffreythompson.org/collision-detection/line-line.php
        def lines_to_line_intersection(xs1, ys1, x2, y2, x3, y3, x4, y4):
            #todo correct those ones that are being divided by 0 RuntimeWarning: divide by zero encountered in true_divide
            nA = ((x4 - x3) * (ys1 - y3) - (y4 - y3) * (xs1 - x3))
            dA = ((y4 - y3) * (x2 - xs1) - (x4 - x3) * (y2 - ys1))
            zA = dA == 0.
            not_zA = np.logical_not(zA)
            prcA = np.zeros(shape=len(xs1))
            prcA[not_zA] = nA[not_zA]/dA[not_zA]
            prcA[zA] = 2.#just to make it false

            nB = ((x2 - xs1) * (ys1 - y3) - (y2 - ys1) * (xs1 - x3))
            dB = ((y4 - y3) * (x2 - xs1) - (x4 - x3) * (y2 - ys1))
            zB = dB == 0.
            not_zB = np.logical_not(zB)
            prcB = np.zeros(shape=len(xs1))
            prcB[not_zB] =  nB[not_zB]/dB[not_zB]
            prcB[zB] = 2.

            bA = np.logical_and(0 <= prcA, prcA <=1)
            bB = np.logical_and(0 <= prcB, prcB <=1)
            return np.logical_and(bA, bB)

        to_left = lines_to_line_intersection(current_pos_batch[:, 0], current_pos_batch[:, 1], goal_pos[0], goal_pos[1],
                                             self.min_x, self.min_y, self.max_x, self.min_y)
        to_right = lines_to_line_intersection(current_pos_batch[:, 0], current_pos_batch[:, 1], goal_pos[0], goal_pos[1],
                                              self.min_x, self.max_y, self.max_x, self.max_y)
        #todo should also do for top and bottom
        inside = np.logical_or(to_left, to_right)
        not_inside = np.logical_not(inside)
        distances = np.zeros(shape=len(current_pos_batch))

        direct_pos = current_pos_batch[not_inside]
        direct_distances = np.linalg.norm(direct_pos - goal_pos, axis=1)
        distances[not_inside] = direct_distances

        undr_pos = current_pos_batch[inside]
        #todo this must be more general
        top = np.array([self.max_x + 0.0001*self.s_x, self.y_mid])
        through_top = np.linalg.norm(undr_pos - top, axis=1) + np.linalg.norm(goal_pos - top)
        bottom = np.array([self.min_x - 0.0001*self.s_x, self.y_mid])
        through_bottom = np.linalg.norm(undr_pos - bottom, axis=1) + np.linalg.norm(goal_pos - bottom)
        min_d = np.minimum(through_top, through_bottom)
        distances[inside] = min_d
        return distances


        '''to_g_v = goal_point - points
        distance = np.linalg.norm(to_g_v, axis=1)
        normalized = to_g_v / distance
        #to y
        to_my_v = self.y_mid - points[:, 1]
        dist_prc = to_my_v /

        x_p = None
        x_inside = self.prob_dstr_x(x_p) > 0
        y_p = None
        y_inside = self.prob_dstr_y(y_p) > 0
        inside = np.logical_and(x_inside, y_inside)
        ds = np.linalg.norm(to_g_v, axis=1)

        #todo
        ds[inside] = modified_distances
        pass'''

    def prob_dstr_x(self, x_coordinates):
        assert self.min_x is None and self.max_x is not None
        prs = uniform.pdf(x_coordinates, loc=self.min_x, scale=self.max_x - self.min_x)
        return prs

    def prob_dstr_y(self, y_coordinates):
        assert self.min_y is None and self.max_y is not None
        prs = uniform.pdf(y_coordinates, loc=self.min_y, scale=self.max_y - self.min_y)
        return prs

class DistMovEstReal(DistMovEst):
    def __init__(self):
        super(DistMovEstReal, self).__init__()
        self.real = True
        self.s_x = 0.09
        self.s_y = 0.03

    def update(self, obstacle_real_list, obstacle_size_real_list):
        obstacle_real_array = np.array(obstacle_real_list)
        obstacle_size_real_array = np.array(obstacle_size_real_list)
        a_x = np.min(obstacle_real_array[:, 0])
        b_x = np.max(obstacle_real_array[:, 0])
        a_y = np.min(obstacle_real_array[:, 1])
        b_y = np.max(obstacle_real_array[:, 1])
        if self.min_x == None:  # assume everything else is
            self.max_x = b_x + self.s_x
            self.min_x = a_x - self.s_x
            self.max_y = b_y + self.s_y
            self.min_y = a_y - self.s_y
        else:
            if b_x + self.s_x > self.max_x:
                self.max_x = b_x + self.s_x
            if a_x - self.s_x < self.min_x:
                self.min_x = a_x - self.s_x
            if b_y + self.s_x > self.max_y:
                self.max_y = b_y + self.s_x
            if a_y - self.s_x < self.min_y:
                self.min_y = a_y - self.s_x

        self.x_mid = np.mean([self.max_x, self.min_x])
        self.y_mid = np.mean([self.max_y, self.min_y])
        self.update_calls += 1
        if self.update_calls == 3:
            self.update_complete = True

    def update_sizes(self, obstacle_real_list, goal_real_list):
        pass

    def calculate_distance_batch(self, goal_pos, current_pos_batch):
        #todo generalize this even more
        # http://www.jeffreythompson.org/collision-detection/line-line.php
        def lines_to_line_intersection(xs1, ys1, x2, y2, x3, y3, x4, y4):
            #todo correct those ones that are being divided by 0 RuntimeWarning: divide by zero encountered in true_divide
            nA = ((x4 - x3) * (ys1 - y3) - (y4 - y3) * (xs1 - x3))
            dA = ((y4 - y3) * (x2 - xs1) - (x4 - x3) * (y2 - ys1))
            zA = dA == 0.
            not_zA = np.logical_not(zA)
            prcA = np.zeros(shape=len(xs1))
            prcA[not_zA] = nA[not_zA]/dA[not_zA]
            prcA[zA] = 2.#just to make it false

            nB = ((x2 - xs1) * (ys1 - y3) - (y2 - ys1) * (xs1 - x3))
            dB = ((y4 - y3) * (x2 - xs1) - (x4 - x3) * (y2 - ys1))
            zB = dB == 0.
            not_zB = np.logical_not(zB)
            prcB = np.zeros(shape=len(xs1))
            prcB[not_zB] =  nB[not_zB]/dB[not_zB]
            prcB[zB] = 2.

            bA = np.logical_and(0 <= prcA, prcA <=1)
            bB = np.logical_and(0 <= prcB, prcB <=1)
            return np.logical_and(bA, bB)

        to_left = lines_to_line_intersection(current_pos_batch[:, 0], current_pos_batch[:, 1], goal_pos[0], goal_pos[1],
                                             self.min_x, self.min_y, self.max_x, self.min_y)
        to_right = lines_to_line_intersection(current_pos_batch[:, 0], current_pos_batch[:, 1], goal_pos[0], goal_pos[1],
                                              self.min_x, self.max_y, self.max_x, self.max_y)
        #todo should also do for top and bottom
        inside = np.logical_or(to_left, to_right)
        not_inside = np.logical_not(inside)
        distances = np.zeros(shape=len(current_pos_batch))

        direct_pos = current_pos_batch[not_inside]
        direct_distances = np.linalg.norm(direct_pos - goal_pos, axis=1)
        distances[not_inside] = direct_distances

        undr_pos = current_pos_batch[inside]
        #todo this must be more general
        top = np.array([self.max_x + 0.001*self.s_x, self.y_mid])
        if goal_pos.shape[0] > 2:
            goal_pos= goal_pos[:2]
        if len(undr_pos) > 0 and undr_pos.shape[1] == 3:
            undr_pos = undr_pos[:, :2]

        through_top = np.linalg.norm(undr_pos - top, axis=1) + np.linalg.norm(goal_pos - top)
        bottom = np.array([self.min_x - 0.001*self.s_x, self.y_mid])
        through_bottom = np.linalg.norm(undr_pos - bottom, axis=1) + np.linalg.norm(goal_pos - bottom)
        min_d = np.minimum(through_top, through_bottom)
        distances[inside] = min_d
        return distances

    def prob_dstr_x(self, x_coordinates):
        pass

    def prob_dstr_y(self, y_coordinates):
        pass



'''if __name__ == '__main__':
    ps = np.array([[1.3, 0.],[1., 1.], [2., 2.5]])
    center = np.array([0.0, 0.0])
    r = 1.5
    push_check_to_edge(ps[2], center, r)
    print('h')'''


'''def calc_show_path(obstacle_pos, obstacle_radius, current_pos, goal_pos):
    extra_r = 0.001
    obstacle_radius = copy.copy(obstacle_radius) + extra_r
    if not check_for_collision(current_pos, goal_pos, obstacle_pos, obstacle_radius):
        return np.linalg.norm(current_pos - goal_pos)

    def get_tangent_points(p, center, r):
        p_o_v = center - p
        theta_p = np.arcsin(r / np.linalg.norm(p_o_v))
        dir_a1 = np.array([np.cos(theta_p) * p_o_v[0] - np.sin(theta_p) * p_o_v[1],
                           np.sin(theta_p) * p_o_v[0] + np.cos(theta_p) * p_o_v[1]])
        dir_a1 = dir_a1 / np.linalg.norm(dir_a1)
        dir_a2 = np.array([np.cos(-theta_p) * p_o_v[0] - np.sin(-theta_p) * p_o_v[1],
                           np.sin(-theta_p) * p_o_v[0] + np.cos(-theta_p) * p_o_v[1]])
        dir_a2 = dir_a2 / np.linalg.norm(dir_a2)
        p_a_l = np.sqrt(np.linalg.norm(p_o_v)**2 - r_o**2)
        a1 = p + dir_a1*p_a_l
        a2 = p + dir_a2 * p_a_l
        return a1, a2

    def get_arc_length(p1, p2, center, r):
        p1v = p1 - center
        p2v = p2 - center
        angle = np.arccos(np.dot(p1v, p2v)/(np.linalg.norm(p1v)*np.linalg.norm(p2v)))

        t1 = np.arctan2(p1v[1], p1v[0]) * 180 / np.pi
        t2 = np.arctan2(p2v[1], p2v[0]) * 180 / np.pi

        if t1 < t2:
            t_min = t1
            t_max = t2
        else:
            t_min = t2
            t_max = t1

        arc = Arc((center[0], center[1]), width=2*r, height=2*r,
                  theta1=t_min, theta2=t_max)
        ax.add_patch(arc)

        return angle * r

    a1_curr, a2_curr = get_tangent_points(current_pos, obstacle_pos, obstacle_radius)
    a1_goal, a2_goal = get_tangent_points(goal_pos, obstacle_pos, obstacle_radius)

    ax.scatter([a1_curr[0], a2_curr[0], a1_goal[0], a2_goal[0]],
               [a1_curr[1], a2_curr[1], a1_goal[1], a2_goal[1]], c='green')

    if np.linalg.norm(a1_curr - a1_goal) < np.linalg.norm(a1_curr - a2_goal):
        t1_c, t1_g = a1_curr, a1_goal
        t2_c, t2_g = a2_curr, a2_goal
    else:
        t1_c, t1_g = a1_curr, a2_goal
        t2_c, t2_g = a2_curr, a1_goal

    arc1_length = get_arc_length(t1_c, t1_g, obstacle_pos, obstacle_radius)
    arc2_length = get_arc_length(t2_c, t2_g, obstacle_pos, obstacle_radius)

    d1 = np.linalg.norm(current_pos - t1_c) + arc1_length + np.linalg.norm(t1_g - goal_pos)
    plt.plot([current_pos[0], t1_c[0]], [current_pos[1], t1_c[1]], color="black")
    plt.plot([goal_pos[0], t1_g[0]], [goal_pos[1], t1_g[1]], color="black")

    d2 = np.linalg.norm(current_pos - t2_c) + arc2_length + np.linalg.norm(t2_g - goal_pos)
    plt.plot([current_pos[0], t2_c[0]], [current_pos[1], t2_c[1]], color="black")
    plt.plot([goal_pos[0], t2_g[0]], [goal_pos[1], t2_g[1]], color="black")
    return min(d1, d2)


if __name__ == '__main__':
    args = get_args()
    env = make_env(args)


    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    space_extra = 0.2#0.08
    setup_env_sizes(env)
    p_o = np.array(random_pos_inside(range_x=[range_x[0]+space_extra, range_x[1]-space_extra],
                                     range_y=[range_y[0]+space_extra, range_y[1]-space_extra], z=0.435,
                                     object_x_y_size=[obstacle_size, obstacle_size]))

    goal_p = p_o - np.array([obstacle_size, 0, 0]) - \
             np.array([np.random.uniform(0, space_extra), np.random.uniform(0, space_extra), 0.0])
    pos_p = p_o + np.array([obstacle_size, 0, 0]) + \
            np.array([np.random.uniform(0, space_extra), np.random.uniform(0, space_extra), 0.0])

    # sample for obstacle
    # move temporally object to another position
    env.env.env._move_object(position=[2.0, 2.0, 2.0])
    env.env.env._set_position(names_list=['obstacle'], position=p_o)
    im_o = take_obstacle_image(env)
    im_o = torch.from_numpy(im_o).float().to(device)
    im_o /= 255
    im_o = im_o.permute([2, 0, 1])
    im_o = torch.unsqueeze(im_o, dim=0)
    mu_o, logvar_o = vae_model_obstacle.encode(im_o)
    mu_o = mu_o.detach().cpu().numpy()

    #calculate size
    mu_size, logvar_size = vae_model_size.encode(im_o.reshape(-1, img_size*img_size*3))
    mu_size = mu_size.detach().cpu().numpy()

    # sample for goals
    data_set = np.empty([2, img_size, img_size, 3])
    env.env.env._move_object(position=goal_p)
    data_set[0] = take_goal_image(env)
    env.env.env._move_object(position=pos_p)
    data_set[1] = take_goal_image(env)
    data = torch.from_numpy(data_set).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = vae_model_goal.encode(data)
    mu = mu.detach().cpu().numpy()

    transformed_mu_o = obstacle_transformation(mu_o)
    mu_o = transformed_mu_o[0]
    mu = goal_transformation(mu)

    #Show _latent
    r_o = get_size_in_space(mu_size[0][0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    a_circle = plt.Circle((mu_o[0], mu_o[1]), r_o, alpha=0.3)
    ax.add_artist(a_circle)
    ax.scatter(mu[0][0], mu[0][1], c='blue')
    ax.scatter(mu[1][0], mu[1][1], c='blue')
    calc_show_path(mu_o, r_o, mu[1], mu[0])

    tr_r = from_real_radius_to_range(obstacle_size)
    tr_g_p = from_real_pos_to_range(goal_p)
    tr_c_p = from_real_pos_to_range(pos_p)
    tr_o_p = from_real_pos_to_range(p_o)

    b_circle = plt.Circle((tr_o_p[0], tr_o_p[1]), tr_r, alpha=0.7, color='yellow')
    ax.add_artist(b_circle)
    ax.scatter(tr_g_p[0], tr_g_p[1], c='yellow')
    ax.scatter(tr_c_p[0], tr_c_p[1], c='yellow')

    plt.show()
    print('h')'''








