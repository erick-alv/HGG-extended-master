import numpy as np
import torch
from j_vae.common_data import obstacle_size

def calculate_angle_goal():
    g_corner_imgs = np.load('../data/FetchPushObstacle/goal_corners.npy')
    from j_vae.train_vae_sb import load_Vae as load_Vae_SB
    vae_model_goal = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_goal')

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    data = torch.from_numpy(g_corner_imgs).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = vae_model_goal.encode(data)
    mu = mu.detach().cpu().numpy()
    #calculates angles from center of coordinate system to corners
    goal_angles = np.arctan2(mu[:, 1], mu[:, 0]) * 180 / np.pi #to degree
    goal_angles = np.array([a if a >=0 else 360 + a for a in goal_angles])
    wanted_angles = np.array([225., 135., 315., 45.])
    to_rotate = wanted_angles - goal_angles
    print(goal_angles[0])

angle_goal = -66.49

def calculate_angle_obstacle():
    o_corner_imgs = np.load('../data/FetchPushObstacle/obstacle_corners.npy')
    from j_vae.train_vae_sb import load_Vae as load_Vae_SB
    vae_model_obstacle = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_obstacle')

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    data = torch.from_numpy(o_corner_imgs).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = vae_model_obstacle.encode(data)
    mu = mu.detach().cpu().numpy()
    '''if reflect_obstacle:
        for i, p in enumerate(mu):
            mu[i] = reflect_obstacle_transformation(p)'''
    # calculates angles from center of coordinate system to corners
    goal_angles = np.arctan2(mu[:, 1], mu[:, 0]) * 180 / np.pi  # to degree
    goal_angles = np.array([a if a >= 0 else 360 + a for a in goal_angles])
    wanted_angles = np.array([225., 135., 315., 45.])
    to_rotate = wanted_angles - goal_angles
    print(goal_angles[0])

angle_obstacle = 67.345


def interval_map_function(a,b,c, d):
    def map(x):
        return c + (d - c)/ (b-a) * (x-a)
    return map

def interval_to_minusone_one(a, b):
    return interval_map_function(a,b, -1.0, 1.0)

table_map_x = interval_to_minusone_one(1.05, 1.55)
table_map_y = interval_to_minusone_one(0.5, 1.0)

g_x_min = table_map_x(1.05+0.015)
g_x_max = table_map_x(1.55-0.015)
g_y_min = table_map_y(0.5+0.015)
g_y_max = table_map_y(1.0-0.015)

o_x_min = table_map_x(1.05+obstacle_size)
o_x_max = table_map_x(1.55-obstacle_size)
o_y_min = table_map_y(0.5+obstacle_size)
o_y_max = table_map_y(1.0-obstacle_size)

goal_map_x = interval_map_function(-1.95111, 2.2392, g_x_min, g_x_max)
goal_map_y = interval_map_function(-2.0518, 1.9019, g_y_min, g_y_max)

obstacle_map_x = interval_map_function(-1.3176, 2.0076, o_x_min, o_x_max)
obstacle_map_y = interval_map_function(-1.842813, 1.33413, o_y_min, o_y_max)


def create_rotation_matrix(angle):
    theta = np.radians(angle)
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return r


def rotate(point, rot_matrix):
    return rot_matrix.dot(point)


def rotate_list_of_points(points, rot_matrix):
    rotated_points = [rotate(p, rot_matrix) for p in points]
    return np.array(rotated_points)


def map_single_point(p, mapper_x, mapper_y):
    return np.array([mapper_x(p[0]), mapper_y(p[1])])


def map_points(points, mapper_x, mapper_y):
    map_x = np.array([mapper_x(p) for p in points[:, 0]])
    map_y = np.array([mapper_y(p) for p in points[:, 1]])
    map_x = np.expand_dims(map_x, axis=1)
    map_y = np.expand_dims(map_y, axis=1)
    mapped = np.concatenate([map_x, map_y], axis=1)
    return mapped


def goal_transformation(points):
    m = create_rotation_matrix(angle_goal)
    rotated = rotate_list_of_points(points, m)
    return map_points(rotated, goal_map_x, goal_map_y)


def torch_goal_transformation(batch_p, device):
    theta = np.radians(angle_goal)
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    m = torch.from_numpy(r).float().to(device)
    n = batch_p.size()[0]
    m = m.unsqueeze(0).repeat(n, 1, 1)
    batch_p = batch_p.unsqueeze(2)
    rotated = torch.bmm(m, batch_p)
    rotated = rotated.squeeze()
    if n == 1:
        rotated = rotated.unsqueeze(0)
    m1 = goal_map_x(rotated[:, 0])
    m1 = torch.unsqueeze(m1, dim=1)
    m2 = goal_map_y(rotated[:, 1])
    m2 = torch.unsqueeze(m2, dim=1)
    s = torch.cat([m1, m2], dim=1)
    return s


def obstacle_transformation(points):
    '''if reflect_obstacle:
        for i, p in enumerate(points):
            points[i] = reflect_obstacle_transformation(p)'''
    m = create_rotation_matrix(angle_obstacle)
    rotated = rotate_list_of_points(points, m)

    return map_points(rotated, obstacle_map_x, obstacle_map_y)


def torch_obstacle_transformation(batch_p, device):
    '''if reflect_obstacle:
        A = inclination_m_obstacle
        B = -1
        C = displacement_b_obstacle
        D = A ** 2 + B ** 2
        n_x = (B ** 2 - A ** 2) * batch_p[:, 0] + 2 * A * B * batch_p[:, 1] - 2 * A * C
        n_x /= D
        n_y = (A ** 2 - B ** 2) * batch_p[:, 1] + 2 * A * B * batch_p[:, 0] - 2 * A * B
        n_y /= D
        batch_p = torch.cat([torch.unsqueeze(n_x, dim=1), torch.unsqueeze(n_y, dim=1)], dim=1).to(device)'''
    theta = np.radians(angle_obstacle)
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    m = torch.from_numpy(r).float().to(device)
    n = batch_p.size()[0]
    m = m.unsqueeze(0).repeat(n, 1, 1)
    batch_p = batch_p.unsqueeze(2)
    rotated = torch.bmm(m, batch_p)
    rotated = rotated.squeeze()
    if n == 1:
        rotated = rotated.unsqueeze(0)
    m1 = obstacle_map_x(rotated[:, 0])
    m1 = torch.unsqueeze(m1, dim=1)
    m2 = obstacle_map_y(rotated[:, 1])
    m2 = torch.unsqueeze(m2, dim=1)
    s = torch.cat([m1, m2], dim=1)
    return s


img_size = 84


def return_min_and_max_from_sizes():
    from j_vae.train_vae import load_Vae
    vae_model_size = load_Vae(path='../data/FetchPushObstacle/vae_model_sizes', img_size=img_size, latent_size=1)
    train_file = '../data/FetchPushObstacle/obstacle_sizes_set.npy'
    data_set = np.load(train_file)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")
    min_val = None
    max_val = None

    for el in data_set:
        data = torch.from_numpy(el).float().to(device)
        data /= 255
        data = data.unsqueeze(0)
        data = data.permute([0, 3, 1, 2])
        data = data.reshape(-1, img_size * img_size * 3)
        mu, logvar = vae_model_size.encode(data)
        mu = mu.detach().cpu().numpy()
        mu = mu[0][0]

        if min_val is None:
            min_val = mu
            max_val = mu
        else:
            if mu < min_val:
                min_val = mu
            if mu > max_val:
                max_val = mu
    print('min val: {} \nmax val: {}'.format(min_val, max_val))


min_latent_size = -11.74#this is best for current size
max_latent_size = 4.702

#min_latent_size = -4.703
#max_latent_size = 4.703

map_size_space = interval_map_function(-11.747,4.703,-4.703, 4.703)#currentlz not used


def get_size_in_space(v, range=[-1, 1]):
    dist = np.abs(max_latent_size-min_latent_size)
    prc = np.abs(v-min_latent_size)/dist
    max_size = np.abs(range[1] - range[0])
    #substracted since space encodes from biggest to smallest
    return (1.- prc)*max_size


def torch_get_size_in_space(v, device, range=[-1, 1]):
    dist = torch.abs(torch.tensor(max_latent_size-min_latent_size).float()).to(device)
    prc = torch.abs(v-min_latent_size)/dist
    max_size = torch.abs(torch.tensor(range[1] - range[0]).float())
    #substracted since space encodes from biggest to smallest
    return (1.- prc)*max_size


def from_real_pos_to_range(pos):
    return map_single_point(pos, table_map_x, table_map_y)


def from_real_radius_to_range(radius, real_range_size = 0.5, latent_range=[-1, 1]):
    prc = radius*2 / real_range_size
    max_radius = np.abs(latent_range[1] - latent_range[0]) / 2
    return prc * max_radius


'''def get_interpolation_points_obstacle():
    g_corner_imgs = np.load('../data/FetchPushObstacle/goal_corners.npy')
    from j_vae.train_vae_sb import load_Vae as load_Vae_SB
    vae_model_goal = load_Vae_SB(path='../data/FetchPushObstacle/vae_sb_model_goal')

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    data = torch.from_numpy(g_corner_imgs).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = vae_model_goal.encode(data)
    mu = mu.detach().cpu().numpy()
    #TODO it is not always in this order!!!
    v1 = mu[1] - mu[0]
    d1 = np.linalg.norm(v1)
    v2 = mu[3] - mu[2]
    d2 = np.linalg.norm(v2)
    p1 = mu[0] + 0.5*d1*v1
    p2 = mu[2] + 0.5*d2*v2
    return p1, p2


def get_line_attrs(p1, p2):
    m = (p2[1] - p1[1])/(p2[0] - p1[0])
    #y = m(x - x_1) + y1 <==> y = m*x - m*x_1 + y1 = m*x + b
    b = -m*p1[0] + p1[1]
    return m, b


reflect_obstacle = True
inclination_m_obstacle = 2.0332017
displacement_b_obstacle = 9.27985


# http://www.sdmath.com/math/geometry/reflection_across_line.html#:~:text=In%20plane%20geometry%2C%20Reflection%20across,to%20the%20axis%20of%20reflection.
def reflect_obstacle_transformation(p):
    A = inclination_m_obstacle
    B = -1
    C = displacement_b_obstacle
    D = A**2 + B**2
    n_x = (B**2 - A**2)*p[0] + 2*A*B*p[1] - 2*A*C
    n_x /= D
    n_y = (A**2 - B**2)*p[1] + 2*A*B*p[0] - 2*A*B
    n_y /= D
    return np.array([n_x, n_y])'''


'''if __name__ == '__main__':
    a = torch.randn(20, 2).float()
    b = a.detach().numpy()

    ra = torch_obstacle_transformation(a, device='cuda')
    rb = obstacle_transformation(b)
    print('h')
    #return_min_and_max_from_sizes()
    p1, p2 = get_interpolation_points_obstacle()
    mm, bb = get_line_attrs(p1, p2)
    print(mm)
    print(bb)'''

if __name__ == '__main__':
    calculate_angle_obstacle()



