import numpy as np
import torch
from j_vae.common_data import obstacle_size, train_file_name, vae_sb_weights_file_name, file_corners_name

def calculate_angle(model, corners_file, ind_1, ind_2):
    corner_imgs = np.load(corners_file)

    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    data = torch.from_numpy(corner_imgs).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    mu, logvar = model.encode(data)
    mu = mu.detach().cpu().numpy()
    #calculates angles from center of coordinate system to corners
    goal_angles = np.arctan2(mu[:, ind_2], mu[:, ind_1]) * 180 / np.pi #to degree
    goal_angles = goal_angles % 360
    wanted_angles = np.array([225., 135., 315., 45.])

    to_rotate = wanted_angles - goal_angles
    to_rotate = to_rotate % 360
    print(np.mean(to_rotate))

angle_goal = 270.46783


angle_obstacle = 35.14


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

goal_map_x = interval_map_function(-1.3009, 1.207909, g_x_min, g_x_max)
goal_map_y = interval_map_function(-1.56553, 1.85339, g_y_min, g_y_max)

obstacle_map_x = interval_map_function(-1.3196, 1.917, o_x_min, o_x_max)
obstacle_map_y = interval_map_function(-1.32474, 1.308284, o_y_min, o_y_max)


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


def torch_goal_transformation(batch_p, device, ind_1, ind_2):
    assert ind_1 != ind_2
    batch_p = torch.cat([batch_p[:, ind_1].unsqueeze(axis=1),batch_p[:, ind_2].unsqueeze(axis=1)], axis=1)
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


def torch_obstacle_transformation(batch_p, device, ind_1, ind_2):
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

    assert ind_1 != ind_2
    batch_p = torch.cat([batch_p[:, ind_1].unsqueeze(axis=1),batch_p[:, ind_2].unsqueeze(axis=1)], axis=1)

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


def print_min_and_max_from_sizes(model, train_file):
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
        mu, logvar = model.encode(data)
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


min_latent_size = -4.2777#this is best for current size
max_latent_size = 12.9686

#min_latent_size = -4.703
#max_latent_size = 4.703

def get_size_in_space(v, range=[-1, 1]):
    dist = np.abs(max_latent_size-min_latent_size)
    prc = np.abs(v-min_latent_size)/dist
    max_size = np.abs(range[1] - range[0])
    #substracted since space encodes from biggest to smallest
    return prc*max_size


def torch_get_size_in_space(v, device, range=[-1, 1]):
    dist = torch.abs(torch.tensor(max_latent_size-min_latent_size).float()).to(device)
    prc = torch.abs(v-min_latent_size)/dist
    max_size = torch.abs(torch.tensor(range[1] - range[0]).float())
    #substracted since space encodes from biggest to smallest
    return prc*max_size


def from_real_pos_to_range(pos):
    return map_single_point(pos, table_map_x, table_map_y)


def from_real_radius_to_range(radius, real_range_size = 0.5, latent_range=[-1, 1]):
    prc = radius*2 / real_range_size
    max_radius = np.abs(latent_range[1] - latent_range[0]) / 2
    return prc * max_radius

#https://dfdazac.github.io/01-sbd.html based on
def analyze_lowest_variance_components(model, latent_size, train_file, batch_size, device):
    avg_var = torch.zeros(latent_size).to(device).detach()
    data_set = np.load(train_file)
    data_size = len(data_set)
    data_set = np.split(data_set, data_size / batch_size)

    for batch_idx, data in enumerate(data_set):
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        mu, logvar = model.encode(data)
        mu = mu.detach()
        logvar = logvar.detach()
        avg_var += torch.mean(logvar.exp(), dim=0)/data_size

    #get_smallest_pair
    sorted_latents = sorted(range(avg_var.shape[0]), key=lambda x: avg_var[x])
    x_latent, y_latent, *_ = sorted_latents
    a = avg_var.cpu().numpy()

    print(f'Axes with lowest variance: {x_latent:d} and {y_latent:d}')


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='the type of attribute that we want to generate/encode', type=str,
                        default='analyze_components', choices=['analyze_components','measure_degree',
                                                               'print_min_max'],
                        required=True)
    args, _ = parser.parse_known_args()
    if args.task == 'measure_degree':
        parser.add_argument('--ind_1', help='first index to extract from latent vector', type=np.int32)
        parser.add_argument('--ind_2', help='second index to extract from latent vector', type=np.int32)
    parser.add_argument('--env', help='gym env id', type=str, default='FetchReach-v1')

    parser.add_argument('--enc_type', help='the type of attribute that we want to generate/encode', type=str,
                        default='goal', choices=['goal', 'obstacle', 'obstacle_sizes', 'goal_sizes'])
    parser.add_argument('--batch_size', help='size image in pixels', type=np.int32, default=16)
    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=84)
    parser.add_argument('--latent_size', help='latent size to train the VAE', type=np.int32, default=5)



    args = parser.parse_args()

    args = parser.parse_args()
    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    base_data_dir = this_file_dir + '../data/'
    data_dir = base_data_dir + args.env + '/'
    train_file = data_dir + train_file_name[args.enc_type]
    weights_path = data_dir + vae_sb_weights_file_name[args.enc_type]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    from j_vae.train_vae_sb import load_Vae as load_Vae_SB
    model = load_Vae_SB(weights_path, args.img_size, args.latent_size)
    if args.task == 'analyze_components':
        analyze_lowest_variance_components(model, args.latent_size, train_file, args.batch_size, device)
    elif args.task == 'print_min_max':
        ##print_min_and_max_from_sizes
        print_min_and_max_from_sizes(model, train_file)
    elif args.task == 'measure_degree':
        ##for angle
        file_corners = data_dir + file_corners_name[args.enc_type]
        calculate_angle(model, file_corners, args.ind_1, args.ind_2)



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




