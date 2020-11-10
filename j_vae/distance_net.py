from envs.distance_graph import DistanceGraph2D
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image
from torch import optim
import numpy as np
from SPACE.model.space.utils import spatial_transform, linear_annealing
from SPACE.model.space.fg import NumericalRelaxedBernoulli, kl_divergence_bern_bern

#-- Data Generation --
def generate_bounding_boxes(field):
    # up to 8 bboxes
    max_boxes = 8
    n_pos = np.linspace(start=0, stop=8, endpoint=True, num=9)
    num_boxes = np.random.choice(n_pos, p=[0.04, ] + 8 * [0.12]).astype(np.int)
    g_bboxes = np.zeros(shape=(max_boxes, 4))
    obstacles_list = []
    for i in range(num_boxes):
        b = gen_single_box_inside_field(field)
        g_bboxes[i] = b
        obstacles_list.append(b)

    # changes order of boxes
    np.random.shuffle(g_bboxes)
    return g_bboxes, obstacles_list


def gen_single_box(possible_begin_x, possible_begin_y, possible_lengths_x, possible_lengths_y):
    field_begin_x = np.random.uniform(low=possible_begin_x[0], high=possible_begin_x[1])
    field_begin_y = np.random.uniform(low=possible_begin_y[0], high=possible_begin_y[1])
    lx = np.random.choice(possible_lengths_x)
    center_x = field_begin_x + lx / 2.
    ly = np.random.choice(possible_lengths_y)
    center_y = field_begin_y + ly / 2.
    box = [center_x, center_y, lx / 2., ly / 2.]
    return box


def gen_single_box_inside_field(field):
    # here lx and ly are half the size
    center_x, center_y, field_lx, field_ly = field

    lx = field_lx
    ly = field_ly
    a = np.random.randint(low=0, high=1)
    if a == 0:
        lx /= 2.
    else:
        ly /= 2.

    b = np.random.randint(low=0, high=1)
    if b == 0:
        lx /= 2.

    c = np.random.randint(low=0, high=1)
    if c == 0:
        ly /= 2.

    # possible lengths. Do not allow too small since graph cannot handle that case
    if lx > 1:
        nx = 20
    else:
        nx = 10
    possible_x = np.linspace(start=lx / nx, stop=2 * lx, endpoint=True, num=nx)
    # this is the whole lenght
    # todo see if needs to ajust begin
    blx = np.random.choice(possible_x)
    begin_x = np.random.uniform(low=center_x - field_lx, high=center_x + field_lx - blx)
    cx = begin_x + blx / 2.

    if ly > 1:
        ny = 20
    else:
        ny = 10
    possible_y = np.linspace(start=ly / ny, stop=2 * ly, endpoint=True, num=ny)
    bly = np.random.choice(possible_y)
    begin_y = np.random.uniform(low=center_y - field_ly, high=center_y + field_ly - bly)
    cy = begin_y + bly / 2.

    return np.array([cx, cy, blx / 2., bly / 2.])

#todo make better pairs of points
def get_data_points_pairs(field):
    #points pairs inside the field
    center_x, center_y, field_lx, field_ly = field
    #inside
    p1 = np.random.uniform(low=[center_x-field_lx, center_y-field_ly],
                           high=[center_x+field_lx, center_y+field_ly], size=[500, 2])
    p2 = np.random.uniform(low=[center_x - field_lx, center_y - field_ly],
                           high=[center_x + field_lx, center_y + field_ly], size=[500, 2])
    '''#left
    pl = np.random.uniform(low=[center_x - field_lx-100 , center_y - field_ly -100],
                           high=[center_x - field_lx, center_y + field_ly+100], size=[100, 2])
    #
    pu = np.random.uniform(low=[center_x - field_lx-100, center_y + field_ly],
                           high=[center_x + field_lx+100, center_y + field_ly +100], size=[200, 2])
    pd = np.random.uniform(low=[center_x - field_lx - 100, center_y + field_ly],
                           high=[center_x + field_lx + 100, center_y + field_ly + 100], size=[200, 2])'''

    #points pairs both outside
    #points pairs one inside, other outside
    #points pairs assure almost all region
    return np.concatenate([p1, p2], axis=1)

def gen_training_data(N, filename):

    ##the function will consist of: [px1,py1, px2, py2, coords_bboxes] -> dist
    ##(4 + 4 + 8*4) -> 1

    # generate region
    data_set = np.zeros(shape=(N, 4+8*4+1))
    pos_fields_begin = [-2,2]
    pos_f_l = np.linspace(start=0.5, stop=3, endpoint=True, num=6)
    counter_t = 0
    while counter_t < N:
        #field = gen_single_box(possible_begin_x=pos_fields_begin, possible_begin_y=pos_fields_begin,
        #                       possible_lengths_x=pos_f_l, possible_lengths_y=pos_f_l)
        field = [0., 0., 1., 1.]
        generated_bounding_boxes, obstacles_list = generate_bounding_boxes(field)



        graph = DistanceGraph2D(args=None, field=field, obstacles=obstacles_list,
                                num_vertices=[100, 100], size_increase=0.0)
        graph.compute_cs_graph()
        graph.compute_dist_matrix()
        points_pairs = get_data_points_pairs(field)
        distances = [graph.get_dist(ppair[:2], ppair[2:]) for ppair in points_pairs]
        distances = [d[0] if d[0] != np.inf else 9999 for d in distances]
        for i, d in enumerate(distances):
            if counter_t < N:
                entry = np.concatenate([points_pairs[i], np.ravel(generated_bounding_boxes), np.array([d])])
                data_set[counter_t] = entry
                counter_t +=1
            else:
                break
        del graph
    np.save(filename, data_set)

#--     Network  --
class DistNet(nn.Module):
    def __init__(self, device, img_size=64, hidden_size=256, fg_sigma=0.15, latent_size=8):
        super().__init__()
        self.device = device
        kernel_size = 3
        conv_size1 = 64
        conv_size2 = 32
        encoder_stride = 2
        self.img_size = img_size
        self.fg_sigma = fg_sigma
        self.use_bg_mask = False
        self.bb_net = nn.Sequential(
            nn.Linear(32, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.p_net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.d = nn.Bilinear(hidden_size, hidden_size, 1)



    def forward(self, x):
        B = x.shape[0]
        D = x.shape[1]
        ps = x[:, :4]
        bs = x[:, 4:]
        np = self.p_net(ps)
        nb = self.bb_net(bs)
        y = self.d(np, nb)
        return y.squeeze()


#function should be [px1,py1, px2, py2, coords_bboxes] -> dist
#therefore datasets are [px1,py1, px2, py2, coords_bboxes, dist]
def train(model, optimizer, device, log_interval_epoch, log_interval_batch,  batch_size, total_epochs,
          train_file_name, evaluation_file_name):
    model.train()
    train_loss = 0
    data_set = np.load(train_file_name)
    data_size = len(data_set)
    for epoch in range(total_epochs):
        #creates indexes and shuffles them. So it can acces the data
        idx_set = np.arange(data_size)
        np.random.shuffle(idx_set)
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        for batch_idx, idx_select in enumerate(idx_set):
            data = data_set[idx_select]
            data = torch.from_numpy(data).float().to(device)
            #data /= 255.

            #
            _, D = data.shape
            input_vec = data[:, 0:D-1]
            expected_dist = data[:, -1]
            optimizer.zero_grad()
            #save_image(imgs_with_mask[0], 'mask_division.png')
            predictions_dist = model(input_vec)
            loss = nn.MSELoss()(predictions_dist, expected_dist)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % log_interval_batch == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), data_size,
                           100. * (batch_idx + 1) / len(data_set),
                           loss.item() / len(data)))
                print('Loss: ', loss.item() / len(data))

        if epoch % log_interval_epoch == 0:
            evaluate_data(model, evaluation_file_name)
            save_checkpoint(model, optimizer, '../data/FetchGenerativeEnv-v1/model_dist', epoch=epoch)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / data_size))
    save_checkpoint(model, optimizer, '../data/FetchGenerativeEnv-v1/model_dist')

def save_checkpoint(model, optimizer, weights_path, epoch=None):
    print('Saving Progress!')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, weights_path)
    if epoch is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weights_path + '_epoch_' + str(epoch))

def evaluate_data(model, evaluation_file_name):
    model.train()
    train_loss = 0
    data_set = np.load(evaluation_file_name)
    data_size = len(data_set)
    # creates indexes and shuffles them. So it can acces the data
    idx_set = np.arange(data_size)
    np.random.shuffle(idx_set)
    idx_set[:100]
    batch_size = 10
    idx_set = np.split(idx_set, len(idx_set) / batch_size)
    for batch_idx, idx_select in enumerate(idx_set):
        with torch.no_grad():
            data = data_set[idx_select]
            data = torch.from_numpy(data).float().to(device)
            _, D = data.shape
            input_vec = data[:, 0:D - 1]
            expected_dist = data[:, -1]
            predictions_dist = model(input_vec)
            loss = nn.MSELoss()(predictions_dist, expected_dist)
            optimizer.step()
            train_loss += loss.item()
            if batch_idx == 0:
                print('Real values: {}'.format(expected_dist))
                print('Predicted values: {}'.format(predictions_dist))
    print('---------- Evaluation Loss: ', train_loss / len(data))





if __name__ == '__main__':
    #gen_training_data(10000, filename='../data/FetchGenerativeEnv-v1/distances.npy')
    #gen_training_data(3000, filename='../data/FetchGenerativeEnv-v1/distances_ev.npy')
    device = 'cuda:0'
    seed = 1
    torch.manual_seed(seed)
    model = DistNet(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train(model, optimizer, device, log_interval_epoch=25, log_interval_batch=400, batch_size=64, total_epochs=400,
          train_file_name='../data/FetchGenerativeEnv-v1/distances.npy',
          evaluation_file_name='../data/FetchGenerativeEnv-v1/distances_ev.npy')

