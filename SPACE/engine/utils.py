import os
import argparse
from argparse import ArgumentParser
from SPACE.model.space.utils import spatial_transform
import torchvision
import torch
from SPACE.config import cfg
this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def get_config():
    cfg.merge_from_file(this_file_dir+'../configs/table.yaml')
    '''if args.opts:
        cfg.merge_from_list(args.opts)'''

    # Seed
    import torch
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(cfg.seed)
    
    return cfg


rbox = torch.zeros(3, 21, 21)
rbox[0, :2, :] = 1
rbox[0, -2:, :] = 1
rbox[0, :, :2] = 1
rbox[0, :, -2:] = 1
rbox = rbox.view(1, 3, 21, 21)

gbox = torch.zeros(3, 21, 21)
gbox[1, :2, :] = 1
gbox[1, -2:, :] = 1
gbox[1, :, :2] = 1
gbox[1, :, -2:] = 1
gbox = gbox.view(1, 3, 21, 21)

blbox = torch.zeros(3, 21, 21)
blbox[2, :2, :] = 1
blbox[2, -2:, :] = 1
blbox[2, :, :2] = 1
blbox[2, :, -2:] = 1
blbox = blbox.view(1, 3, 21, 21)

ybox = torch.zeros(3, 21, 21)
ybox[0, :2, :] = 1
ybox[0, -2:, :] = 1
ybox[0, :, :2] = 1
ybox[0, :, -2:] = 1
ybox[1, :2, :] = 1
ybox[1, -2:, :] = 1
ybox[1, :, :2] = 1
ybox[1, :, -2:] = 1
ybox = ybox.view(1, 3, 21, 21)

abox = torch.zeros(3, 21, 21)
abox[1, :2, :] = 1
abox[1, -2:, :] = 1
abox[1, :, :2] = 1
abox[1, :, -2:] = 1
abox[2, :2, :] = 1
abox[2, -2:, :] = 1
abox[2, :, :2] = 1
abox[2, :, -2:] = 1
abox = abox.view(1, 3, 21, 21)

pbox = torch.zeros(3, 21, 21)
pbox[0, :2, :] = 1
pbox[0, -2:, :] = 1
pbox[0, :, :2] = 1
pbox[0, :, -2:] = 1
pbox[2, :2, :] = 1
pbox[2, -2:, :] = 1
pbox[2, :, :2] = 1
pbox[2, :, -2:] = 1
pbox = pbox.view(1, 3, 21, 21)

boxes = torch.cat((rbox, gbox, blbox, ybox, abox, pbox))


def visualize(x, z_pres, z_where_scale, z_where_shift, rbox=rbox, gbox=gbox, num_obj=8 * 8):
    """
        x: (bs, 3, *img_shape)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    B, _, *img_shape = x.size()
    bs = z_pres.size(0)
    # num_obj = 8 * 8
    z_pres = z_pres.view(-1, 1, 1, 1)
    # z_scale = z_where[:, :, :2].view(-1, 2)
    # z_shift = z_where[:, :, 2:].view(-1, 2)
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = spatial_transform(z_pres * gbox + (1 - z_pres) * rbox,
                             torch.cat((z_scale, z_shift), dim=1),
                             torch.Size([bs * num_obj, 3, *img_shape]),
                             inverse=True)
    bbox = (bbox + torch.stack(num_obj * (x,), dim=1).view(-1, 3, *img_shape)).clamp(0.0, 1.0)
    return bbox


def add_bbox(x, score, z_where_scale, z_where_shift, rbox=rbox, gbox=gbox, num_obj=8 * 8):
    B, _, *img_shape = x.size()
    bs = score.size(0)
    score = score.view(-1, 1, 1, 1)
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = spatial_transform(score * gbox + (1 - score) * rbox,
                             torch.cat((z_scale, z_shift), dim=1),
                             torch.Size([bs * num_obj, 3, *img_shape]),
                             inverse=True)
    bbox = (bbox + x.repeat(1, 3, 1, 1).view(-1, 3, *img_shape)).clamp(0.0, 1.0)
    return bbox


def bbox_in_one(x, z_pres, z_where_scale, z_where_shift, gbox=gbox):
    B, _, *img_shape = x.size()
    B, N, _ = z_pres.size()
    #from (B, G*G, D) to (B*G*G, D) #N = G*G
    '''z_p = np.squeeze(z_pres.detach().cpu().numpy())
    z_sc = np.squeeze(z_where_scale.detach().cpu().numpy(), axis=0)
    z_sh = np.squeeze(z_where_shift.detach().cpu().numpy(), axis=0)
    indices = z_p > 0.98
    coordinates = z_sh[indices]
    sizes = z_sc[indices]
    fig, ax = plt.subplots()
    ax.scatter([-1, -1, 1, 1], [-1, 1, -1, 1])
    ax.scatter(coordinates[:, 0], coordinates[:, 1], marker='s')
    for i in range(len(coordinates)):
        # Create a Rectangle patch
        x_c = coordinates[i, 0] - sizes[i, 0]
        y_c = coordinates[i, 1] - sizes[i, 1]
        rect = patches.Rectangle((x_c, y_c),
                                 sizes[i, 0], sizes[i, 1], linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.savefig('../output/coordinates_im.png')
    plt.close()'''

    z_pres = z_pres.view(-1, 1, 1, 1)
    z_scale = z_where_scale.reshape(shape=(-1, 2))
    z_shift = z_where_shift.reshape(shape=(-1, 2))
    # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
    # kbox = boxes[argmax_cluster.view(-1)]
    bbox = spatial_transform(z_pres * gbox,  # + (1 - z_pres) * rbox,
                             torch.cat((z_scale, z_shift), dim=1),
                             torch.Size([B * N, 3, *img_shape]),
                             inverse=True)

    #to (B, G*G, 3, H, W)
    bbox = (bbox.view(B, N, 3, *img_shape).sum(dim=1).clamp(0.0, 1.0) + x).clamp(0.0, 1.0)
    return bbox



# Times 10 to prevent index out of bound.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 10

