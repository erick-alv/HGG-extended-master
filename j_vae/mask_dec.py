import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image
from torch import optim
import numpy as np


class SpatialBroadcast4D(nn.Module):
    """
    A normal BroadCast but for every element of the
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions

        :param x: (B, S, L)
        :param width: W
        :param height: H
        :return: (B, S, L + 2, W, H)
        """
        B, S, L = x.size()
        # (B, S, L, 1, 1)
        x = x[:, :, :, None, None]
        # (B, S, L, W, H)
        x = x.expand(B, S, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, S, 2, H, W)
        coords = coords[None, None].expand(B, S, 2, height, width)

        # (B, S,  L + 2, W, H)
        x = torch.cat((x, coords), dim=2)
        #(B*S, L+2, W, H)
        #reconstructions
        #(B, S, 3,H,W)

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(shape=(x.size(0), -1))

class MaskDec(nn.Module):
    def __init__(self, num_slots, img_size=64, feat_size=128):
        super().__init__()
        embed_size = img_size // 16
        self.num_slots = num_slots
        self.feat_size = feat_size
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 16x downsampled: (64, H/16, W/16)
            Flatten(),
            nn.Linear(64 * embed_size ** 2, num_slots*feat_size),
            nn.ELU(),
        )
        self.spatial_broadcast4d = SpatialBroadcast4D()
        self.img_size = img_size

        kernel_size = 3
        decoder_stride = 1
        #kernel_size = 3, stride = 1, padding =1, dilation =1 is a size preserving convolution

        self.dec = nn.Sequential(
            nn.Conv2d(in_channels=feat_size + 2, kernel_size=kernel_size,
                                 stride=decoder_stride, out_channels=img_size*2, padding=1),
            nn.Conv2d(in_channels=img_size*2, kernel_size=kernel_size,
                      stride=decoder_stride, out_channels=img_size, padding=1),
            nn.Conv2d(in_channels=img_size,
                      kernel_size=kernel_size, stride=decoder_stride, out_channels=img_size, padding=1),
            nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                      stride=decoder_stride, out_channels=1, padding=1)
        )

    def forward(self, x):
        B = x.shape[0]
        #(B, S*D)
        x = self.enc(x)
        #to (B, S, D)
        x = x.reshape(shape=(B, self.num_slots, -1))
        #(B, S,  L + 2, W, H)
        x = self.spatial_broadcast4d(x, self.img_size, self.img_size)
        #(B, S,  L + 2, W, H) -> (B*S,  L + 2, W, H)
        x = x.view(B*self.num_slots, x.shape[2], x.shape[3], x.shape[4])
        #(B*S, 1, W, H)
        masks = self.dec(x)
        #(B, S, 1, W, H)
        masks = masks.view(B, self.num_slots, 1, masks.shape[2], masks.shape[3])
        return torch.sigmoid(masks)

def show_im(np_rgb_array):
    from PIL import Image
    im = Image.fromarray(np_rgb_array.astype(np.uint8))
    im.show()
    im.close()

def np_data_batch_to_torch(np_data_batch, device):
    data_batch = torch.from_numpy(np_data_batch).float().to(device)
    data_batch /= 255.
    data_batch = data_batch.permute([0, 3, 1, 2])
    return data_batch

def train(model, optimizer, device, log_interval,  batch_size):
    model.train()
    train_loss = 0
    data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy')
    data_size = len(data_set)
    for epoch in range(10):
        #creates indexes and shuffles them. So it can acces the data
        idx_set = np.arange(data_size)
        np.random.shuffle(idx_set)
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        for batch_idx, idx_select in enumerate(idx_set):
            data = data_set[idx_select]
            data = torch.from_numpy(data).float().to(device)
            data /= 255.
            input_ims = data[:, 0, :, :, :]
            input_ims = input_ims.permute([0, 3, 1, 2])
            orig_masks = data[:, 1:, :, :, 0:1]#also leave just one color channel since it is a mask converted to rgb
            orig_masks = orig_masks.permute([0, 1, 4, 2, 3])
            optimizer.zero_grad()
            rec_masks = model(input_ims)
            BCE = F.binary_cross_entropy(rec_masks, orig_masks)
            loss = BCE
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % log_interval == 0:
                visualize_masks(orig_masks, rec_masks, 'recs_{}_{}.png'.format(epoch, batch_idx))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), data_size,
                           100. * (batch_idx + 1) / len(data_set),
                           loss.item() / len(data)))
                print('Loss: ', loss.item() / len(data))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / data_size))


def visualize_masks(orig_masks, rec_masks, file_name):
    im = torch.cat([orig_masks, rec_masks], dim=1)
    im = im.reshape(shape=(im.shape[0]*im.shape[1], im.shape[2], im.shape[4], im.shape[4]))

    save_image(im, file_name, nrow=12, pad_value=0.3)


if __name__ == '__main__':
    data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_val.npy')
    '''data = data_set[:4]
    data = torch.from_numpy(data).float()
    data /= 255.
    data = data.permute([0, 3, 1, 2])

    rec = model(data)
    print(rec.shape)'''
    device = 'cuda:0'
    model = MaskDec(6).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    train(model, optimizer, device, log_interval=400, batch_size=4)


    '''device = 'cuda:0'
    from j_vae.train_monet import load_Vae
    model = load_Vae(path='../data/FetchGenerativeEnv-v1/all_sb_model', img_size=64, latent_size=6)
    model = model.to(device)
    data_size = len(data_set)
    new_data_set = np.zeros(shape=(data_size, 6 + 1, 64, 64, 3))
    batch_size = 8
    # creates indexes and shuffles them. So it can acces the data
    idx_set = np.arange(data_size)
    idx_set = np.split(idx_set, len(idx_set) / batch_size)
    with torch.no_grad():
        for batch_idx, idx_select in enumerate(idx_set):
            data_np = data_set[idx_select]
            data = np_data_batch_to_torch(data_np.copy(), device)
            masks = model.get_masks(data)
            # (B, S, 1, H, W)
            masks = masks.cpu().numpy()
            # (B, S, 3, H, W)
            masks = np.tile(masks, (1, 1, 3, 1, 1))
            # to (B, S, H, W, 3)
            masks = np.transpose(masks, axes=(0, 1, 3, 4, 2))
            # (B, S+1, H, W, 3)
            #make masks visible in image
            masks *= 255.
            ims = np.concatenate([np.expand_dims(data_np, axis=1), masks], axis=1)
            new_data_set[batch_idx:batch_idx + batch_size] = ims
    np.save('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy', new_data_set)'''