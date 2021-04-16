 # based on https://github.com/stelzner/monet
# License: MIT
# Author: Karl Stelzner
import argparse
import torch
from torch import nn, optim
import torch.distributions as dists
import numpy as np
from PIL import Image
import os
from utils.os_utils import make_dir
this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(single_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(single_conv(channel_base * 2**(i+1), channel_base * 2**i))

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks-1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i -1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)

class AttentionNet(nn.Module):
    def __init__(self, num_blocks, channel_base):
        super().__init__()
        self.unet = UNet(num_blocks=num_blocks,
                         in_channels=4,
                         out_channels=2,
                         channel_base=channel_base)

    def forward(self, x, scope):
        inp = torch.cat((x, scope), 1)
        logits = self.unet(inp)
        alpha = torch.softmax(logits, 1)
        # output channel 0 represents alpha_k,
        # channel 1 represents (1 - alpha_k).
        mask = scope * alpha[:, 0:1]
        new_scope = scope * alpha[:, 1:2]
        return mask, new_scope


class EncoderNet(nn.Module):
    def __init__(self, width, height, device, latent_size=16, full_connected_size=256, input_channels=4,
                 kernel_size=3, encoder_stride=2, conv_size1=32, conv_size2=64):
        super().__init__()
        self.device = device
        self.conv_size1 = conv_size1
        self.conv_size2 = conv_size2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size2,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size2, out_channels=conv_size2,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True)
        )

        red_width = width
        red_height = height
        for i in range(3):
            red_width = (red_width - 1) // 2
            red_height = (red_height - 1) // 2

        self.red_width = red_width
        self.red_height = red_height

        self.fc1 = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
        )
        self.fc21 = nn.Linear(full_connected_size, latent_size)
        self.fc22 = nn.Linear(full_connected_size, latent_size)

    def forward(self, x):
        cx = self.convs(x)
        f_cx = cx.reshape(-1, self.red_width * self.red_height * self.conv_size2)
        #x = x.view(x.shape[0], -1)
        e = self.fc1(f_cx)
        return self.fc21(e), self.fc22(e)

class DecoderNet(nn.Module):
    def __init__(self, width, height, device, latent_size, output_channels=4,
                 kernel_size=3, conv_size1=32, decoder_stride=1):
        super().__init__()
        self.device = device
        self.height = height
        self.width = width
        self.latent_size = latent_size

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=latent_size + 2, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=output_channels, kernel_size=1),
        )
        ys = torch.linspace(-1, 1, self.height + 4)
        xs = torch.linspace(-1, 1, self.width + 4)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 4, self.width + 4)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result

class Monet_VAE(nn.Module):
    def __init__(self, height, width, device, latent_size, num_blocks, channel_base, num_slots,
                 full_connected_size, color_channels, kernel_size, encoder_stride,decoder_stride,
                 conv_size1, conv_size2):
        super().__init__()
        self.device = device
        self.num_slots = num_slots
        self.latent_size = latent_size
        self.height = height
        self.width = width
        self.color_channels = color_channels
        self.attention = AttentionNet(num_blocks=num_blocks, channel_base=channel_base)
        self.encoder = EncoderNet(width=width, height=height, device=device, latent_size=latent_size,
                                  full_connected_size=full_connected_size, input_channels=color_channels+1,
                                  kernel_size=kernel_size, encoder_stride=encoder_stride, conv_size1=conv_size1,
                                  conv_size2=conv_size2)
        self.decoder = DecoderNet(width=width, height=height, device=device, latent_size=latent_size,
                                  output_channels=color_channels+1, kernel_size=kernel_size, conv_size1=conv_size1,
                                  decoder_stride=decoder_stride)

    def _encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        mu, logvar = self.encoder(encoder_input)
        return mu, logvar

    def get_masks(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.num_slots - 1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        # list len S (B, 1, H, W)
        masks.append(scope)
        #(S, B, 1, H, W)
        masks = torch.stack(masks)
        #(B, S, 1, H, W)
        masks = masks.permute([1, 0, 2, 3, 4])

        return masks

    def encode(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(self.num_slots-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        mu_s = []
        logvar_s = []
        for i, mask in enumerate(masks):
            mu, logvar = self._encoder_step(x, mask)
            mu_s.append(mu)
            logvar_s.append(logvar)

        return mu_s, logvar_s, masks

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def _decoder_step(self, z):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        return x_recon, mask_pred

    def decode(self, z_s, masks):
        full_reconstruction = torch.zeros(
            (masks[0].shape[0], self.color_channels, self.width, self.height)).to(self.device)
        x_recon_s, mask_pred_s = [], []
        for i in range(len(masks)):
            x_recon, mask_pred = self._decoder_step(z_s[i])
            x_recon_s.append(x_recon)
            mask_pred_s.append(mask_pred)
            full_reconstruction += x_recon*masks[i]
        return full_reconstruction, x_recon_s, mask_pred_s

    def forward(self, x):
        mu_s, logvar_s, masks = self.encode(x)
        z_s = [self._reparameterize(mu_s[i], logvar_s[i]) for i in range(len(mu_s))]
        full_reconstruction, x_recon_s, mask_pred_s = self.decode(z_s, masks)
        return mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s


def loss_function(x, x_recon_s, masks, mask_pred_s, mu_s, logvar_s, beta, gamma, bg_sigma, fg_sigma, device):
    batch_size = x.shape[0]
    p_xs = torch.zeros(batch_size).to(device)
    kl_z = torch.zeros(batch_size).to(device)
    for i in range(len(masks)):
        kld = -0.5 * torch.sum(1 + logvar_s[i] - mu_s[i].pow(2) - logvar_s[i].exp(), dim=1)
        '''for t in kld:
            assert not torch.isnan(t)
            assert not torch.isinf(t)'''
        kl_z += kld
        if i == 0:
            sigma = bg_sigma
        else:
            sigma = fg_sigma
        dist = dists.Normal(x_recon_s[i], sigma)
        #log(p_theta(x|z_k))
        p_x = dist.log_prob(x)
        p_x *= masks[i]
        p_x = torch.sum(p_x, [1, 2, 3])
        '''for t in p_x:
            assert not torch.isnan(t)
            assert not torch.isinf(t)'''
        p_xs += -p_x#this iterartive sum might not be correct since log(x*y) = log(x)+log(y)


    masks = torch.cat(masks, 1)
    tr_masks = torch.transpose(masks, 1, 3)
    q_masks = dists.Categorical(probs=tr_masks)
    stacked_mask_preds = torch.stack(mask_pred_s, 3)
    q_masks_recon = dists.Categorical(logits=stacked_mask_preds)
    #avoid problem of kl_divergence becoming inf
    smallest_num = torch.finfo(q_masks_recon.probs.dtype).tiny
    q_masks_recon.probs[q_masks_recon.probs == 0.] = smallest_num

    kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
    kl_masks = torch.sum(kl_masks, [1, 2])
    '''for t in kl_masks:
        assert not torch.isnan(t)
        assert not torch.isinf(t)'''
    loss = gamma * kl_masks + p_xs + beta* kl_z
    return loss

def train(epoch, model, optimizer, device, log_interval, train_file, batch_size, beta, gamma, bg_sigma, fg_sigma):
    model.train()
    train_loss = 0
    data_set = np.load(train_file)

    data_size = len(data_set)
    #creates indexes and shuffles them. So it can acces the data
    idx_set = np.arange(data_size)
    np.random.shuffle(idx_set)
    idx_set = idx_set[:12800]
    idx_set = np.split(idx_set, len(idx_set) / batch_size)
    for batch_idx, idx_select in enumerate(idx_set):
        data = data_set[idx_select]
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        optimizer.zero_grad()
        mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s = model(data)
        loss_batch = loss_function(data, x_recon_s, masks, mask_pred_s, mu_s, logvar_s,
                                   beta, gamma, bg_sigma, fg_sigma, device=device)
        loss = torch.mean(loss_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), data_size,
                       100. * (batch_idx + 1) / len(data_set),
                       loss.item() / len(data)))
            print('Loss: ', loss.item() / len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / data_size))

def numpify(tensor):
    return tensor.cpu().detach().numpy()

def visualize_masks(imgs, masks, recons, file_name):
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0), (0, 127, 255), (0,255, 127)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks_argmax = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks_argmax[i, y, x]]

    imgs *= 255.0
    recons *= 255.0
    masks *= 255.0
    masks_ims = [np.stack([masks[:, i, :, :]]*3, axis=1) for i in range(masks.shape[1])]
    masks_ims = [np.concatenate(np.transpose(m, (0, 2, 3, 1)), axis=1) for m in masks_ims]

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    imgs = np.concatenate(imgs, axis=1)
    seg_maps = np.transpose(seg_maps, (0, 2, 3, 1))
    seg_maps = np.concatenate(seg_maps, axis=1)
    recons = np.transpose(recons, (0, 2, 3, 1))
    recons = np.concatenate(recons, axis=1)
    all_list = [imgs, seg_maps, recons]+masks_ims
    all_im_array = np.concatenate(all_list, axis=0)
    all_im = Image.fromarray(all_im_array.astype(np.uint8))
    all_im.save(file_name)

def train_Vae(batch_size, img_size, latent_size, train_file, vae_weights_path, beta, gamma, bg_sigma, fg_sigma,
              epochs=100, no_cuda=False, seed=1, log_interval=100, load=False,
              num_blocks=5, channel_base=64, num_slots=6,
              full_connected_size=256, color_channels=3, kernel_size=3, encoder_stride=2,decoder_stride=1,
              conv_size1=32, conv_size2=64):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    if load:
        model = Monet_VAE(height=img_size, width=img_size, device=device, latent_size=latent_size,
                          num_blocks=num_blocks,
                          channel_base=channel_base, num_slots=num_slots, full_connected_size=full_connected_size,
                          color_channels=color_channels, kernel_size=kernel_size, encoder_stride=encoder_stride,
                          decoder_stride=decoder_stride, conv_size1=conv_size1, conv_size2=conv_size2).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
        checkpoint = torch.load(vae_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
    else:
        model = Monet_VAE(height=img_size, width=img_size, device=device, latent_size=latent_size,
                          num_blocks=num_blocks,
                          channel_base=channel_base, num_slots=num_slots, full_connected_size=full_connected_size,
                          color_channels=color_channels, kernel_size=kernel_size, encoder_stride=encoder_stride,
                          decoder_stride=decoder_stride, conv_size1=conv_size1, conv_size2=conv_size2).to(device)

        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
        start_epoch = 1

    for epoch in range(start_epoch, epochs + start_epoch):
        train(epoch=epoch, model=model, optimizer=optimizer, device=device, log_interval=log_interval,
              train_file=train_file, batch_size=batch_size, beta=beta,
              gamma=gamma, bg_sigma=bg_sigma, fg_sigma=fg_sigma)
        if not (epoch % 5) or epoch == 1:
            compare_with_data_set(model, device, filename_suffix='epoch_{}'.format(epoch), latent_size=latent_size,
                                  train_file=train_file)
            print('Saving Progress!')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, vae_weights_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, vae_weights_path+'_epoch_'+str(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, vae_weights_path)


def compare_with_data_set(model, device, filename_suffix, latent_size, train_file):
    data_set = np.load(train_file)
    data_size = len(data_set)
    idx = np.random.randint(0, data_size, size=10)
    data = data_set[idx]
    print(data.shape)
    with torch.no_grad():
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s = model(data)
        visualize_masks(imgs=numpify(data),masks=numpify(torch.cat(masks, dim=1)), recons=numpify(full_reconstruction),
                        file_name=this_file_dir+'results/reconstruction_{}.png'.format(filename_suffix))


def load_Vae(path, img_size, latent_size, no_cuda=False, seed=1, num_blocks=5, channel_base=64, num_slots=6,
                 full_connected_size=256, color_channels=3,kernel_size=3, encoder_stride=2,decoder_stride=1,
                 conv_size1=32, conv_size2=64):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = Monet_VAE(height=img_size, width=img_size, device=device, latent_size=latent_size, num_blocks=num_blocks,
                      channel_base=channel_base, num_slots=num_slots,full_connected_size=full_connected_size,
                      color_channels=color_channels,kernel_size=kernel_size, encoder_stride=encoder_stride,
                      decoder_stride=decoder_stride, conv_size1=conv_size1, conv_size2=conv_size2).to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str)
    parser.add_argument('--task', help='use monet for training or testing', type=str,
                        choices=['train', 'test'], required=True)

    parser.add_argument('--batch_size', help='number of batch to train', type=np.float, default=32)
    parser.add_argument('--train_epochs', help='number of epochs to train vae', type=np.int32, default=40)
    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=64)
    parser.add_argument('--latent_size', help='latent size to train the VAE', type=np.int32, default=6)
    parser.add_argument('--num_slots', help='number of slots', type=np.int32, default=6)
    parser.add_argument('--beta', help='beta val for the reconstruction loss', type=np.float, default=8.)#5#8
    parser.add_argument('--gamma', help='gamma val for the mask loss', type=np.float, default=5.)#2.)#5
    parser.add_argument('--bg_sigma', help='', type=np.float, default=0.09)
    parser.add_argument('--fg_sigma', help='', type=np.float, default=0.11)


    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    make_dir(this_file_dir+'results/', clear=False)
    base_data_dir = this_file_dir + '../data/'
    data_dir = base_data_dir + args.env + '/'

    train_file = data_dir + 'all_set.npy'
    weights_path = data_dir + 'all_sb_model'

    if args.task == 'train':
        train_Vae(epochs=args.train_epochs, batch_size=args.batch_size,img_size=args.img_size,
                  latent_size=args.latent_size, train_file=train_file,
                  vae_weights_path=weights_path, beta=args.beta, gamma=args.gamma, bg_sigma=args.bg_sigma,
                  fg_sigma=args.fg_sigma, load=False, num_slots=args.num_slots)
    else:
        device = torch.device("cuda")
        model = load_Vae(path=weights_path, img_size=args.img_size, latent_size=args.latent_size)
        compare_with_data_set(model=model, device=device, latent_size=args.latent_size,
                         filename_suffix="test", train_file=train_file)
