from j_vae.common_data import train_file_name, vae_sb_weights_file_name
import argparse
import torch
from torch import nn, optim
import torch.distributions as dists
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from torch.distributions.normal import Normal

this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
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
            self.down_convs.append(double_conv(cur_in_channels,
                                               channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks-1, 0, -1):
            self.tconvs.append(nn.ConvTranspose2d(channel_base * 2**i,
                                                  channel_base * 2**(i-1),
                                                  2, stride=2))

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks-2, -1, -1):
            self.up_convs.append(double_conv(channel_base * 2**(i+1), channel_base * 2**i))

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
        '''self.convs = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
            kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size2,
            kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size2, out_channels=conv_size2,
            kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True)
        )'''

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
        #todo is 4 since for conv layers; if less the have to reduce this as well
        for i in range(3):#4):
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
        '''self.convs = nn.Sequential(
            nn.Conv2d(in_channels=latent_size+2, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=output_channels, kernel_size=1),
        )'''

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=latent_size + 2, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=decoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=output_channels, kernel_size=1),
        )
        ys = torch.linspace(-1, 1, self.height + 4)#4#8
        xs = torch.linspace(-1, 1, self.width + 4)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 4, self.width + 4)#todo 8
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result

class Monet2_VAE(nn.Module):
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
        self.encoderBG = EncoderNet(width=width, height=height, device=device, latent_size=latent_size,
                                  full_connected_size=full_connected_size, input_channels=color_channels + 1,
                                  kernel_size=kernel_size, encoder_stride=encoder_stride, conv_size1=conv_size1,
                                  conv_size2=conv_size2)
        self.decoder = DecoderNet(width=width, height=height, device=device, latent_size=latent_size,
                                  output_channels=color_channels+1, kernel_size=kernel_size, conv_size1=conv_size1,
                                  decoder_stride=decoder_stride)
        self.decoderBG = DecoderNet(width=width, height=height, device=device, latent_size=latent_size,
                                  output_channels=color_channels + 1, kernel_size=kernel_size, conv_size1=conv_size1,
                                  decoder_stride=decoder_stride)
        '''self.z_depth_net = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
            nn.Linear(full_connected_size, 1 * 2)
        )'''

    def _encoder_step(self, x, mask, slot_index):
        encoder_input = torch.cat((x, mask), 1)
        if slot_index == 0:
            mu, logvar = self.encoderBG(encoder_input)
        else:
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
            mu, logvar = self._encoder_step(x, mask, i)
            mu_s.append(mu)
            logvar_s.append(logvar)

        return mu_s, logvar_s, masks

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def _decoder_step(self, z, slot_index):
        if slot_index == 0:
            decoder_output = self.decoderBG(z)
        else:
            decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :3])
        mask_pred = decoder_output[:, 3]
        return x_recon, mask_pred

    def decode(self, z_s, masks):
        full_reconstruction = torch.zeros(
            (masks[0].shape[0], self.color_channels, self.width, self.height)).to(self.device)
        full_reconstruction2 = torch.zeros(
            (masks[0].shape[0], self.color_channels, self.width, self.height)).to(self.device)
        x_recon_s, mask_pred_s = [], []
        for i in range(len(masks)):
            x_recon, mask_pred = self._decoder_step(z_s[i], i)
            x_recon_s.append(x_recon)
            mask_pred_s.append(mask_pred)
            full_reconstruction += x_recon*masks[i]
            # -> (B, 1, H, W)
            m_uns = torch.unsqueeze(mask_pred, dim=1)
            full_reconstruction2 += x_recon * torch.sigmoid(m_uns)
        return full_reconstruction, full_reconstruction2, x_recon_s, mask_pred_s

    def forward(self, x, training=False, params_dict ={}):
        B = x.shape[0]
        mu_s, logvar_s, masks = self.encode(x)
        z_s = [self._reparameterize(mu_s[i], logvar_s[i]) for i in range(len(mu_s))]
        full_reconstruction, full_reconstruction2, x_recon_s, mask_pred_s = self.decode(z_s, masks)
        '''#extra loss
        #take masks of fg
        fg_masks_united = torch.sum(torch.stack(masks[1:]), dim=0)
        #extract all fg objects
        fg_objects = x * fg_masks_united

        #unite all fg objects
        fg_objects_recon_united = torch.sum(torch.stack(x_recon_s[1:]), dim=0)

        fg_dist = Normal(fg_objects_recon_united, self.fg_sigma)
        fg_likelihood = fg_dist.log_prob(fg_objects)

        log_like = fg_likelihood






        log_like = log_like.flatten(start_dim=1).sum(1)
        loss = -log_like.mean()

        return loss, mu_s, logvar_s, masks, full_reconstruction, x_recon_s, mask_pred_s'''
        if training:
            fg_sigma = params_dict['fg_sigma']
            bg_sigma = params_dict['bg_sigma']
            beta = params_dict['beta']
            gamma = params_dict['gamma']
            #calculates the loss

            batch_size = x.shape[0]
            p_xs = torch.zeros(batch_size).to(x.device)
            kl_z = torch.zeros(batch_size).to(x.device)
            for i in range(len(masks)):
                kld = -0.5 * torch.sum(1 + logvar_s[i] - mu_s[i].pow(2) - logvar_s[i].exp(), dim=1)
                kl_z += kld
                if i == 0:
                    sigma = bg_sigma
                else:
                    sigma = fg_sigma
                dist = dists.Normal(x_recon_s[i], sigma)
                # log(p_theta(x|z_k))
                p_x = dist.log_prob(x)
                p_x *= masks[i]
                p_x = torch.sum(p_x, [1, 2, 3])
                p_xs += -p_x  # this iterartive sum might not be correct since log(x*y) = log(x)+log(y)

            mask_pred_s = [m.unsqueeze(dim=1) for m in mask_pred_s]
            mask_pred_s = torch.cat(mask_pred_s, 1)

            mask_pred_softmaxed = F.softmax(mask_pred_s, dim=1)
            mask_pred_softmaxed_permuted = mask_pred_softmaxed.permute([0, 2, 3, 1])
            summed = mask_pred_softmaxed.sum(dim=1)

            mask_pred_s_permuted = mask_pred_s.permute([0, 2, 3, 1])

            masks= torch.cat(masks, 1)
            masks_permuted = masks.permute([0, 2, 3, 1])
            q_masks = dists.Categorical(probs=masks_permuted)
            q_masks_recon = dists.Categorical(logits=mask_pred_s_permuted)
            # avoid problem of kl_divergence becoming inf
            smallest_num = torch.finfo(q_masks_recon.probs.dtype).tiny
            q_masks_recon.probs[q_masks_recon.probs == 0.] = smallest_num

            q_test = dists.Categorical(probs=mask_pred_softmaxed_permuted)
            kl_test = dists.kl_divergence(q_masks_recon, q_test)

            kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
            kl_masks = torch.sum(kl_masks, [1, 2])
            loss_batch = gamma * kl_masks + p_xs + beta * kl_z
            loss = torch.mean(loss_batch)

            return loss, mu_s, logvar_s, masks, full_reconstruction, full_reconstruction2, x_recon_s, mask_pred_s
        else:
            masks = torch.cat(masks, dim=1)
            #transform to probs with softmax
            #mask_pred_s = [m.unsqueeze(dim=1) for m in mask_pred_s]
            mask_pred_s = F.softmax(torch.stack(mask_pred_s, dim=3), dim=3)
            mask_pred_s = mask_pred_s.permute((0, 3, 1, 2))
            #mask_pred_s = torch.cat(mask_pred_s, dim=1)
            #mask_pred_s = F.softmax(mask_pred_s, dim=1)
            return mu_s, logvar_s, masks, full_reconstruction, full_reconstruction2, x_recon_s, mask_pred_s





def train(epoch, model, optimizer, device, log_interval, train_file, batch_size, beta, gamma, bg_sigma, fg_sigma):
    model.train()
    train_loss = 0
    data_set = np.load(train_file)
    data_set = data_set[:128]

    data_size = len(data_set)
    #creates indexes and shuffles them. So it can acces the data
    idx_set = np.arange(data_size)
    np.random.shuffle(idx_set)
    idx_set = idx_set[:128]
    idx_set = np.split(idx_set, len(idx_set) / batch_size)
    for batch_idx, idx_select in enumerate(idx_set):
        data = data_set[idx_select]
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        optimizer.zero_grad()
        loss, mu_s, logvar_s, masks, full_reconstruction, \
        full_reconstruction2, x_recon_s, mask_pred_s = model(data, training=True, params_dict ={'fg_sigma':fg_sigma,
                                                                                                'bg_sigma':bg_sigma,
                                                                                                'beta':beta,
                                                                                                'gamma':gamma})
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

def visualize_masks(imgs, masks, masks_recon, recons, recons2, x_recon_s, file_name):
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
    x_recon_s *= 255.0
    masks_ims = [np.stack([masks[:, i, :, :]]*3, axis=1) for i in range(masks.shape[1])]
    masks_ims = [np.concatenate(np.transpose(m, (0, 2, 3, 1)), axis=1) for m in masks_ims]

    masks_recon_ims = [np.stack([masks_recon[:, i, :, :]]*3, axis=1) for i in range(masks_recon.shape[1])]
    masks_recon_ims = [np.concatenate(np.transpose(m, (0, 2, 3, 1)), axis=1) for m in masks_recon_ims]

    x_recon_s = x_recon_s.reshape((masks.shape[0], masks.shape[1], 3, masks.shape[2], masks.shape[2] ))
    x_recon_s_ims = [x_recon_s[:, i, :, :] for i in range(x_recon_s.shape[1])]
    x_recon_s_ims = [np.concatenate(np.transpose(m, (0, 2, 3, 1)), axis=1) for m in x_recon_s_ims]

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    imgs = np.concatenate(imgs, axis=1)
    seg_maps = np.transpose(seg_maps, (0, 2, 3, 1))
    seg_maps = np.concatenate(seg_maps, axis=1)
    recons = np.transpose(recons, (0, 2, 3, 1))
    recons = np.concatenate(recons, axis=1)
    recons2 = np.transpose(recons2, (0, 2, 3, 1))
    recons2 = np.concatenate(recons2, axis=1)
    all_list = [imgs, seg_maps, recons, recons2]+masks_ims+x_recon_s_ims+masks_recon_ims
    all_im_array = np.concatenate(all_list, axis=0)
    all_im = Image.fromarray(all_im_array.astype(np.uint8))
    all_im.save(file_name)

def train_Vae(batch_size, img_size, latent_size, train_file, vae_weights_path, beta, gamma, bg_sigma, fg_sigma,
              epochs=100, no_cuda=False, seed=1, log_interval=100, load=False,
              num_blocks=5, channel_base=64, num_slots=6,
              full_connected_size=256, color_channels=3,kernel_size=3, encoder_stride=2,decoder_stride=1,
              conv_size1=32, conv_size2=64):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    if load:
        model = Monet2_VAE(height=img_size, width=img_size, device=device, latent_size=latent_size,
                           num_blocks=num_blocks,
                           channel_base=channel_base, num_slots=num_slots, full_connected_size=full_connected_size,
                           color_channels=color_channels, kernel_size=kernel_size, encoder_stride=encoder_stride,
                           decoder_stride=decoder_stride, conv_size1=conv_size1, conv_size2=conv_size2).to(device)

        #todo check which optimizer is better
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
        #optimizer = optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = torch.load(vae_weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
    else:
        model = Monet2_VAE(height=img_size, width=img_size, device=device, latent_size=latent_size,
                           num_blocks=num_blocks,
                           channel_base=channel_base, num_slots=num_slots, full_connected_size=full_connected_size,
                           color_channels=color_channels, kernel_size=kernel_size, encoder_stride=encoder_stride,
                           decoder_stride=decoder_stride, conv_size1=conv_size1, conv_size2=conv_size2).to(device)
        #for w in model.parameters():
        #    std_init = 0.01
        #    nn.init.normal_(w, mean=0., std=std_init)
        #print('Initialized parameters')
        # todo check which optimizer is better
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
        #optimizer = optim.Adam(model.parameters(), lr=1e-4)
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
        mu_s, logvar_s, masks, full_reconstruction, full_reconstruction2, x_recon_s, mask_pred_s = model(data)

        visualize_masks(imgs=numpify(data),
                        masks=numpify(masks),
                        masks_recon=numpify(mask_pred_s),
                        recons=numpify(full_reconstruction),
                        recons2=numpify(full_reconstruction2),
                        x_recon_s=numpify(torch.cat(x_recon_s)),
                        file_name=this_file_dir+'results/reconstruction_{}.png'.format(filename_suffix))


def load_Vae(path, img_size, latent_size, no_cuda=False, seed=1, num_blocks=5, channel_base=64, num_slots=6,
                 full_connected_size=256, color_channels=3,kernel_size=3, encoder_stride=2,decoder_stride=1,
                 conv_size1=32, conv_size2=64):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = Monet2_VAE(height=img_size, width=img_size, device=device, latent_size=latent_size, num_blocks=num_blocks,
                       channel_base=channel_base, num_slots=num_slots, full_connected_size=full_connected_size,
                       color_channels=color_channels, kernel_size=kernel_size, encoder_stride=encoder_stride,
                       decoder_stride=decoder_stride, conv_size1=conv_size1, conv_size2=conv_size2).to(device)
    #todo see with which optimizer is better
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str)

    parser.add_argument('--enc_type', help='the type of attribute that we want to generate/encode', type=str,
                        default='all', choices=['all', 'goal', 'obstacle', 'obstacle_sizes', 'goal_sizes'])
    parser.add_argument('--batch_size', help='number of batch to train', type=np.float, default=4)#8)
    parser.add_argument('--train_epochs', help='number of epochs to train vae', type=np.int32, default=40)
    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=64)
    parser.add_argument('--latent_size', help='latent size to train the VAE', type=np.int32, default=6)
    parser.add_argument('--num_slots', help='number of slots', type=np.int32, default=8)
    parser.add_argument('--beta', help='beta val for the reconstruction loss', type=np.float, default=8.)#8.)#5#8
    parser.add_argument('--gamma', help='gamma val for the mask loss', type=np.float, default=5.)
    parser.add_argument('--bg_sigma', help='', type=np.float, default=0.09)
    parser.add_argument('--fg_sigma', help='', type=np.float, default=0.11)

    args = parser.parse_args()

    # get names corresponding folders, and files where to store data
    #make_dir(this_file_dir+'results/', clear=False)
    base_data_dir = this_file_dir + '../data/'
    data_dir = base_data_dir + args.env + '/'

    train_file = data_dir + train_file_name[args.enc_type]
    weights_path = data_dir + vae_sb_weights_file_name[args.enc_type]

    train_Vae(epochs=args.train_epochs, batch_size=args.batch_size,img_size=args.img_size,latent_size=args.latent_size,
              train_file=train_file,
              vae_weights_path=weights_path, beta=args.beta, gamma=args.gamma, bg_sigma=args.bg_sigma,
              fg_sigma=args.fg_sigma, load=False, num_slots=args.num_slots)


    '''device = torch.device("cuda")

    model = load_Vae(path=weights_path, img_size=args.img_size, latent_size=args.latent_size)


    compare_with_data_set(model=model, device=device, latent_size=args.latent_size,
                     filename_suffix="test", train_file=train_file )'''
