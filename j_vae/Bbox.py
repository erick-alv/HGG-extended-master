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


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions

        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.expand(B, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].expand(B, 2, height, width)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(shape=(x.size(0), -1))

'''class MaskDec(nn.Module):
    def __init__(self, num_slots, device, img_size=64, feat_size=128, full_connected_size=256, latent_size = 8):
        super().__init__()
        embed_size = img_size // 16
        self.num_slots = num_slots
        self.feat_size = feat_size
        self.device = device
        kernel_size = 3
        conv_size1 = 64
        conv_size2 = 32
        encoder_stride = 2
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size2,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size2, out_channels=conv_size2,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True)
        )
        #self.spatial_broadcast4d = SpatialBroadcast4D()
        self.img_size = img_size
        #todo calculate these values
        self.flatten = Flatten()
        self.red_width = red_width = 7
        self.red_height = red_height = 8
        self.fc1 = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
        )
        self.fc21 = nn.Linear(full_connected_size, latent_size)
        self.fc22 = nn.Linear(full_connected_size, latent_size)

        kernel_size = 3
        decoder_stride = 1
        # kernel_size = 3, stride = 1, padding =1, dilation =1 is a size preserving convolution

        self.spatial_broadcast = SpatialBroadcast()
        self.dec = nn.Sequential(
            nn.Conv2d(in_channels=latent_size + 2, kernel_size=kernel_size,
                      stride=decoder_stride, out_channels=img_size * 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=img_size * 2, kernel_size=kernel_size,
                      stride=decoder_stride, out_channels=img_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=img_size,
                      kernel_size=kernel_size, stride=decoder_stride, out_channels=img_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                      stride=decoder_stride, out_channels=4, padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def forward(self, x):
        B = x.shape[0]
        H = x.shape[2]
        W = x.shape[3]


        #division is just used so that the array is then divisible by num slots
        spacer = torch.zeros(size=[B, 3, 1, W]).to(device)
        # (B, 3, H, W) -> (B, 3, H+2, W)
        x_mod = torch.cat([spacer, x, spacer], dim=2)

        # (B, 3, H, W) -> (B, 3, (H+2)*K, W)
        x_repeat = torch.repeat_interleave(x_mod, self.num_slots, dim=2)

        # (B, 3, (H+2)*K, W) -> (B, some_channel, (redH)*K, redW)
        x_enc = self.enc(x_repeat)
        ch = x_enc.shape[1]
        # -> (B, some_channel, K, (redH),  redW)
        x_enc = x_enc.reshape(shape=(B, ch, self.num_slots, self.red_height, self.red_width))#todo verify is dividing correctly
        # -> (B, K, some_channel, (redH),  redW)
        x_enc = x_enc.permute([0, 2, 1, 3, 4])
        # -> (B*K, some_channel, (redH),  redW)
        x_enc = x_enc.reshape(shape=(B*self.num_slots, ch,  self.red_height, self.red_width))
        x_enc_flattened = self.flatten(x_enc)
        x_enc_red = self.fc1(x_enc_flattened)
        x_mu = self.fc21(x_enc_red)
        x_logvar = self.fc22(x_enc_red)
        # (B*K, latent_size)
        z = self.reparameterize(x_mu, x_logvar)
        # -> (B*K, latent_size + 2, W, H)
        z_sb = self.spatial_broadcast(z, self.img_size, self.img_size)
        #(B * K, 4, W, H) #one is the maks channel
        rec = self.dec(z_sb)
        # -> (B, K, 4, W, H)
        rec = rec.view(B, self.num_slots, rec.shape[1], rec.shape[2], rec.shape[3])
        x_recon = torch.sigmoid(rec[:, :, :3, :, :])
        mask_pred = torch.sigmoid(rec[:, :, 3:4, :, :])


        #to (B, S, D)
        #x = x.reshape(shape=(B, self.num_slots, -1))
        #(B, S,  L + 2, W, H)
        #x = self.spatial_broadcast4d(x, self.img_size, self.img_size)
        #(B, S,  L + 2, W, H) -> (B*S,  L + 2, W, H)
        #x = x.view(B*self.num_slots, x.shape[2], x.shape[3], x.shape[4])
        #(B*S, 1, W, H)
        #masks = self.dec(x)
        #(B, S, 1, W, H)
        #masks = masks.view(B, self.num_slots, 1, masks.shape[2], masks.shape[3])
        #return torch.sigmoid(masks)
        def get_kld(mu, logvar):
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return KLD
        kld = get_kld(x_mu, x_logvar)
        return x_recon, mask_pred, kld'''

class Bbox(nn.Module):
    def __init__(self, num_slots, device, img_size=64, full_connected_size=256, fg_sigma=0.15,latent_size=8):
        super().__init__()
        self.num_slots = num_slots
        self.device = device
        kernel_size = 3
        conv_size1 = 64
        conv_size2 = 32
        encoder_stride = 2
        self.img_size = img_size
        self.fg_sigma = fg_sigma
        self.encs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=conv_size1,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size1, out_channels=conv_size2,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=conv_size2, out_channels=conv_size2,
                      kernel_size=kernel_size, stride=encoder_stride),
            nn.ReLU(inplace=True)
        ) for _ in range(num_slots)])

        #todo calculate these values
        self.flatten = Flatten()
        self.red_width = red_width = 7
        self.red_height = red_height = 7

        self.z_pos_net = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
            nn.Linear(full_connected_size, 2 * 2)
        )
        self.z_scale_net = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
            nn.Linear(full_connected_size, 2*2)
        )
        self.z_pres_net = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
            nn.Linear(full_connected_size, 1)
        )
        self.z_depth_net = nn.Sequential(
            nn.Linear(conv_size2 * red_width * red_height, full_connected_size),
            nn.ReLU(inplace=True),
            nn.Linear(full_connected_size, 1*2)
        )
        
        self.z_pres_start_value = 0.1
        self.z_pres_end_value = 0.01
        self.z_pres_start_step = 4000
        self.z_pres_end_step = 10000
        self.register_buffer('prior_z_pres_prob', torch.tensor(self.z_pres_start_value))
        
        
        self.tau_start_value = 2.5
        self.tau_start_step = 0
        self.tau_end_step = 20000
        self.tau_end_value = 0.5
        self.register_buffer('tau', torch.tensor(self.tau_start_value))

        self.use_bg_mask = False
        


    '''def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std)'''

    def anneal(self, global_step):
        """
        Update everything

        :param global_step: global step (training)
        :return:
        """

        '''self.prior_z_pres_prob = linear_annealing(self.prior_z_pres_prob.device, global_step,
                                                  self.z_pres_start_step, self.z_pres_end_step,
                                                  self.z_pres_start_value, self.z_pres_end_value)
        self.tau = linear_annealing(self.tau.device, global_step,
                                    self.tau_start_step, self.tau_end_step,
                                    self.tau_start_value, self.tau_end_value)'''

        self.prior_z_pres_prob = torch.tensor(self.z_pres_start_value, device=self.device)
        self.tau = torch.tensor(self.tau_start_value, device=self.device)

        #if global_step > 20000 and self.use_bg_mask==False:
        if global_step > 8000:#10000:
            if self.use_bg_mask == False:
                print('!!!!!!!!!!!!!!! \n *** \n using bg mask \n *** \n!!!!!!!!!!!!!')
            self.use_bg_mask = True
        else:
            self.use_bg_mask = False



    @property
    def z_depth_prior(self):
        return Normal(0., 1.)

    @property
    def z_scale_prior(self):
        return Normal(0., 1.)

    @property
    def z_shift_prior(self):
        return Normal(0., 1.)

    def encode(self, x):
        B=x.shape[0]
        CH = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        # (B, 3, H, W) -> (B, 3, H*K, W)
        x_repeat = torch.repeat_interleave(x, self.num_slots, dim=2)
        # ->(B, 3,  H, K, W)
        x_repeat = x_repeat.reshape(shape=(B, CH, H, self.num_slots , W))
        # -> (K, B, 3, H, W)
        x_repeat_per_K = x_repeat.permute([3, 0, 1, 2, 4])
        # should be the same image for K times
        #save_image(x_repeat[0:self.num_slots, 0, :, :, :], 'correct_div.png')
        #K*[(B, channel, redH, redW)]
        x_encs = [self.encs[i](x_repeat_per_K[i]) for i in range(self.num_slots)]
        # -> (K, B, channel, redH, redW)
        x_encs = torch.stack(x_encs)
        # -> (K, B, channel, redH, redW) -> (B, K, channel, redH, redW) -> (B*K, channel, redH, redW)
        x_encs = x_encs.permute([1, 0, 2, 3, 4]).reshape(shape=(B*self.num_slots, x_encs.shape[2], x_encs.shape[3],
                                                                x_encs.shape[4]))
        # -> (B*K, channel* redH* redW)
        x_encs_flattened = self.flatten(x_encs)

        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(x_encs_flattened))
        #z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=self.tau)
        #z_pres_y = z_pres_post.rsample()
        #z_pres = torch.sigmoid(z_pres_y)
        z_pres = torch.sigmoid(z_pres_logits)

        #z_depth_mean, z_depth_std = self.z_depth_net(x_encs_flattened).chunk(2, 1)
        #z_depth_std = F.softplus(z_depth_std)
        #z_depth_post = Normal(z_depth_mean, z_depth_std)
        #z_depth = z_depth_post.rsample()
        z_depth = torch.zeros_like(z_pres).to(self.device)

        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(x_encs_flattened).chunk(2, 1)
        #z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        #z_scale_post = Normal(z_scale_mean, z_scale_std)
        #z_scale = z_scale_post.rsample()
        # to range (0, 1)
        #z_scale = z_scale.sigmoid()
        z_scale = z_scale_mean.sigmoid()

        z_pos_mean, z_pos_std = self.z_pos_net(x_encs_flattened).chunk(2, 1)
        #z_pos_std = F.softplus(z_pos_std)
        #z_pos_post = Normal(z_pos_mean, z_pos_std)
        #z_pos = z_pos_post.rsample()
        #z_pos = z_pos.tanh()
        z_pos = z_pos_mean.tanh()
        return z_pres.reshape(shape=(B, self.num_slots, 1)), z_depth.reshape(shape=(B, self.num_slots, 1)), \
               z_scale.reshape(shape=(B, self.num_slots, 2)), z_pos.reshape(shape=(B, self.num_slots, 2))


    def forward(self, x, global_step, ims_with_masks, masks):
        B = x.shape[0]
        CH = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        self.anneal(global_step)

        # (B, 3, H, W) -> (B, 3, H*K, W)
        x_repeat = torch.repeat_interleave(x, self.num_slots, dim=2)
        # ->(B, 3,  H, K, W)
        x_repeat = x_repeat.reshape(shape=(B, CH, H, self.num_slots , W))
        # -> (K, B, 3, H, W)
        x_repeat_per_K = x_repeat.permute([3, 0, 1, 2, 4])
        # should be the same image for K times
        #save_image(x_repeat[0:self.num_slots, 0, :, :, :], 'correct_div.png')
        #K*[(B, channel, redH, redW)]
        x_encs = [self.encs[i](x_repeat_per_K[i]) for i in range(self.num_slots)]
        # -> (K, B, channel, redH, redW)
        x_encs = torch.stack(x_encs)
        # -> (K, B, channel, redH, redW) -> (B, K, channel, redH, redW) -> (B*K, channel, redH, redW)
        x_encs = x_encs.permute([1, 0, 2, 3, 4]).reshape(shape=(B*self.num_slots, x_encs.shape[2], x_encs.shape[3],
                                                                x_encs.shape[4]))
        # -> (B*K, channel* redH* redW)
        x_encs_flattened = self.flatten(x_encs)

        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(x_encs_flattened))
        z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=self.tau)
        # Unbounded
        z_pres_y = z_pres_post.rsample()
        # in (0, 1)
        z_pres = torch.sigmoid(z_pres_y)


        z_depth_mean, z_depth_std = self.z_depth_net(x_encs_flattened).chunk(2, 1)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_post = Normal(z_depth_mean, z_depth_std)
        z_depth = z_depth_post.rsample()

        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(x_encs_flattened).chunk(2, 1)
        z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        z_scale_post = Normal(z_scale_mean, z_scale_std)
        z_scale = z_scale_post.rsample()
        # to range (0, 1)
        z_scale = z_scale.sigmoid()

        z_pos_mean, z_pos_std = self.z_pos_net(x_encs_flattened).chunk(2, 1)
        z_pos_std = F.softplus(z_pos_std)
        z_pos_post = Normal(z_pos_mean, z_pos_std)
        z_pos = z_pos_post.rsample()
        z_pos = z_pos.tanh()

        kl_loss = self.calculate_kl_loss(z_pres_logits, z_depth_post, z_scale_post, z_pos_post)

        z_where = torch.cat((z_scale, z_pos), dim=-1)
        # z_where has (B*K, ...); therefore x_repeat must have same
        dec_type = 1
        if dec_type == 0:
            x_repeat = x_repeat.permute([0, 3, 1, 2, 4]).reshape(shape=(B * self.num_slots, 3, H, W))
            extended = spatial_transform(image=x_repeat, z_where=z_where,
                                         out_dims=(B * self.num_slots, 3, H, W),
                                         inverse=False)
            resize_normal = spatial_transform(image=extended, z_where=z_where,
                                              out_dims=(B * self.num_slots, 3, H, W),
                                              inverse=True)
            # mutiply with the presence variable
            # (B*K, 3, H, W)
            ims_with_masks_recs = resize_normal * z_pres.view(-1, 1, 1, 1)

            # reshaping ims with mask
            # (B, K+1, 3, H, W) -> (B, K, 3, H) eliminate mask from background
            ims_with_masks_fg = ims_with_masks[:, 1:, :, :, :]
            # -> (B*K, 3, H, W)
            ims_with_masks_fg = ims_with_masks_fg.reshape(shape=(B * self.num_slots, 3, H, W))
            # calculate the likelihood
            fg_dist = Normal(ims_with_masks_recs, self.fg_sigma)
            fg_likelihood = fg_dist.log_prob(ims_with_masks_fg)
            log_like = fg_likelihood
        elif dec_type==1:
            #important z_where is (B, K,...)  meaning per element in batch must be same image/mask k times
            # (B, K+1, 3, H, W) -> (B, K, 3, H) eliminate mask from background
            masks_fg = masks[:, 1:, :, :, :]

            #compose fg_masks in one
            #masks_fg = torch.sum(masks_fg, dim=1)
            #masks_fg = masks_fg.reshape(shape=(B, 1, 1, H, W)).\
            #    repeat([1, self.num_slots, 1, 1, 1]).reshape(shape=(B*self.num_slots, 1, H, W))
            #save_image(masks_fg, 'composed_mask.png', nrow=self.num_slots)
            # -> (B*K, 1, H, W)

            #use masks but not compose
            masks_fg = masks_fg.reshape(shape=(B * self.num_slots, 1, H, W))

            masks_cut = spatial_transform(image=masks_fg, z_where=z_where,
                                         out_dims=(B * self.num_slots, 1, H, W),
                                         inverse=False)
            masks_recs = spatial_transform(image=masks_cut, z_where=z_where,
                                         out_dims=(B * self.num_slots, 1, H, W),
                                         inverse=True)
            x_repeat = x_repeat.permute([0, 3, 1, 2, 4]).reshape(shape=(B * self.num_slots, 3, H, W))
            ims_with_masks_recs = x_repeat * masks_recs * z_pres.view(-1, 1, 1, 1)

            # (B, K+1, 3, H, W) -> (B, K, 3, H) eliminate mask from background
            ims_with_masks_fg = ims_with_masks[:, 1:, :, :, :]
            # -> (B*K, 3, H, W)
            ims_with_masks_fg = ims_with_masks_fg.reshape(shape=(B * self.num_slots, 3, H, W))
            # calculate the likelihood
            fg_dist = Normal(ims_with_masks_recs, self.fg_sigma)
            fg_likelihood = fg_dist.log_prob(ims_with_masks_fg)
            #log_like = fg_likelihood

            ones = torch.ones(size=(B * self.num_slots, 1, H, W)).to(self.device)
            # (B * K, 1, H, W)
            ones_cuts = spatial_transform(image=ones, z_where=z_where,
                                          out_dims=(B * self.num_slots, 1, H, W),
                                          inverse=True)
            # (B * K, 1, H, W) this should leave just the valid boxes
            ones_cuts = ones_cuts * z_pres.view(-1, 1, 1, 1)
            # (B * K, 1, H, W) -> (B,K, 1, H, W) -> (B, 1, H, W) blend Bboxes
            ones_cuts_mixed = ones_cuts.reshape((B, self.num_slots, 1, H, W)).sum(dim=1)
            bbox_omega = torch.ones(size=(B, 1, H, W)).to(self.device) - ones_cuts_mixed
            # (B, 3, H, W)
            ims_with_masks_bg_rec = x * bbox_omega

            if self.use_bg_mask:
                #todo calculate with respect to bg mask and not the image
                #bg_dist = Normal(ims_with_masks_bg_rec, self.fg_sigma)
                #bg_likelihood = bg_dist.log_prob(ims_with_masks[:, 0, :, :, :])
                bg_dist = Normal(bbox_omega.repeat([1, 3, 1, 1]), self.fg_sigma)
                bg_likelihood = bg_dist.log_prob(masks[:, 0, :, :, :].repeat([1, 3, 1, 1]))
                #bg_likelihood2 = bg_likelihood * (1/self.num_slots) *0.2#since we will replicate divide it
                bg_likelihood = bg_likelihood * (1/self.num_slots)
                bg_likelihood = bg_likelihood.reshape(shape=(B, 1, 3, H, W)).repeat([1, self.num_slots, 1, 1, 1]).reshape(shape=(B * self.num_slots, 3, H, W))

                #other likelihood to avoid that the bounding boxes do not cover part of object/ forget of object at all
                covered_m = ims_with_masks_fg - ones_cuts
                covered_m = torch.clamp(covered_m, min=0.)
                empty_background = torch.zeros_like(covered_m).to(self.device)
                all_cover_dist = Normal(covered_m, self.fg_sigma)
                all_cover_likelihood = all_cover_dist.log_prob(empty_background)

                log_like = torch.stack((fg_likelihood, bg_likelihood, all_cover_likelihood), dim=1)

                log_like = torch.logsumexp(log_like, dim=1)


            else:
                log_like = fg_likelihood

            # (B, 1, 3, H, W)
            ims_with_masks_bg_rec= ims_with_masks_bg_rec.reshape((B, 1, 3, H, W))
            ims_with_masks_recs = torch.cat([ims_with_masks_recs.reshape(shape=(B, self.num_slots, 3, H, W)),
                       ims_with_masks_bg_rec], dim=1)


        '''
        fg_bbox_labels = bbox_data[:, 1:, :4]
        labels_where = fg_bbox_labels.reshape(shape=(B*self.num_slots, 4))
        real_zooms = spatial_transform(image=x_repeat, z_where=labels_where,
                                     out_dims=(B*self.num_slots, 3, H, W),
                                     inverse=False)
        #save_image(x_repeat, 'current_images.png', nrow=self.num_slots)
        #save_image(real_zooms, 'real_zooms.png', nrow=self.num_slots)
        #resize_normal = spatial_transform(image=extended, z_where=labels_where,
        #                                  out_dims=(B * self.num_slots, 3, H, W),
        #                                  inverse=True)'''

        '''#substract bBoxes of image to compare to background, return elements, indices with shape (B, 3, H, W)
        rec_combined, combined_indices = torch.max(ims_with_masks_recs.reshape(shape=(B, self.num_slots, 3, H, W)), dim=1)
        bboxes_substracted = x - rec_combined
        #add to recognized
        bg_dist = Normal(bboxes_substracted, self.fg_sigma)#todo should use other sigma?
        bg_likelihood = bg_dist.log_prob(ims_with_masks[:, 0, :, :, :])
        #repeats it along every
        bg_likelihood = bg_likelihood.repeat(self.num_slots, 1, 1, 1) * (1/self.num_slots)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        log_like = torch.logsumexp(log_like, dim=1)'''



        log_like = log_like.flatten(start_dim=1).sum(1)
        elbo = log_like - kl_loss
        #one wants to maximize the elbo, therefore we make it negative. So the optimizer minimizes it
        loss = (-elbo).mean()

        #final_recs = ims_with_masks_recs.reshape(shape=(B, self.num_slots, 3, H, W))
        final_recs = ims_with_masks_recs
        #final_recs = torch.cat([bboxes_substracted.reshape(shape=(B, 1, 3, H, W)), ims_with_masks_recs.reshape(shape=(B, self.num_slots, 3, H, W))], dim=1)

        return z_pres, z_depth, z_scale, z_pos, z_where, loss, final_recs





    def calculate_kl_loss(self, z_pres_logits, z_depth_post, z_scale_post, z_pos_post):
        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)
        kl_z_depth = kl_divergence(z_depth_post, self.z_depth_prior)
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        kl_z_pos = kl_divergence(z_pos_post, self.z_shift_prior)
        # Reduce (B, G*G, D) -> (B,)
        kl_z_pres, kl_z_depth, kl_z_scale, kl_z_pos = [
            x.flatten(start_dim=1).sum(1) for x in [kl_z_pres, kl_z_depth, kl_z_scale, kl_z_pos]
        ]
        kl = kl_z_scale + kl_z_pos + kl_z_pres + kl_z_depth
        return kl


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

def preprocess_bounding_boxes(data_set_masks):
    def define_bbox(start_ind_x,end_ind_x,start_ind_y, end_ind_y, W, H):
        mid_x = (start_ind_x + end_ind_x) / 2.
        mid_y = (start_ind_y + end_ind_y) / 2.
        bbox_center_x = -1 + mid_x/float(W)*2.
        bbox_center_y = -1 + mid_y / float(H) * 2.
        width_x = (end_ind_x - start_ind_x) / 2.
        width_y = (end_ind_y - start_ind_y) / 2.
        bbox_size_x = width_x / (0.5*float(W))
        bbox_size_y = width_y / (0.5 * float(H))
        return bbox_size_x, bbox_size_y, bbox_center_x, bbox_center_y

    B = len(data_set_masks)
    K = len(data_set_masks[0])
    b_box_info = np.zeros(shape=(B, K, 5))
    data_set_masks = data_set_masks / 255.
    ind_small = data_set_masks <= 1e-4
    data_set_masks[ind_small] = 0.
    for b in range(B):
        for k in range(K):
            mask = data_set_masks[b, k]
            #show_im(255. * np.repeat(mask, 3, axis=2))
            border_indices = np.where(mask > 1e-4)
            if len(border_indices[2]) == 0:
                b_box_info[b, k] = np.array([0., 0., 0., 0., 0.])
            else:
                bbox_size_x, bbox_size_y, bbox_center_x, bbox_center_y = define_bbox(start_ind_x=border_indices[0].min(), end_ind_x=border_indices[0].max(),
                                          start_ind_y=border_indices[1].min(), end_ind_y=border_indices[1].max(),
                                          W=mask.shape[0], H=mask.shape[1])
                b_box_info[b, k] = np.array([bbox_size_x, bbox_size_y, bbox_center_x, bbox_center_y, 1.])
    return b_box_info

def train(model, optimizer, device, log_interval_epoch, log_interval_batch,  batch_size):
    model.train()
    train_loss = 0
    data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy')
    data_size = len(data_set)
    #bbox_labels = np.load('../data/FetchGenerativeEnv-v1/all_set_with_masks_bbox.npy')
    global_step = 0
    for epoch in range(300):
        #creates indexes and shuffles them. So it can acces the data
        idx_set = np.arange(data_size)
        np.random.shuffle(idx_set)
        #idx_set = idx_set[:5120]#todo use all set
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        for batch_idx, idx_select in enumerate(idx_set):
            data = data_set[idx_select]
            #show_im(data[0][0].copy())
            data = torch.from_numpy(data).float().to(device)
            data /= 255.
            #bbox_data = bbox_labels[idx_select]
            #bbox_data = torch.from_numpy(bbox_data).to(device)

            #
            input_ims = data[:, 0, :, :, :]
            input_ims = input_ims.permute([0, 3, 1, 2])
            orig_masks = data[:, 1:, :, :, 0:1]#also leave just one color channel since it is a mask converted to rgb
            orig_masks = orig_masks.permute([0, 1, 4, 2, 3])
            imgs_with_mask = data[:, 0:1, :, :, :]
            imgs_with_mask = imgs_with_mask.permute([0, 1, 4, 2, 3])
            imgs_with_mask = imgs_with_mask*orig_masks
            optimizer.zero_grad()
            #save_image(imgs_with_mask[0], 'mask_division.png')
            z_pres, z_depth, z_scale, z_pos, z_where, loss, \
            ims_with_masks_recs = model(input_ims, global_step, imgs_with_mask, orig_masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % log_interval_batch == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), data_size,
                           100. * (batch_idx + 1) / len(data_set),
                           loss.item() / len(data)))
                print('Loss: ', loss.item() / len(data))

            global_step += 1
        if epoch % log_interval_epoch == 0:
            visualize_masks(input_ims, imgs_with_mask, orig_masks, ims_with_masks_recs,
                            'recs_{}_{}.png'.format(epoch, batch_idx), z_pres=z_pres)
            save_checkpoint(model, optimizer, '../data/FetchGenerativeEnv-v1/model_bbox', epoch=epoch)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / data_size))
    visualize_masks(input_ims, imgs_with_mask, orig_masks, ims_with_masks_recs,
                    'recs_{}_{}.png'.format(epoch, batch_idx), z_pres=z_pres)
    save_checkpoint(model, optimizer, '../data/FetchGenerativeEnv-v1/model_bbox')


def visualize_masks(orig_imgs, imgs_with_mask, orig_masks, rec_imgs_with_mask, file_name, z_pres=None):
    img_show = imgs_with_mask.permute([1, 0, 2, 3, 4]).reshape(shape=(imgs_with_mask.shape[0] * imgs_with_mask.shape[1],
                                                                      imgs_with_mask.shape[2], imgs_with_mask.shape[3],
                                                                      imgs_with_mask.shape[4]))


    orig_masks = orig_masks.repeat(1, 1, 3, 1, 1)
    orig_masks_show = orig_masks.permute([1, 0, 2, 3, 4]).\
        reshape(shape=(orig_masks.shape[0] * orig_masks.shape[1], orig_masks.shape[2],
                       orig_masks.shape[3], orig_masks.shape[4]))

    rec_imgs_mask_show = rec_imgs_with_mask.permute([1, 0, 2, 3, 4]). \
        reshape(shape=(rec_imgs_with_mask.shape[0] * rec_imgs_with_mask.shape[1], rec_imgs_with_mask.shape[2],
                       rec_imgs_with_mask.shape[3], rec_imgs_with_mask.shape[4]))

    '''rec_masks = rec_masks.repeat(1, 1, 3, 1, 1)
    rec_masks_show = rec_masks.permute([1, 0, 2, 3, 4]). \
        reshape(shape=(rec_masks.shape[0] * rec_masks.shape[1], rec_masks.shape[2],
                       rec_masks.shape[3], rec_masks.shape[4]))

    img_show = torch.cat([orig_imgs, img_show, orig_masks_show, rec_imgs_mask_show, rec_masks_show])'''
    img_show = torch.cat([orig_imgs, img_show, orig_masks_show, rec_imgs_mask_show])
    save_image(img_show, file_name, nrow=imgs_with_mask.shape[0], pad_value=0.3)

    if z_pres is not None:
        print(z_pres)



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


def load_Model(path, img_size, latent_size, device, num_slots, seed=1,
                 full_connected_size=256, fg_sigma=0.15):
    torch.manual_seed(seed)
    model = Bbox(num_slots, device, img_size=img_size, latent_size=latent_size,
                 full_connected_size=full_connected_size, fg_sigma=fg_sigma).to(device)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def rec_dataset(model, dataset_path):
    model.eval()
    train_loss = 0
    data_set = np.load(dataset_path)
    data_size = len(data_set)
    global_step = 0

    # creates indexes and shuffles them. So it can acces the data
    idx_set = np.arange(data_size)
    np.random.shuffle(idx_set)
    idx_set = np.split(idx_set, len(idx_set) / 16)
    idx_select = idx_set[0]
    data = data_set[idx_select]
    data = torch.from_numpy(data).float().to(device)
    data /= 255.
    input_ims = data[:, 0, :, :, :]
    input_ims = input_ims.permute([0, 3, 1, 2])
    orig_masks = data[:, 1:, :, :, 0:1]  # also leave just one color channel since it is a mask converted to rgb
    orig_masks = orig_masks.permute([0, 1, 4, 2, 3])
    imgs_with_mask = data[:, 0:1, :, :, :]
    imgs_with_mask = imgs_with_mask.permute([0, 1, 4, 2, 3])
    imgs_with_mask = imgs_with_mask * orig_masks
    # save_image(imgs_with_mask[0], 'mask_division.png')
    z_pres, z_depth, z_scale, z_pos, z_where, loss, \
    ims_with_masks_recs = model(input_ims, global_step, imgs_with_mask, orig_masks)

    visualize_masks(input_ims, imgs_with_mask, orig_masks, ims_with_masks_recs,
                    'recs_test.png', z_pres=z_pres)

if __name__ == '__main__':
    device = 'cuda:0'
    #one less since background mask is not used
    seed=1
    torch.manual_seed(seed)
    model = Bbox(5, device).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    train(model, optimizer, device, log_interval_epoch=10, log_interval_batch=400, batch_size=16)#4)


    '''seed = 1
    torch.manual_seed(seed)
    model = load_Model('../data/FetchGenerativeEnv-v1/model_bbox', 64, 8, device, 5)
    rec_dataset(model, '../data/FetchGenerativeEnv-v1/double_env_with_masks.npy')'''



    '''data_set = np.load('../data/FetchGenerativeEnv-v1/all_set.npy')
    #just take half the data
    data_set = data_set[:19200]
    device = 'cuda:0'
    from j_vae.train_monet import load_Vae, visualize_masks
    model = load_Vae(path='../data/FetchGenerativeEnv-v1/all_sb_model', img_size=64, latent_size=6)
    model = model.to(device)
    data_size = len(data_set)
    new_data_set = np.zeros(shape=(data_size, 6 + 1, 64, 64, 3))
    batch_size = 10
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

            #visualize_masks(np.transpose(data_np.copy()/255., axes=[0, 3, 1, 2]),
            #                np.squeeze(masks.copy()),
            #                np.transpose(data_np.copy()/255., axes=[0, 3, 1, 2]), 'results/new_data.png')
            # (B, S, 3, H, W)
            masks = np.tile(masks, (1, 1, 3, 1, 1))
            # to (B, S, H, W, 3)
            masks = np.transpose(masks, axes=(0, 1, 3, 4, 2))



            # (B, S+1, H, W, 3)
            #make masks visible in image
            masks *= 255.
            ims = np.concatenate([np.expand_dims(data_np, axis=1), masks], axis=1)
            new_data_set[batch_idx*batch_size:batch_idx*batch_size + batch_size] = ims
            #show_im(np.concatenate([i for i in ims[0]], axis=0))
    np.save('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy', new_data_set)'''
    '''new_data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy')
    bbox_info = preprocess_bounding_boxes(new_data_set[:, 1:, :, :, 0:1].copy())
    np.save('../data/FetchGenerativeEnv-v1/all_set_with_masks_bbox.npy', bbox_info)'''