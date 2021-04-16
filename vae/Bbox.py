#based on original SPACE https://github.com/zhixuan-lin/SPACE
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image
from torch import optim
import argparse
import numpy as np
from vae.SPACE_utils import spatial_transform, linear_annealing, NumericalRelaxedBernoulli, kl_divergence_bern_bern
from vae.train_monet import load_Vae as load_Monet

mse_loss = nn.MSELoss()

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(shape=(x.size(0), -1))


class Bbox(nn.Module):
    def __init__(self, num_slots, device, img_size=64, full_connected_size=256, fg_sigma=0.15):
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
        '''self.z_pres_end_value = 0.01
        self.z_pres_start_step = 4000
        self.z_pres_end_step = 10000'''
        self.register_buffer('prior_z_pres_prob', torch.tensor(self.z_pres_start_value))
        
        
        self.tau_start_value = 2.5
        '''self.tau_start_step = 0
        self.tau_end_step = 20000
        self.tau_end_value = 0.5'''
        self.register_buffer('tau', torch.tensor(self.tau_start_value))

        self.use_bg_mask = False
        self.use_ind_extra_loss = False

    def anneal(self, global_step):
        """
        Update everything

        :param global_step: global step (training)
        :return:
        """
        self.prior_z_pres_prob = torch.tensor(self.z_pres_start_value, device=self.device)
        self.tau = torch.tensor(self.tau_start_value, device=self.device)

        if global_step > 8000:
            if self.use_bg_mask == False:
                print('!!!!!!!!!!!!!!! \n *** \n using bg mask \n *** \n!!!!!!!!!!!!!')
                self.use_bg_mask = True
        else:
            self.use_bg_mask = False

        if global_step > 18000:
            if self.use_ind_extra_loss == False:
                print('!!!!!!!!!!!!!!! \n *** \n using ind extra \n *** \n!!!!!!!!!!!!!')
                self.use_ind_extra_loss = True
        else:
            self.use_ind_extra_loss = False

    #
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
        z_pres = torch.sigmoid(z_pres_logits)
        z_depth = torch.zeros_like(z_pres).to(self.device)

        z_scale_mean, _z_scale_std = self.z_scale_net(x_encs_flattened).chunk(2, 1)
        z_scale = z_scale_mean.sigmoid()

        z_pos_mean, z_pos_std = self.z_pos_net(x_encs_flattened).chunk(2, 1)
        z_pos = z_pos_mean.tanh()
        return z_pres.reshape(shape=(B, self.num_slots, 1)), z_depth.reshape(shape=(B, self.num_slots, 1)), \
               z_scale.reshape(shape=(B, self.num_slots, 2)), z_pos.reshape(shape=(B, self.num_slots, 2))


    def forward(self, x, global_step, ims_with_masks, masks, bg_image=None):
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
        for t in kl_loss:
            assert not torch.isnan(t)
            assert not torch.isinf(t)
        z_where = torch.cat((z_scale, z_pos), dim=-1)

        #important z_where is (B, K,...)  meaning per element in batch must be same image/mask k times
        # (B, K+1, 3, H, W) -> (B, K, 3, H) eliminate mask from background
        masks_fg = masks[:, 1:, :, :, :]
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

        extra_loss = torch.zeros(size=[]).to(x.device)
        if self.use_ind_extra_loss and bg_image is not None:
            with torch.no_grad():
                index=np.random.randint(low=0, high=self.num_slots)
                index = int(index)
                modified_ims = mod_ims(ims=x.clone(), masks=masks_fg.clone().reshape(B, self.num_slots, 1, H, W),
                                       bg_image=bg_image, index=index, p_all=z_pos.clone(),d_all=z_scale.clone())
            mod_x_repeat = torch.repeat_interleave(
                modified_ims, self.num_slots, dim=2
            ).reshape(shape=(B, CH, H, self.num_slots, W))
            mod_x_repeat_per_K = mod_x_repeat.permute([3, 0, 1, 2, 4])
            mod_x_encs = [self.encs[i](mod_x_repeat_per_K[i]) for i in range(self.num_slots)]
            mod_x_encs = torch.stack(mod_x_encs)
            mod_x_encs = mod_x_encs.permute([1, 0, 2, 3, 4]).reshape(
                shape=(B * self.num_slots, mod_x_encs.shape[2], mod_x_encs.shape[3],
                       mod_x_encs.shape[4]))
            mod_x_encs_flattened = self.flatten(mod_x_encs)

            mod_z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(mod_x_encs_flattened))
            mod_z_pres_post = NumericalRelaxedBernoulli(logits=mod_z_pres_logits, temperature=self.tau)
            mod_z_pres_y = mod_z_pres_post.rsample()
            mod_z_pres = torch.sigmoid(mod_z_pres_y)

            scale_std_bias = 1e-15
            mod_z_scale_mean, mod__z_scale_std = self.z_scale_net(mod_x_encs_flattened).chunk(2, 1)
            mod_z_scale_std = F.softplus(mod__z_scale_std) + scale_std_bias
            mod_z_scale_post = Normal(mod_z_scale_mean, mod_z_scale_std)
            mod_z_scale = mod_z_scale_post.rsample()
            # to range (0, 1)
            mod_z_scale = mod_z_scale.sigmoid()

            mod_z_pos_mean, mod_z_pos_std = self.z_pos_net(mod_x_encs_flattened).chunk(2, 1)
            mod_z_pos_std = F.softplus(mod_z_pos_std)
            mod_z_pos_post = Normal(mod_z_pos_mean, mod_z_pos_std)
            mod_z_pos = mod_z_pos_post.rsample()
            mod_z_pos = mod_z_pos.tanh()

            mod_z_pres = mod_z_pres.reshape(B, self.num_slots, 1)
            mod_z_scale = mod_z_scale.reshape(B, self.num_slots, 2)
            mod_z_pos = mod_z_pos.reshape(B, self.num_slots, 2)
            mod_z = torch.cat([mod_z_pres, mod_z_scale, mod_z_pos], dim=2)
            idx_0_setter = torch.ones_like(mod_z).to(x.device)
            idx_0_setter[:, index, :] = 0
            mod_z *= idx_0_setter


            c_z_pres = z_pres.clone().reshape(B, self.num_slots, 1)
            c_z_scale = z_scale.clone().reshape(B, self.num_slots, 2)
            c_z_pos = z_pos.clone().reshape(B, self.num_slots, 2)
            c_z = torch.cat([c_z_pres, c_z_scale, c_z_pos], dim=2)
            c_z *= idx_0_setter
            z_extra_loss = mse_loss(mod_z, c_z)
            extra_loss = z_extra_loss
            assert not torch.isnan(extra_loss)
            assert not torch.isinf(extra_loss)
            #calculate loss of difference with

        if self.use_bg_mask:
            bg_dist = Normal(bbox_omega.repeat([1, 3, 1, 1]), self.fg_sigma)
            bg_likelihood = bg_dist.log_prob(masks[:, 0, :, :, :].repeat([1, 3, 1, 1]))
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

        log_like = log_like.flatten(start_dim=1).sum(1)
        for t in log_like:
            assert not torch.isnan(t)
            assert not torch.isinf(t)
        elbo = log_like - kl_loss
        #one wants to maximize the elbo, therefore we make it negative. So the optimizer minimizes it
        loss = (-elbo).mean()+extra_loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        #final_recs = ims_with_masks_recs.reshape(shape=(B, self.num_slots, 3, H, W))
        final_recs = ims_with_masks_recs
        #final_recs = torch.cat([bboxes_substracted.reshape(shape=(B, 1, 3, H, W)), ims_with_masks_recs.reshape(shape=(B, self.num_slots, 3, H, W))], dim=1)

        return z_pres, z_depth, z_scale, z_pos, z_where, loss, final_recs


    def calculate_kl_loss(self, z_pres_logits, z_depth_post, z_scale_post, z_pos_post):

        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)
        for t in kl_z_pres:
            assert not torch.isnan(t)
            assert not torch.isinf(t)
        z_scale_post.scale = torch.clamp(z_scale_post.scale, min=1e-16)#avoid kl loss becoming inf
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        for t in kl_z_scale.flatten():
            assert not torch.isnan(t)
            assert not torch.isinf(t)
        z_pos_post.scale = torch.clamp(z_pos_post.scale, min=1e-16)  # avoid kl loss becoming inf
        kl_z_pos = kl_divergence(z_pos_post, self.z_shift_prior)
        for t in kl_z_pos.flatten():
            assert not torch.isnan(t)
            assert not torch.isinf(t)
        # Reduce (B, G*G, D) -> (B,)
        kl_z_pres, kl_z_scale, kl_z_pos = [
            x.flatten(start_dim=1).sum(1) for x in [kl_z_pres, kl_z_scale, kl_z_pos]
        ]
        kl = kl_z_scale + kl_z_pos + kl_z_pres
        return kl


def mod_ims(ims, masks, bg_image, index, p_all, d_all):

    B, S, _, H, W = masks.shape
    C = 3
    ims = ims.reshape(B, 1, C, H, W)
    els = ims * masks
    # pos (B*S, 2) in [-1, 1]
    p_all = p_all.reshape(B, S, 2)
    d_all = d_all.reshape(B, S, 2)
    p = p_all[:, index, :]
    d = d_all[:, index, :]
    z_where = torch.cat([d, p], dim=-1)
    p_repos = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(B, 2))).to(ims.device).float()
    p_repos = p_repos.float()
    dim_change = torch.from_numpy(np.random.uniform(low=-0.02, high=0.02, size=(B, 2))).to(ims.device).float()
    d_repos = d + dim_change
    d_repos = torch.clamp(d_repos, min=0, max=1)
    z_where_repos = torch.cat( [d_repos, p_repos], dim=-1)

    els_to_mod = els[:, index, :, :, :]
    masks_to_mod = masks[:, index, :, :, :]

    els_r1 = spatial_transform(image=els_to_mod, z_where=z_where, out_dims=(B, 3, H, W),
                               inverse=False)
    m_r1 = spatial_transform(image=masks_to_mod, z_where=z_where, out_dims=(B, 1, H, W),
                             inverse=False)
    els_mod = spatial_transform(image=els_r1, z_where=z_where_repos, out_dims=(B, 3, H, W),
                                inverse=True)
    m_mod = spatial_transform(image=m_r1, z_where=z_where_repos, out_dims=(B, 1, H, W),
                              inverse=True)

    # masks from other elements; user to avoid that modified element is over them
    o_masks = masks.clone()
    o_masks[:, index, :, :, :] = 0
    o_masks_sum = torch.sum(o_masks, dim=1)
    o_masks_sum = torch.clamp(o_masks_sum, max=1.)
    m_mod = m_mod - o_masks_sum
    m_mod = torch.clamp(m_mod, min=0.)
    masks[:, index, :, :, :] = m_mod
    els_remod = m_mod * els_mod
    els[:, index, :, :, :] = els_remod
    masks_sum = torch.sum(masks, dim=1)
    masks_sum = torch.clamp(masks_sum, max=1)

    new_im = torch.ones(size=(B, C, H, W)).to(device)
    new_im *= bg_image
    new_im -= masks_sum
    new_im = torch.clamp(new_im, min=0)
    new_els = torch.sum(els * masks, dim=1)
    new_im += new_els
    return new_im

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

def train(model, optimizer, device, log_interval_epoch, log_interval_batch, batch_size, num_epochs, resume_path = None, resume_on_epoch=None):
    train_loss = 0

    bg_im = np.load('../data/FetchGenerativeEnv-v1/bg_im.npy')
    bg_im = torch.from_numpy(bg_im).float().to(device)
    bg_im /= 255.
    bg_im = bg_im.permute([2, 0, 1])

    data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy')
    data_size = len(data_set)

    if resume_path is not None:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = resume_on_epoch + 1
        num_epochs = num_epochs - resume_on_epoch
        global_step = int(resume_on_epoch * (data_size//batch_size) + 1)
    else:
        start_epoch = 0
        global_step = 0

    model.train()
    for epoch in range(start_epoch, num_epochs + start_epoch):
        #creates indexes and shuffles them. So it can acces the data
        idx_set = np.arange(data_size)
        np.random.shuffle(idx_set)
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        for batch_idx, idx_select in enumerate(idx_set):
            data = data_set[idx_select]
            #show_im(data[0][0].copy())
            data = torch.from_numpy(data).float().to(device)
            data /= 255.
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
            ims_with_masks_recs = model(input_ims, global_step, imgs_with_mask, orig_masks, bg_im)
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
        if epoch % log_interval_epoch == 0 or epoch == num_epochs -1:
            visualize_masks(input_ims, imgs_with_mask, orig_masks, ims_with_masks_recs,
                            'recs_{}_{}.png'.format(epoch, batch_idx), z_pres=z_pres)
            save_checkpoint(model, optimizer, '../data/FetchGenerativeEnv-v1/model_bboxv2', epoch=epoch)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / data_size))
    visualize_masks(input_ims, imgs_with_mask, orig_masks, ims_with_masks_recs,
                    'recs_{}_{}.png'.format(epoch, batch_idx), z_pres=z_pres)
    save_checkpoint(model, optimizer, '../data/FetchGenerativeEnv-v1/model_bboxv2')


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

    img_show = torch.cat([orig_imgs, img_show, orig_masks_show, rec_imgs_mask_show])
    save_image(img_show, file_name, nrow=imgs_with_mask.shape[0], pad_value=0.3)



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


def load_Model(path, img_size, device, num_slots, seed=1,
                 full_connected_size=256, fg_sigma=0.15):
    torch.manual_seed(seed)
    model = Bbox(num_slots, device, img_size=img_size,
                 full_connected_size=full_connected_size, fg_sigma=fg_sigma).to(device)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def rec_dataset(model, dataset_path):
    model.eval()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str)
    parser.add_argument('--task', help='prepare training data with monet; train or test BboxEncoder', type=str,
                        choices=['prepare','train', 'test', 'try'], required=True)

    parser.add_argument('--batch_size', help='number of batch to train', type=np.float, default=16)
    parser.add_argument('--lr', help='learninf rate', type=np.float, default=1e-4)
    parser.add_argument('--train_epochs', help='number of epochs to train vae', type=np.int32, default=500)
    parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=64)
    # one less since background mask is not used
    parser.add_argument('--num_slots', help='number of slots', type=np.int32, default=5)
    parser.add_argument('--fg_sigma', help='', type=np.float, default=0.15)

    args = parser.parse_args()
    device = 'cuda:0'
    seed=1
    torch.manual_seed(seed)
    if args.task == 'prepare':
        data_set = np.load('../data/FetchGenerativeEnv-v1/all_set.npy')
        # just take half the data #todo use all data??
        data_set = data_set[:19200]
        model = load_Monet(path='../data/FetchGenerativeEnv-v1/all_sb_model', img_size=args.img_size, latent_size=6)
        model = model.to(device)
        data_size = len(data_set)
        #todo make this configurable
        new_data_set = np.zeros(shape=(data_size, 6 + 1, args.img_size, args.img_size, 3))
        batch_size = 10
        # creates indexes and shuffles them. So it can acces the data
        idx_set = np.arange(data_size)
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        aa = 0
        with torch.no_grad():
            for batch_idx, idx_select in enumerate(idx_set):
                data_np = data_set[idx_select]
                data = np_data_batch_to_torch(data_np.copy(), device)
                masks = model.get_masks(data)
                def create_all_background(ims, masks):
                    B, S, C, H, W = masks.shape
                    bg_Im = ims * masks[:, 0, :, :, :]
                    mean_c0 = bg_Im[:, 0, :, :][bg_Im[:, 0, :, :] > 0.0001].mean()
                    mean_c1 = bg_Im[:, 1, :, :][bg_Im[:, 1, :, :] > 0.0001].mean()
                    mean_c2 = bg_Im[:, 2, :, :][bg_Im[:, 2, :, :] > 0.0001].mean()
                    bg = torch.zeros(size=(3, H, W))
                    bg[0, :, :] = mean_c0
                    bg[1, :, :] = mean_c1
                    bg[2, :, :] = mean_c2
                    return bg
                if aa == 0:
                    aa +=1
                    bg = create_all_background(data, masks)
                    bg = bg.permute(dims=[1, 2, 0])
                    bg *= 255.
                    bg = bg.cpu().numpy()
                    np.save('../data/FetchGenerativeEnv-v1/bg_im.npy', bg)
                # (B, S, 1, H, W)
                masks = masks.cpu().numpy()
                masks = np.tile(masks, (1, 1, 3, 1, 1))
                # to (B, S, H, W, 3)
                masks = np.transpose(masks, axes=(0, 1, 3, 4, 2))
                # (B, S+1, H, W, 3)
                # make masks visible in image
                masks *= 255.
                ims = np.concatenate([np.expand_dims(data_np, axis=1), masks], axis=1)
                new_data_set[batch_idx * batch_size:batch_idx * batch_size + batch_size] = ims
        np.save('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy', new_data_set)
    elif args.task == 'train':
        model = Bbox(args.num_slots, device).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        train(model, optimizer, device, log_interval_epoch=10, log_interval_batch=400,
              batch_size=args.batch_size, num_epochs=args.train_epochs)
    elif args.task == 'test':
        model = load_Model('../data/FetchGenerativeEnv-v1/model_bboxv2_epoch_160',
                           img_size=args.img_size, device=device, num_slots=args.num_slots)
        rec_dataset(model, dataset_path='../data/FetchGenerativeEnv-v1/all_set_with_masks.npy')
    elif args.task == 'try':
        data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_with_masks.npy')
        bg_im = np.load('../data/FetchGenerativeEnv-v1/bg_im.npy')
        bg_im = torch.from_numpy(bg_im).float().to(device)
        bg_im /= 255.
        bg_im = bg_im.permute([2, 0, 1])
        #model = load_Monet(path='../data/FetchGenerativeEnv-v1/all_sb_model', img_size=args.img_size, latent_size=6)
        #model = model.to(device)
        data_size = len(data_set)
        # todo make this configurable
        new_data_set = np.zeros(shape=(data_size, 6 + 1, args.img_size, args.img_size, args.img_size))
        batch_size = 10
        # creates indexes and shuffles them. So it can acces the data
        idx_set = np.arange(data_size)
        idx_set = np.split(idx_set, len(idx_set) / batch_size)
        aa = 0
        with torch.no_grad():
            for batch_idx, idx_select in enumerate(idx_set):
                aa += 1
                data_np = data_set[idx_select]
                data = torch.from_numpy(data_np).float().to(device)
                data /= 255.
                data = data.permute([0, 1, 4, 2, 3])
                B, S, C, H, W = data.shape
                data_ims = data[:, 0:1,:,:,:]
                data_masks = data[:, 1:, :, :, :]
                data = data.reshape(shape=(B*S, C, H, W))
                save_image(data_ims.reshape(shape=(B, C, H, W)), 'try{}.png'.format(aa), pad_value=0.2)
                def mod_ims(ims, masks, bg_image):
                    #leave just one color channel
                    masks =  masks[:, :, 0:1, :, :]#todo this might not be necessary
                    els = ims*masks
                    #save_image(els.reshape(B*(S-1), C, H, W), 'els{}.png'.format(aa), pad_value=0.2)
                    #save_image(masks.reshape(B * (S - 1), 1, H, W), 'masks{}.png'.format(aa), pad_value=0.2)
                    #pos (B*S, 2) in [-1, 1]#todo better take
                    p = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(B, 2)))
                    p = p.float()
                    #dim (B*S, 2) in [0,1]#todo better put
                    d = torch.from_numpy(np.random.uniform(low=0, high=1, size=(B, 2)))
                    d = d.float()
                    z_where = torch.cat([p, d], dim=-1)
                    p_repos = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(B, 2)))
                    p_repos = p_repos.float()
                    # dim (B*S, 2) in [0,1]
                    '''d_repos = torch.rand(size=(B * S, 2))
                    d_repos = d_repos.sigmoid()'''
                    z_where_repos = torch.cat([p_repos, d], dim=-1)

                    index = int(np.random.randint(low=1, high=S - 1))
                    print(index)
                    els_to_mod = els[:, index, :, :, :]
                    masks_to_mod = masks[:, index, :, :, :]

                    els_r1 = spatial_transform(image=els_to_mod, z_where=z_where,out_dims=(B, 3, H, W),
                                              inverse=False)
                    m_r1 = spatial_transform(image=masks_to_mod, z_where=z_where,out_dims=(B, 1, H, W),
                                              inverse=False)
                    els_mod= spatial_transform(image=els_r1, z_where=z_where_repos, out_dims=(B, 3, H, W),
                                               inverse=True)
                    m_mod = spatial_transform(image=m_r1, z_where=z_where_repos, out_dims=(B, 1, H, W),
                                             inverse=True)

                    #masks from other elements; user to avoid that modified element is over them
                    o_masks = masks.clone()
                    o_masks[:, index, :, :, :] = 0
                    #not background mask TODO see if other has
                    o_masks = o_masks[:, 1:, :, :,:]
                    o_masks_sum = torch.sum(o_masks, dim=1)
                    o_masks_sum = torch.clamp(o_masks_sum, max=1.)
                    m_mod = m_mod - o_masks_sum
                    m_mod = torch.clamp(m_mod, min=0.)
                    #save_image(m_mod, 'm_mod{}.png'.format(aa), pad_value=0.2)
                    masks[:, index, :, :, :] = m_mod
                    #els[:, index, :, :, :] = els_mod
                    #save_image(els.reshape(B * (S - 1), C, H, W), 'els_mod{}.png'.format(aa), pad_value=0.2)
                    #save_image(masks.reshape(B * (S - 1), 1, H, W), 'masks_mod{}.png'.format(aa), pad_value=0.2)
                    els_remod = m_mod * els_mod
                    els[:, index, :, :, :] = els_remod
                    #save_image(els.reshape(B * (S - 1), C, H, W), 'els_remod{}.png'.format(aa), pad_value=0.2)
                    masks_sum = torch.sum(masks[:, 1:, :, :, :], dim=1)
                    masks_sum = torch.clamp(masks_sum, max=1)

                    new_im = torch.ones(size=(B, C, H, W)).to(device)
                    new_im *= bg_image
                    new_im -= masks_sum
                    new_im = torch.clamp(new_im, min=0)
                    new_els = torch.sum(els[:, 1:, :, :,:]*masks[:, 1:, :, :, :], dim=1)
                    new_im += new_els
                    save_image(new_im, 'new_im{}.png'.format(aa), pad_value=0.2)
                mod_ims(data_ims, data_masks, bg_im)


                if aa == 5:
                    break
    else:
        print("No valid task")