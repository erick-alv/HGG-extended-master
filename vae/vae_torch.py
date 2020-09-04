import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from skimage import io as skio
from custom_start import get_args_and_initialize
import numpy as np
import pandas as pd
import io
from utils.image_util import make_text_im

LAST_WEIGHTS_NAME = 'no_obstacles_ae_weights_last'
CHECKPOINT_NAME = 'no_obstacles_ae_weights_'
class VAE(nn.Module):
    def __init__(self, x_y_dim, channels_dim, latent_dim):
        super(VAE, self).__init__()

        self.x_y_dim = x_y_dim
        self.channels_dim = channels_dim
        self.flattened_dim = x_y_dim * x_y_dim * channels_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 1600
        #network construction
        self.fc1 = nn.Linear(self.flattened_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.flattened_dim)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, self.flattened_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def save_vae(self, args, filename, trainer=None, epoch=None):
        save_dict = {'model_state_dict': self.state_dict()}
        if trainer is not None:
            save_dict['optimizer_state_dict'] = trainer.optimizer.state_dict()
        path = args.dirpath + 'weights_dir/' + filename
        if epoch is not None:
            path = path + '_' + str(epoch)
        torch.save(save_dict, path)

    def save_train_checkpoint(self, args, filename, trainer, epoch):
        self.save_vae(args, filename, trainer, epoch)

    def load_vae(self, args, filename, trainer=None):
        path = args.dirpath + 'weights_dir/' + filename
        save_dict = torch.load(path)
        self.load_state_dict(save_dict['model_state_dict'])
        if trainer is not None:
            trainer.optimizer.load_state_dict(save_dict['optimizer_state_dict'])

    def load_train_checkpoint(self, args, filename, trainer, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load_vae(args, filename, trainer)

class VAE_Trainer:
    def __init__(self, model, train_loader, test_loader, args, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.model.flattened_dim), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + self.args.vae_kl_beta * KLD

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.args.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += float(loss.item())
            self.optimizer.step()
            if batch_idx % self.args.vae_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))
        print(batch_idx)

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))
        if epoch % self.args.vae_checkpoint_interval == 0:
            self.model.save_train_checkpoint(args, CHECKPOINT_NAME, trainer=self, epoch=epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, data, in enumerate(self.test_loader):
                data = data.to(self.args.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += float(self.loss_function(recon_batch, data, mu, logvar).item())
                if i == 0 and epoch % 100 == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(self.args.vae_batch_size,
                                                                       self.model.channels_dim,
                                                                       self.model.x_y_dim, self.model.x_y_dim)[:n]])
                    save_image(comparison.cpu(), self.args.dirpath + self.args.vae_results_folder
                               +'reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


class CustomImagesDataset(torch.utils.data.Dataset):
    def __init__(self, directory, file_name_prefix, file_name_suffix, transform=None, train=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        images_names = [f for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f)) and
                        f.endswith(file_name_suffix) and f.startswith(file_name_prefix)]
        images_names.sort(key=lambda x: int(x.replace(file_name_prefix, '').replace(file_name_suffix, '')))
        q = int(0.9*len(images_names))
        r = len(images_names) - q
        t1 = q // 2
        t2 = t1 + r
        if train:
            self.images_names = images_names[:t1] + images_names[t2:]
        else:
            self.images_names = images_names[t1:t2]
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.directory,
                                self.images_names[idx])
        image = skio.imread(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)
        return sample


class ImagePointsDataset(torch.utils.data.Dataset):
    def __init__(self, directory, csv_file_name, transform=None, train=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(directory + csv_file_name)
        q = int(0.9*len(self.df))
        r = len(self.df) - q
        t1 = q // 2
        t2 = t1 + r
        if train:
            df1 = self.df.iloc[:t1, :]
            df2 = self.df.iloc[t2:, :]
            self.df =  df1.append(df2, ignore_index=True)
        else:
            self.df = self.df.iloc[t1:t2, :]
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file_name = self.df.iloc[idx].loc['image']
        if not img_file_name.endswith('.png'):
            img_file_name = img_file_name + '.png'
        img_name = os.path.join(self.directory, img_file_name)
        image = skio.imread(img_name)
        gripper_pos = self.df.iloc[idx].loc['gripper_pos']
        object_pos = self.df.iloc[idx].loc['object_pos']
        sample = {'image': image, 'gripper_pos': gripper_pos, 'object_pos': object_pos}

        if self.transform:
            sample = self.transform(sample)
        return sample


class StrNumpyToTensorTransform(object):
    """reads str to numpy.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, sample_keys):
        self.sample_keys = sample_keys

    def __call__(self, sample):
        new_sample = {}
        for n in self.sample_keys:
            np_val = np.loadtxt(io.StringIO(sample[n]))
            tensor_val = torch.from_numpy(np_val)
            new_sample.update({n: tensor_val})
        for k in sample.keys():
            if k in self.sample_keys:
                continue
            else:
                new_sample.update({k: sample[k]})

        return new_sample


class ImgToTensor(object):
    def __init__(self, sample_keys):
        self.sample_keys = sample_keys

    def __call__(self, sample):
        for n in self.sample_keys:
            val = transforms.ToTensor()(sample[n])
            sample.update({n: val})
        return sample


img_point_transform = transforms.Compose([StrNumpyToTensorTransform(['gripper_pos', 'object_pos']),
                                          ImgToTensor(['image'])])


class AeEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=1600):
        super(AeEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1)

    def forward(self, x):
        return self.encode(x)

class AeEncoderImage(AeEncoder):
    def __init__(self, x_y_dim, channels_dim, latent_dim, hidden_dim=1600):
        self.x_y_dim = x_y_dim
        self.channels_dim = channels_dim
        self.flattened_dim = x_y_dim * x_y_dim * channels_dim
        super(AeEncoderImage, self).__init__(self.flattened_dim, latent_dim, hidden_dim)

    def encode(self, x):
        x = torch.reshape(x, (-1, self.flattened_dim))
        return super(AeEncoderImage, self).encode(x)

class VaeEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=1600):
        super(VaeEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return mu, logvar


class VaeEncoderImage(VaeEncoder):
    def __init__(self, x_y_dim, channels_dim, latent_dim, hidden_dim=1600):
        self.x_y_dim = x_y_dim
        self.channels_dim = channels_dim
        self.flattened_dim = x_y_dim * x_y_dim * channels_dim
        super(VaeEncoderImage, self).__init__(self.flattened_dim, latent_dim, hidden_dim)

    def encode(self, x):
        x = x.view(-1, self.flattened_dim)
        return super(VaeEncoderImage, self).encode(x)

#TODO write max and min for points this is giving me always values between 0 and 1 as output
class VaeDecoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dim=1600):
        super(VaeDecoder, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.output_dim)

    def decode(self, z):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, z):
        return self.decode(z)


class VaeDecoderImage(VaeDecoder):
    def __init__(self, x_y_dim, channels_dim, latent_dim, hidden_dim=1600):
        self.x_y_dim = x_y_dim
        self.channels_dim = channels_dim
        self.flattened_dim = x_y_dim * x_y_dim * channels_dim
        super(VaeDecoderImage, self).__init__(self.flattened_dim, latent_dim, hidden_dim)


class VaeDecoderRange(VaeDecoder):
    def __init__(self, output_dim, latent_dim, min_range, max_range, device, hidden_dim = 1600):
        if not isinstance(max_range, torch.Tensor):
            self.min_range = torch.Tensor(min_range)
        else:
            self.min_range = min_range
        if not isinstance(min_range, torch.Tensor):
            self.max_range = torch.Tensor(max_range)
        else:
            self.max_range = max_range
        #values for range transformation
        self.max_range = self.max_range.unsqueeze(0).to(device)
        self.min_range = self.min_range.unsqueeze(0).to(device)
        self.range_dist = self.max_range - self.min_range
        super(VaeDecoderRange, self).__init__(output_dim, latent_dim, hidden_dim)

    def decode(self, z):
        r = super(VaeDecoderRange, self).decode(z)
        #transforms [0,1] -> [self.min_range, self.max_range]
        r = r * self.range_dist + self.min_range
        return r



class MultiAE(nn.Module):
    def __init__(self, encoder, decoders):
        super(MultiAE, self).__init__()
        self.m = nn.ModuleList()
        self.m.append(encoder)
        for dec in decoders:
            self.m.append(dec)
        self.encoder = self.m[0]
        self.decoders = self.m[1:]
        self.reconstruction_dims = [dec.output_dim for dec in self.decoders]
        # like reconstruction dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        zs = [z.clone() for _ in self.decoders]
        return [dec(zs[i]) for i, dec in enumerate(self.decoders)]

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def save_vae(self, args, filename, trainer=None, epoch=None):
        save_dict = {'model_state_dict': self.state_dict()}
        if trainer is not None:
            save_dict['optimizer_state_dict'] = trainer.optimizer.state_dict()
        path = args.dirpath + 'weights_dir/' + filename
        if epoch is not None:
            path = path + '_' + str(epoch)
        torch.save(save_dict, path)

    def save_train_checkpoint(self, args, filename, trainer, epoch):
        self.save_vae(args, filename, trainer, epoch)

    def load_vae(self, args, filename, trainer=None):
        path = args.dirpath + 'weights_dir/' + filename
        save_dict = torch.load(path)
        self.load_state_dict(save_dict['model_state_dict'])
        if trainer is not None:
            trainer.optimizer.load_state_dict(save_dict['optimizer_state_dict'])

    def load_train_checkpoint(self, args, filename, trainer, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load_vae(args, filename, trainer)


class MultiVAE(nn.Module):#todo must it really be a subclass of Module??
    def __init__(self, encoder, decoders):
        super(MultiVAE, self).__init__(encoder, decoders)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Multi_Trainer:
    def __init__(self, model, args, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.args = args

    # Reconstruction + KL divergence losses summed over all elements and batch
    def calc_rec_loss_bce(self, recon_x, x, flattened_dim):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, flattened_dim), reduction='sum')
        return BCE

    def calc_rec_loss_mse(self, recon_x, x, flattened_dim):
        MSE = nn.MSELoss()(recon_x,x.view(-1, flattened_dim))
        return MSE

    def calc_kl_loss(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.args.vae_kl_beta * KLD

    def apply_loss(self, losses):
        loss = None
        for loss_val in losses:
            if loss is None:
                loss = loss_val + 0
            else:
                loss = loss + loss_val

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def setup_vae_just_images(args, recover_filename=None):
    train_loader = torch.utils.data.DataLoader(
        CustomImagesDataset(directory=args.dirpath + args.vae_training_images_folder,
                            file_name_prefix=args.training_images_prefix, file_name_suffix='.png', train=True,
                            transform=transforms.ToTensor()), batch_size=args.vae_batch_size, shuffle=True,
        **args.if_cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(
        CustomImagesDataset(directory=args.dirpath + args.vae_training_images_folder,
                            file_name_prefix=args.training_images_prefix, file_name_suffix='.png', train=False,
                            transform=transforms.ToTensor()), batch_size=args.vae_batch_size, shuffle=True,
        **args.if_cuda_kwargs)
    model = VAE(args.img_dim, args.img_channels, args.latent_dim).to(args.device)

    trainer = VAE_Trainer(model, train_loader, test_loader, args)
    if recover_filename is not None:
        model.load_vae(args, recover_filename, trainer)
    return model, trainer, train_loader, test_loader


def setup_vae_images_and_points(args, recover_filename=None):
    trd = ImagePointsDataset(directory=args.dirpath + args.vae_training_images_folder,
                             csv_file_name=args.training_csv_file, transform=img_point_transform, train=True)
    train_loader = torch.utils.data.DataLoader(trd, batch_size=args.vae_batch_size, shuffle=True,
                                               **args.if_cuda_kwargs)
    ted = ImagePointsDataset(directory=args.dirpath + args.vae_training_images_folder,
                             csv_file_name=args.training_csv_file, transform=img_point_transform, train=False)
    test_loader = torch.utils.data.DataLoader(ted, batch_size=args.vae_batch_size, shuffle=True,
                                              **args.if_cuda_kwargs)
    encoder = VaeEncoderImage(args.img_dim, args.img_channels, args.latent_dim).to(args.device)
    dec1 = VaeDecoderImage(args.img_dim, args.img_channels, args.latent_dim).to(args.device)
    dec2 = VaeDecoderRange(6, args.latent_dim, min_range=[-2.]*6, max_range=[2]*6, device=args.device).to(args.device)
    model = MultiVAE(encoder, [dec1, dec2]).to(args.device)
    trainer = Multi_Trainer(model, args)
    if recover_filename is not None:
        model.load_vae(args, recover_filename, trainer)
    return model, trainer, train_loader, test_loader

def setup_vae_images_and_points_ae(args, recover_filename=None):
    trd = ImagePointsDataset(directory=args.dirpath + args.vae_training_images_folder,
                             csv_file_name=args.training_csv_file, transform=img_point_transform, train=True)
    train_loader = torch.utils.data.DataLoader(trd, batch_size=args.vae_batch_size, shuffle=True,
                                               **args.if_cuda_kwargs)
    ted = ImagePointsDataset(directory=args.dirpath + args.vae_training_images_folder,
                             csv_file_name=args.training_csv_file, transform=img_point_transform, train=False)
    test_loader = torch.utils.data.DataLoader(ted, batch_size=args.vae_batch_size, shuffle=True,
                                              **args.if_cuda_kwargs)
    model, trainer = setup_distance_ae_and_trainer(args, recover_filename)
    return model, trainer, train_loader, test_loader

def setup_distance_ae_and_trainer(args,  recover_filename=None, compat_extra=False):#todo remove
    encoder = AeEncoderImage(args.img_dim, args.img_channels, args.latent_dim).to(args.device)
    #the vae decoder works as well for the ae
    dec1 = VaeDecoderImage(args.img_dim, args.img_channels, args.latent_dim).to(args.device)
    dec2 = VaeDecoderRange(6, args.latent_dim, min_range=[-2.] * 6, max_range=[2] * 6,
                           device=args.device, hidden_dim=800).to(args.device)
    if compat_extra:
        dec3 = VaeDecoderRange(1, args.latent_dim, min_range=[0],max_range=[200],device=args.device,hidden_dim=400)
        model = MultiAE(encoder, [dec1, dec2, dec3]).to(args.device)
    else:
        model = MultiAE(encoder, [dec1, dec2]).to(args.device)
    trainer = Multi_Trainer(model, args)
    if recover_filename is not None:
        model.load_vae(args, recover_filename, trainer)
    return model, trainer


def training_loop_images_points(args, start_epoch, model, trainer, train_loader, test_loader):
    for epoch in range(start_epoch, args.vae_epochs):
        #training
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            for k in data.keys():
                data.update({k: data[k].to(args.device)})
            recon_batch, mu, logvar = model(data['image'])
            loss_image = trainer.calc_rec_loss_bce(recon_batch[0], data['image'], model.decoders[0].output_dim)
            vec = torch.cat([data['gripper_pos'], data['object_pos']], dim=1).type_as(recon_batch[1])
            loss_point = 1000. * trainer.calc_rec_loss_mse(recon_batch[1], vec, model.decoders[1].output_dim)
            #todo see better way to scale loss
            kl_loss = trainer.calc_kl_loss(mu, logvar)
            trainer.apply_loss([loss_image, loss_point, kl_loss])
            loss = loss_image + loss_point + kl_loss
            train_loss += float(loss.item())
            if batch_idx % args.vae_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item()/ len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        if epoch %args.vae_checkpoint_interval == 0:
            model.save_train_checkpoint(args, CHECKPOINT_NAME, trainer=trainer, epoch=epoch)

        if epoch % args.vae_tr_test_interval == 0 or epoch == args.vae_epochs-1:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for i, data, in enumerate(test_loader):
                    for k in data.keys():
                        data.update({k: data[k].to(args.device)})
                    recon_batch, mu, logvar = model(data['image'])
                    loss_image = trainer.calc_rec_loss_bce(recon_batch[0], data['image'], model.decoders[0].output_dim)
                    vec = torch.cat([data['gripper_pos'], data['object_pos']], dim=1).type_as(recon_batch[1])
                    loss_point = trainer.calc_rec_loss_mse(recon_batch[1], vec, model.decoders[1].output_dim)
                    kl_loss = trainer.calc_kl_loss(mu, logvar)
                    loss = loss_image + loss_point + kl_loss
                    test_loss += float(loss.item())
                    if i == 0:
                        n = min(data['image'].size(0), 8)
                        str_format = 'Real:\ng[{:.4f},\n{:.4f},\n{:.4f}]\no[{:.4f},\n{:.4f},\n{:.4f}]\nPredicted:' \
                                     '\ng[{:.4f},\n{:.4f},\n{:.4f}]\no[{:.4f},\n{:.4f},\n{:.4f}]'
                        predicted_points = recon_batch[1].view(args.vae_batch_size,model.decoders[1].output_dim)[:n]
                        real_grippers = data['gripper_pos'][:n]
                        real_objects = data['object_pos'][:n]
                        text_ims = [make_text_im(model.encoder.x_y_dim, 15,
                                                  str_format.format(real_grippers[i][0], real_grippers[i][1],
                                                                    real_grippers[i][2], real_objects[i][0],
                                                                    real_objects[i][1], real_objects[i][2],
                                                                    predicted_points[i][0], predicted_points[i][1],
                                                                    predicted_points[i][2], predicted_points[i][3],
                                                                    predicted_points[i][4], predicted_points[i][5]))
                                    for i in range(n)]
                        text_ims = np.array(text_ims)
                        text_ims = np.moveaxis(text_ims, [1,2,3], [2, 3, 1])
                        text_ims = torch.from_numpy(text_ims).to(args.device).type_as(data['image'][0])
                        ims = data['image'][:n]
                        rec_ims = recon_batch[0].view(args.vae_batch_size,model.encoder.channels_dim,
                                                      model.encoder.x_y_dim, model.encoder.x_y_dim)[:n]
                        comparison = torch.cat([ims, rec_ims, text_ims], dim=2)
                        save_image(comparison.cpu(), args.dirpath + args.vae_results_folder
                                   + 'reconstruction_' + str(epoch) + '.png')

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))
            #with torch.no_grad():
            #    sample = torch.randn(64, model.decoder.latent_dim).to(args.device)
            #    sample = model.decode(sample).cpu()
            #    save_image(sample.view(64, model.decoder.channels_dim, model.decoder.x_y_dim, model.decoder.x_y_dim),
            #               args.dirpath + args.vae_results_folder +'sample_' + str(epoch) + '.png')
    model.save_vae(args, LAST_WEIGHTS_NAME, trainer=trainer)

def training_loop_images_points_ae(args, start_epoch, model, trainer, train_loader, test_loader):
    for epoch in range(start_epoch, args.vae_epochs):
        #training
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            for k in data.keys():
                data.update({k: data[k].to(args.device)})
            recon_batch = model(data['image'])
            loss_image = trainer.calc_rec_loss_bce(recon_batch[0], data['image'], model.decoders[0].output_dim)
            vec = torch.cat([data['gripper_pos'], data['object_pos']], dim=1).type_as(recon_batch[1])
            loss_point = 1000. * trainer.calc_rec_loss_mse(recon_batch[1], vec, model.decoders[1].output_dim)
            #todo see better way to scale loss
            trainer.apply_loss([loss_image, loss_point])
            loss = loss_image + loss_point
            train_loss += float(loss.item())
            if batch_idx % args.vae_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item()/ len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        if epoch %args.vae_checkpoint_interval == 0:
            model.save_train_checkpoint(args, CHECKPOINT_NAME, trainer=trainer, epoch=epoch)

        if epoch % args.vae_tr_test_interval == 0 or epoch == args.vae_epochs-1:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for i, data, in enumerate(test_loader):
                    for k in data.keys():
                        data.update({k: data[k].to(args.device)})
                    recon_batch = model(data['image'])
                    loss_image = trainer.calc_rec_loss_bce(recon_batch[0], data['image'], model.decoders[0].output_dim)
                    vec = torch.cat([data['gripper_pos'], data['object_pos']], dim=1).type_as(recon_batch[1])
                    loss_point = trainer.calc_rec_loss_mse(recon_batch[1], vec, model.decoders[1].output_dim)
                    loss = loss_image + loss_point
                    test_loss += float(loss.item())
                    if i == 0:
                        n = min(data['image'].size(0), 8)
                        str_format = 'Real:\ng[{:.4f},\n{:.4f},\n{:.4f}]\no[{:.4f},\n{:.4f},\n{:.4f}]\nPredicted:' \
                                     '\ng[{:.4f},\n{:.4f},\n{:.4f}]\no[{:.4f},\n{:.4f},\n{:.4f}]'
                        predicted_points = recon_batch[1].view(args.vae_batch_size,model.decoders[1].output_dim)[:n]
                        real_grippers = data['gripper_pos'][:n]
                        real_objects = data['object_pos'][:n]
                        text_ims = [make_text_im(model.encoder.x_y_dim, 15,
                                                  str_format.format(real_grippers[i][0], real_grippers[i][1],
                                                                    real_grippers[i][2], real_objects[i][0],
                                                                    real_objects[i][1], real_objects[i][2],
                                                                    predicted_points[i][0], predicted_points[i][1],
                                                                    predicted_points[i][2], predicted_points[i][3],
                                                                    predicted_points[i][4], predicted_points[i][5]))
                                    for i in range(n)]
                        text_ims = np.array(text_ims)
                        text_ims = np.moveaxis(text_ims, [1,2,3], [2, 3, 1])
                        text_ims = torch.from_numpy(text_ims).to(args.device).type_as(data['image'][0])
                        ims = data['image'][:n]
                        rec_ims = recon_batch[0].view(args.vae_batch_size,model.encoder.channels_dim,
                                                      model.encoder.x_y_dim, model.encoder.x_y_dim)[:n]
                        comparison = torch.cat([ims, rec_ims, text_ims], dim=2)
                        save_image(comparison.cpu(), args.dirpath + args.vae_results_folder
                                   + 'reconstruction_' + str(epoch) + '.png')

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))
    model.save_vae(args, LAST_WEIGHTS_NAME, trainer=trainer)


def training_loop_just_images(args, start_epoch, model, trainer):
    for epoch in range(start_epoch, args.vae_epochs):
        trainer.train(epoch)
        if epoch % args.vae_tr_test_interval == 0 or epoch == args.vae_epochs:
            trainer.test(epoch)
        if epoch % 100 == 0 or epoch == args.vae_epochs-1:
            with torch.no_grad():
                sample = torch.randn(64, model.latent_dim).to(args.device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, model.channels_dim, model.x_y_dim, model.x_y_dim),
                           args.dirpath + args.vae_results_folder +'sample_' + str(epoch) + '.png')
    model.save_vae(args, LAST_WEIGHTS_NAME, trainer=trainer)


if __name__ == "__main__":
    args = get_args_and_initialize()
    model, trainer, train_loader, test_loader = setup_vae_images_and_points_ae(args)
    training_loop_images_points_ae(args, 0, model, trainer, train_loader, test_loader)
    # start_training(args)
    #model, trainer, _, _ = setup_vae_just_images(args)

    #training_loop_just_images(args, 0, model, trainer)
    #resume_training
    #epoch = 600
    #model, trainer, _, _ = setup_vae_just_images(args, recover_filename=recover_filename)
    #training_loop_just_images(args, epoch, model, trainer)