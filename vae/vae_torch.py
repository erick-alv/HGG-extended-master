import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from skimage import io
from custom_start import get_args_and_initialize
import numpy as np

class CustomImagesDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, train=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        images_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        images_names.sort(key=lambda x: int(x[6:-4]))
        t = int(0.9*len(images_names))
        if train:
            self.images_names = images_names[:t]
        else:
            self.images_names = images_names[t:]
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.directory,
                                self.images_names[idx])
        image = io.imread(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)
        return sample

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

class VAE_Trainer:
    def __init__(self, model, train_loader, test_loader, args, lr=1e-3):
        assert isinstance(model, VAE)
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
        return BCE + KLD

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
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))
        if epoch % self.args.checkpoint_interval == 0:
            save_train_checkpoint(args, 'trained_weights', vae=self.model, trainer=self, epoch=epoch)

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
                    comparison = torch.cat([data[:n], recon_batch.view(self.args.batch_size, self.model.channels_dim,
                                                                       self.model.x_y_dim, self.model.x_y_dim)[:n]])
                    save_image(comparison.cpu(),
                               self.args.dirpath + 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

def setup_vae(args, recover_filename=None):
    train_loader = torch.utils.data.DataLoader(
        CustomImagesDataset(directory=args.dirpath + 'images/next_obs/', train=True,
                            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **args.if_cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(
        CustomImagesDataset(directory=args.dirpath + 'images/next_obs/', train=True,
                            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **args.if_cuda_kwargs)
    model = VAE(84, 3, 20).to(args.device)
    trainer = VAE_Trainer(model, train_loader, test_loader, args)
    if recover_filename is not None:
        load_vae(args, recover_filename, model, trainer)
    return model, trainer, train_loader, test_loader

def save_vae(args, filename, vae, trainer=None, epoch=None):
    assert isinstance(vae, VAE)
    save_dict = {'model_state_dict': vae.state_dict()}
    if trainer is not None:
        assert isinstance(trainer, VAE_Trainer)
        save_dict['optimizer_state_dict'] = trainer.optimizer.state_dict()
    path = args.dirpath + 'weights_dir/'+filename
    if epoch is not None:
        path = path + '_' + str(epoch)
    torch.save(save_dict, path)

def save_train_checkpoint(args, filename, vae, trainer, epoch):
    save_vae(args, filename, vae, trainer, epoch)

def load_vae(args, filename, vae, trainer=None):
    assert isinstance(vae, VAE)
    path = args.dirpath + 'weights_dir/' + filename
    save_dict = torch.load(path)
    vae.load_state_dict(save_dict['model_state_dict'])
    if trainer is not None:
        assert isinstance(trainer, VAE_Trainer)
        trainer.optimizer.load_state_dict(save_dict['optimizer_state_dict'])

def load_train_checkpoint(args, filename, vae, trainer, epoch):
    if not filename.endswith(str(epoch)):
        filename = filename + '_' + str(epoch)
    load_vae(args, filename, vae, trainer)

def start_training(args):
    model, trainer, _, _ = setup_vae(args)
    training_loop(args, 1, model, trainer)

def resume_training(args, epoch):
    recover_filename = 'trained_weights_'+str(epoch)
    model, trainer, _, _ = setup_vae(args, recover_filename=recover_filename)
    training_loop(args, epoch, model, trainer)

def training_loop(args, start_epoch, model, trainer):
    for epoch in range(start_epoch, args.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)
        if epoch % 100 == 0 or epoch == args.epochs:
            with torch.no_grad():
                sample = torch.randn(64, model.latent_dim).to(args.device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, model.channels_dim, model.x_y_dim, model.x_y_dim),
                           args.dirpath + 'results/sample_' + str(epoch) + '.png')
    save_vae(args, 'trained_last', vae=model, trainer=trainer)

if __name__ == "__main__":
    args = get_args_and_initialize()
    #start_training(args)
    resume_training(args, 600)

    '''args = get_args_and_initialize()
    model, trainer = setup_vae(args, recover_filename='trained_last')
    with torch.no_grad():
        sample = torch.randn(64, model.latent_dim).to(args.device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, model.channels_dim, model.x_y_dim, model.x_y_dim),
                   args.dirpath + 'results/sample_after_recover' + '.png')'''