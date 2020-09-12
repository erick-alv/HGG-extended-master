from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

doing_goal = True

if doing_goal:
    train_file = '../data/FetchPushObstacleFetchEnv-v1/goal_set.npy'
    n_path = '../data/FetchPushObstacleFetchEnv-v1/vae_sb_model_goal'

else:
    train_file = '../data/FetchPushObstacleFetchEnv-v1/obstacle_set.npy'
    n_path = '../data/FetchPushObstacleFetchEnv-v1/vae_sb_model_obstacle'

def spatial_broadcast(z, width, height):
    z_b = np.tile(A=z, reps=(height, width, 1))
    x = np.linspace(-1,1, width)
    y = np.linspace(-1,1,width)
    x_b, y_b = np.meshgrid(x, y)
    x_b = np.expand_dims(x_b, axis=2)
    y_b = np.expand_dims(y_b, axis=2)
    z_sb = np.concatenate([z_b, x_b, y_b], axis=-1)
    return z_sb

def torch_spatial_broadcast(z, width, height, device):
    z_b = torch.stack([z] * width, dim=1)
    z_b = torch.stack([z_b] * height, dim=2)
    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    n = z.size()[0]
    x_b, y_b = torch.meshgrid(x, y)
    x_b = torch.unsqueeze(x_b, 2)
    y_b = torch.unsqueeze(y_b, 2)
    x_b = torch.stack([x_b] * n, dim=0).to(device)
    y_b = torch.stack([y_b] * n, dim=0).to(device)
    z_sb = torch.cat([z_b, x_b, y_b], dim=-1)
    z_sb = z_sb.permute([0, 3, 1, 2])
    return z_sb



class VAE_SB(nn.Module):
    def __init__(self, device, img_size=84, latent_size=2,full_connected_size=320, input_channels=3,
                 kernel_size=3, encoder_stride=2, decoder_stride=1):
        super(VAE_SB, self).__init__()
        self.device = device
        self.img_size = img_size
        self.c1 = nn.Conv2d(in_channels=input_channels, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=1)
        self.c2 = nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=0)
        self.c3 = nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=1)
        self.c4 = nn.Conv2d(in_channels=img_size, kernel_size=kernel_size,
                            stride=encoder_stride, out_channels=img_size, padding=1)
        #the number of cahnnels is img_size
        self.fc1 = nn.Linear(5 * 5 * img_size, full_connected_size)
        # Try to reduce
        self.fc21 = nn.Linear(full_connected_size, latent_size)
        self.fc22 = nn.Linear(full_connected_size, latent_size)

        self.dc1 = nn.Conv2d(in_channels=latent_size+2, kernel_size=kernel_size,
                             stride=decoder_stride, out_channels=img_size, padding=1)
        self.dc2 = nn.Conv2d(in_channels=img_size,
                             kernel_size=kernel_size, stride=decoder_stride, out_channels=img_size, padding=1)
        self.dc3 = nn.Conv2d(in_channels=img_size,
                             kernel_size=kernel_size, stride=decoder_stride, out_channels=3, padding=1)

    def encode(self, x):
        #h1 = F.relu(self.fc1(x))
        #return self.fc21(h1), self.fc22(h1)
        e1 = F.relu(self.c1(x))
        e2 = F.relu(self.c2(e1))
        e3 = F.relu(self.c3(e2))
        e4 = F.relu(self.c4(e3))
        e = e4.reshape(-1, 5 * 5 * self.img_size)
        e = F.relu(self.fc1(e))
        return self.fc21(e), self.fc22(e)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #return (mu + eps * std) / 11
        return (mu + eps * std)
        #return mu

    # maybe z * 11
    def decode(self, z):
        tz = torch_spatial_broadcast(z, self.img_size, self.img_size, self.device)
        d1 =  F.relu(self.dc1(tz))
        d2 = F.relu(self.dc2(d1))
        d3 = F.relu(self.dc3(d2))
        return torch.sigmoid(d3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    '''def format(self, rgb_array):
        data = torch.from_numpy(rgb_array).float().to(device='cuda')
        data /= 255
        data = data.permute([2, 0, 1])
        data = data.reshape([-1, 3, img_size, img_size])
        return data.reshape(-1, img_size * img_size * 3)'''

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # Try to adjust
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + 1.8*KLD

# torch.Size([128, 1, img_size, img_size])
def train(epoch, model, optimizer, device, log_interval, batch_size):
    model.train()
    train_loss = 0
    data_set = np.load(train_file)
    data_size = len(data_set)
    data_set = np.split(data_set, data_size / batch_size)

    for batch_idx, data in enumerate(data_set):
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #save_image(data.cpu().view(-1, 3, img_size, img_size),
            #           'results/original.png')
            #save_image(recon_batch.cpu().view(-1, 3, img_size, img_size),
            #           'results/recon.png')

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), data_size,
                100. * (batch_idx+1) / len(data_set),
                loss.item() / len(data)))
            print('Loss: ', loss.item() / len(data))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / data_size))


def train_Vae(batch_size=128, epochs=100, no_cuda=False, seed=1, log_interval=100, load=False,
              img_size=84, latent_size=2):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    if load:
        model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        checkpoint = torch.load(n_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
    else:
        model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        start_epoch = 1

    for epoch in range(start_epoch, epochs + start_epoch):
        train(epoch, model, optimizer, device, log_interval, batch_size)
        # test(epoch, model, test_loader, batch_size, device)
        # with torch.no_grad():
        #    sample = torch.randn(64, 5).to(device)
        #    sample = model.decode(sample).cpu()
        #    save_image(sample.view(64, 3, img_size, img_size),
        #               'results/sample.png')
        if not (epoch % 5) or epoch == 1:
            test_on_data_set(model, device,'epoch_{}'.format(epoch), latent_size=latent_size)
            print('Saving Progress!')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, n_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, n_path+'_epoch_'+str(epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, n_path)

def test_Vae(no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device).to(device)
    checkpoint = torch.load(n_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_on_data_set(model, device, 'test')


def test_on_data_set(model, device, filename_suffix, latent_size=2):
    data_set = np.load(train_file)
    data_size = len(data_set)
    idx = np.random.randint(0, data_size, size=10)
    data = data_set[idx]
    with torch.no_grad():
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        recon, mu, logvar = model(data)

        mu = mu.view(10, latent_size)
        logvar = logvar.view(10, latent_size)

        comparison = torch.cat([data, recon])
        save_image(comparison.cpu(), 'results/reconstruction_{}.png'.format(filename_suffix),
                   nrow=10)

def load_Vae(path, no_cuda=False, seed=1, img_size=84, latent_size=2):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = VAE_SB(device, img_size=img_size, latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model

#adapted from https://github.com/Natsu6767/Variational-Autoencoder/blob/master/main.py
import matplotlib.pyplot as plt
from scipy.stats import norm


def show_2d_manifold(img_size, no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device).to(device)
    checkpoint = torch.load(n_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    n = 20  # figure with nxn images
    figure = np.zeros((img_size * n, img_size * n, 3))
    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))#with probabilities to values of the distribution
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))
    #grid_x = np.linspace(-2, 2, n, endpoint=True)#with values of the distribution
    #grid_y = np.linspace(-2, 2, n, endpoint=True)
    # Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            z_sample = np.array([xi, yj])
            z_sample = torch.from_numpy(z_sample).to(device).float()
            z_sample = torch.unsqueeze(z_sample, 0)
            im_decoded = model.decode(z_sample)
            im_decoded = im_decoded.view(3, img_size, img_size)
            im_decoded = im_decoded.permute([1, 2, 0])
            #im2 = im_decoded.detach().cpu()
            #im2 *= 255
            #im2 = im2.type(torch.uint8).numpy()
            im_decoded = im_decoded.detach().cpu().numpy()
            #if i == 0 and j == 0:
            #    Image.fromarray(im2).show()
            #if i == len(grid_x)-1 and j == len(grid_y)-1:
            #   Image.fromarray(im2).show()
            figure[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size, :] = im_decoded
    plt.figure(figsize=(15, 15))
    plt.imshow(figure)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n*img_size, img_size))
    ax.set_yticks(np.arange(0, n*img_size, img_size))
    plt.grid(color='black', linewidth='1.2')
    plt.show()


def show_2d_manifold_with_fixed_axis(img_size, no_cuda=False, seed=1, fixed_axis=0, fixed_prob_val=0.5):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device).to(device)
    checkpoint = torch.load(n_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    assert fixed_axis in [0,1,2]
    n = 20  # figure with nxn images
    figure = np.zeros((img_size * n, img_size * n, 3))
    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))#with probabilities to values of the distribution
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n, endpoint=True))
    #grid_x = np.linspace(-2, 2, n, endpoint=True)#with values of the distribution
    #grid_y = np.linspace(-2, 2, n, endpoint=True)
    # Decode for each square in the grid.
    fixed_val = norm.ppf(fixed_prob_val)
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            if fixed_axis == 0:
                z_sample = np.array([fixed_val, xi, yj])
            elif fixed_axis ==1:
                z_sample = np.array([xi, fixed_val, yj])
            else:
                z_sample = np.array([xi, yj, fixed_val])
            z_sample = torch.from_numpy(z_sample).to(device).float()
            im_decoded = model.decode(z_sample)
            im_decoded = im_decoded.view(3, img_size, img_size)
            im_decoded = im_decoded.permute([1, 2, 0])
            #im2 = im_decoded.detach().cpu()
            #im2 *= 255
            #im2 = im2.type(torch.uint8).numpy()
            im_decoded = im_decoded.detach().cpu().numpy()
            #if i == 0 and j == 0:
            #    Image.fromarray(im2).show()
            #if i == len(grid_x)-1 and j == len(grid_y)-1:
            #   Image.fromarray(im2).show()
            figure[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size, :] = im_decoded
    plt.figure(figsize=(15, 15))
    plt.imshow(figure)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n*img_size, img_size))
    ax.set_yticks(np.arange(0, n*img_size, img_size))
    plt.grid(color='black', linewidth='1.2')
    plt.show()


def show_1d_manifold(img_size, no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    model = VAE_SB(device).to(device)
    checkpoint = torch.load(n_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    n = 20  # figure with nxn images
    figure = np.zeros((img_size*n,img_size, 3))

    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.005, 0.995, n, endpoint=True))  # with probabilities to values of the distribution
    # grid_x = np.linspace(-2, 2, n, endpoint=True)#with values of the distribution

    # Decode for each square in the grid.
    for i, xi in enumerate(grid_x):
        z_sample = np.array([xi])
        z_sample = torch.from_numpy(z_sample).to(device).float()
        im_decoded = model.decode(z_sample)
        im_decoded = im_decoded.view(3, img_size, img_size)
        im_decoded = im_decoded.permute([1, 2, 0])
        # im2 = im_decoded.detach().cpu()
        # im2 *= 255
        # im2 = im2.type(torch.uint8).numpy()
        im_decoded = im_decoded.detach().cpu().numpy()
        # if i == 0 and j == 0:
        #    Image.fromarray(im2).show()
        # if i == len(grid_x)-1 and j == len(grid_y)-1:
        #   Image.fromarray(im2).show()
        figure[i * img_size: (i + 1) * img_size, :, :] = im_decoded

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, img_size, img_size))
    ax.set_yticks(np.arange(0, n * img_size, img_size))
    plt.grid(color='black', linewidth='1.2')

    plt.show()

if __name__ == '__main__':
    print('Train VAE...')
    train_Vae(batch_size=32, epochs=15, load=False, latent_size=3)
    # test_VAE_SB(device)
    # show_1d_manifold()
    #show_2d_manifold(84)
    print('Successfully trained VAE')
