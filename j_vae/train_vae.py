from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


img_size = 84
encoding_of = 'size'

if encoding_of == 'goal':
    latent_size = 2
    n_path = '../data/FetchPushObstacle/vae_model_goal_test'
    train_file = '../data/FetchPushObstacle/goal_set.npy'
elif encoding_of == 'obstacle':
    latent_size = 2
    n_path = '../data/FetchPushObstacle/vae_model_obstacle'
    train_file = '../data/FetchPushObstacle/obstacle_set.npy'
elif encoding_of == 'size':
    latent_size = 1
    n_path = '../data/FetchPushObstacle/vae_model_obstacle_sizes'
    train_file = '../data/FetchPushObstacle/obstacle_sizes_set.npy'
elif encoding_of == 'size_and_position':
    latent_size = 3
    n_path = '../data/FetchPushObstacle/vae_model_sizes_pos'
    train_file = '../data/FetchPushObstacle/obstacle_sizes_points_set.npy'


class VAE(nn.Module):
    def __init__(self, img_size, latent_size, fully_connected_size=256):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(img_size * img_size * 3, fully_connected_size)
        # Try to reduce
        self.fc21 = nn.Linear(fully_connected_size, latent_size)
        self.fc22 = nn.Linear(fully_connected_size, latent_size)
        self.fc3 = nn.Linear(latent_size, fully_connected_size)
        self.fc4 = nn.Linear(fully_connected_size, img_size * img_size * 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #return (mu + eps * std) / 11
        return (mu + eps * std)
        #return mu

    # maybe z * 11
    def decode(self, z):
        #h3 = F.relu(self.fc3(z * 11))
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.reshape(-1, img_size * img_size * 3))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def format(self, rgb_array):
        data = torch.from_numpy(rgb_array).float().to(device='cuda')
        data /= 255
        data = data.permute([2, 0, 1])
        data = data.reshape([-1, 3, img_size, img_size])
        return data.reshape(-1, img_size * img_size * 3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.reshape(-1, img_size * img_size * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # Try to adjust
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

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
        data = data.reshape([-1, 3, img_size, img_size])
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


def train_Vae(batch_size=128, epochs=100, no_cuda=False, seed=1, log_interval=100, load=False):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    if load:
        model = VAE(img_size=img_size, latent_size=latent_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        checkpoint = torch.load(n_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1
    else:
        model = VAE(img_size=img_size, latent_size=latent_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        start_epoch = 1

    for epoch in range(start_epoch, epochs + start_epoch):
        train(epoch, model, optimizer, device, log_interval, batch_size)
        # test(epoch, model, test_loader, batch_size, device)
        # with torch.no_grad():
        #    sample = torch.randn(64, 5).to(device)
        #    sample = model.decode(sample).cpu()
        #    save_image(sample.view(64, 3, img_size, img_size),
        #               'results/sample.png')
        if not (epoch % 25) or epoch == 1:
            test_on_data_set(model, device,'epoch_{}'.format(epoch))
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
    model = VAE(img_size=img_size, latent_size=latent_size).to(device)
    checkpoint = torch.load(n_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_on_data_set(model, device, 'test')


def test_on_data_set(model, device, filename_suffix):
    data_set = np.load(train_file)
    data_size = len(data_set)
    idx = np.random.randint(0, data_size, size=10)
    data = data_set[idx]
    with torch.no_grad():
        data = torch.from_numpy(data).float().to(device)
        data /= 255
        data = data.permute([0, 3, 1, 2])
        data = data.reshape([-1, 3, img_size, img_size])
        recon, mu, logvar = model(data)

        recon = recon.view(10, 3, img_size, img_size)

        comparison = torch.cat([data, recon])
        save_image(comparison.cpu(), 'results/reconstruction_{}.png'.format(filename_suffix),
                   nrow=10)

def load_Vae(path, img_size, latent_size, no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model = VAE(img_size=img_size, latent_size=latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model

#adapted from https://github.com/Natsu6767/Variational-Autoencoder/blob/master/main.py
import matplotlib.pyplot as plt
from scipy.stats import norm


def show_2d_manifold(no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE(img_size=img_size, latent_size=latent_size).to(device)
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


def show_2d_manifold_with_fixed_axis(no_cuda=False, seed=1, fixed_axis=0, fixed_prob_val=0.5):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda else "cpu")
    model = VAE(img_size=img_size, latent_size=latent_size).to(device)
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


def show_1d_manifold(no_cuda=False, seed=1):
    cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")
    model = VAE(img_size=img_size, latent_size=latent_size).to(device)
    checkpoint = torch.load(n_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    n = 20  # figure with nxn images
    figure = np.zeros((img_size*n,img_size, 3))

    # Contruct grid of latent variable values.
    grid_x = norm.ppf(np.linspace(0.00005, 0.99995, n, endpoint=True))  # with probabilities to values of the distribution
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
    # Train VAE
    print('Train VAE...')
    #train_Vae(batch_size=128, epochs=80, load=False)
    #test_Vae()
    #show_2d_manifold()
    show_1d_manifold()
    print('Successfully trained VAE')