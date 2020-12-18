import torch
import torchvision
import pandas as pd
import os
import numpy as np
from torch.utils.data import SubsetRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import io
from tqdm import tqdm
from matplotlib import patches
import math
import sys
from dist_net import DistNet

def strIsNaN(s):
    return s != s

def flat_entries(bboxes_list, ppair):
    return np.concatenate([bboxes_list.ravel(), ppair.ravel()])


def label_str_to_numpy(labelstr):
    if strIsNaN(labelstr):
        # empty image
        print('error')
        exit()
    else:
        label = np.atleast_1d(np.loadtxt(io.BytesIO(labelstr.encode('utf-8'))))
    return label

class DistancesDataset(torch.utils.data.Dataset):
    def __init__(self,root,transform=None, transform_output=None):
        self.transform = transform
        self.transform_output = transform_output
        self.root=root
        data=pd.read_csv(os.path.join(root, "distances.csv"))
        self.data = data
        self.max_min_entries = pd.read_csv(os.path.join(root, "dist_info.csv"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        bboxes_list = label_str_to_numpy(self.data['bbox'][idx])
        ppair = label_str_to_numpy(self.data['ppair'][idx])
        x = flat_entries(bboxes_list, ppair)
        x = torch.from_numpy(x).float()
        if self.transform is not None:
            x = self.transform(x)

        distance = self.data['distance'][idx]

        if distance == 9999:
            distance = self.max_min_entries['max'][0] + 1.
            #one is True
            is_infinite = torch.from_numpy(np.array(1.)).float()
        else:
            #zero is False
            is_infinite = torch.from_numpy(np.array(0.)).float()

        distance = torch.from_numpy(np.array(distance)).float()
        if self.transform_output is not None:
            distance = self.transform_output(distance)

        labels = {'distance': distance,
                  'is_infinite':is_infinite}
        return x, labels


def combine_batch_els(batch):
    batch_dict = {}
    x_els = []
    for x, labels_dict in batch:
        x_els.append(x)
        for key, value in labels_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    for key, value in batch_dict.items():
        batch_dict[key] = torch.stack(value, dim=0)
    x_els = torch.stack(x_els, dim=0)
    return x_els, batch_dict

class NormalizeTransform(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = x - self.mean
        x = x / self.std

        return x


class Solver(object):
    def __init__(self, model, train_loader, eval_loader, loss_func, optimizer, device, save_file_path,
                 verbose=True, print_every=100):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.save_file_path = save_file_path

        self.verbose = verbose
        self.print_every = print_every

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_loss = None

        self.train_loss_history = []
        self.val_loss_history = []

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        model = self.model
        loss_func = self.loss_func
        optimizer = self.optimizer
        iterator = iter(self.train_loader)

        model.train()
        for batch_idx, data in enumerate(iterator):
            x, labels_dict = data
            x = x.to(self.device)
            for key, val in labels_dict.items():
                labels_dict[key] = val.to(self.device)

            optimizer.zero_grad()
            y_pred_dict = model(x)
            loss = loss_func(y_pred_dict, labels_dict)
            loss.backward()
            optimizer.step()

    def check_loss(self, validation=True):
        model = self.model
        loss_func = self.loss_func
        if validation:
            loader = self.eval_loader
        else:
            loader = self.train_loader
        iterator = iter(loader)
        acc_loss = 0

        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(iterator):
                x, labels_dict = data
                x = x.to(self.device)
                for key, val in labels_dict.items():
                    labels_dict[key] = val.to(self.device)

                y_pred_dict = model(x)
                loss = loss_func(y_pred_dict, labels_dict)
                acc_loss += loss.item()

        return acc_loss / (batch_idx + 1)

    def train(self, epochs=1000):
        """
        Run optimization to train the model.
        """

        for t in range(epochs):
            # Update the model parameters.
            self._step()

            # Check the performance of the model.
            train_loss = self.check_loss(validation=False)
            val_loss = self.check_loss(validation=True)

            # Record the losses for later inspection.
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # Keep track of the best model
            self.update_best_loss(val_loss)

            if self.verbose and t % self.print_every == 0:
                print('(Epoch %d / %d) train loss: %f; val_loss: %f' % (
                    t, epochs, train_loss, val_loss))
                save_checkpoint(self.model, self.optimizer, weights_path=self.save_file_path, epoch=t)
        save_checkpoint(self.model, self.optimizer, weights_path=self.save_file_path+'_last')


    def update_best_loss(self, val_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_val_loss or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_checkpoint(self.model, self.optimizer, weights_path=self.save_file_path+'_best')


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


def calc_loss(pred_dict, labels_dict):
    loss = torch.nn.functional.mse_loss(pred_dict['distance'], labels_dict['distance'])
    return loss

def load_DistNet_model(path, device, input_size, output_size, seed=1):
    torch.manual_seed(seed)

    checkpoint = torch.load(path)
    model = DistNet(device=device, input_size=input_size, output_size=output_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    '''loader = torch.utils.data.DataLoader(dataset, batch_size=1000000, collate_fn=combine_batch_els)
    iterator = iter(loader)

    data_x, data_dict = next(iterator)
    x_acc_mean = data_x.mean(dim=0)
    x_acc_std = data_x.std(dim=0)
    d_acc_mean = data_dict['distance'].mean()
    d_acc_std = data_dict['distance'].std()
    print(x_acc_mean)
    print(x_acc_std)
    print(d_acc_mean)
    print(d_acc_std)'''
    '''tensor([1.3007, 0.7500, 0.0900, 0.0300, 1.3002, 0.7500, 1.2999, 0.7501],
       dtype=torch.float64)
    tensor([0.0325, 0.0000, 0.0000, 0.0000, 0.1450, 0.1451, 0.1451, 0.1452],
       dtype=torch.float64)
    tensor(0.4120, dtype=torch.float64)
    tensor(0.4274, dtype=torch.float64)'''

    #todo make param
    normTransformInput = NormalizeTransform(
        mean=torch.from_numpy(
            np.array([1.3007, 0.7500, 0.0900, 0.0300, 1.3002, 0.7500, 1.2999, 0.7501])
        ).float(),
        std=torch.from_numpy(
            np.array([0.0325, 1e-8, 1e-8, 1e-8, 0.1450, 0.1451, 0.1451, 0.1452])
        ).float()
    )

    normTransformOutput = NormalizeTransform(mean=0.4120, std=0.4274)

    dataset = DistancesDataset(root='data/FetchPushMovingObstacleEnv-v1', transform=normTransformInput,
                               transform_output=normTransformOutput)

    validation_split = 0.4
    batch_size = 12

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, collate_fn=combine_batch_els)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, collate_fn=combine_batch_els)

    device = 'cuda:0'
    model = DistNet(device=device, input_size=8, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    seed = 1
    torch.manual_seed(seed)

    solver = Solver(model=model, train_loader=train_loader, eval_loader=validation_loader,
                    loss_func=calc_loss, optimizer=optimizer,device=device,
                    save_file_path='data/FetchPushMovingObstacleEnv-v1/dist_model', print_every=1, verbose=True)
    solver.train(epochs=8)

    plt.plot(solver.val_loss_history, label="Validation Loss")
    plt.plot(solver.train_loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
