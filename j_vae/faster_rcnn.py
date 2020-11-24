import torch
import torchvision
import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import io
from tqdm import tqdm
from matplotlib import patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import j_vae.engine_utils as engine_utils
import math
import sys

def strIsNaN(s):
    return s != s

def box_str_to_numpy(boxstr):
    if strIsNaN(boxstr):
        # empty image
        raise Exception('no box can be empty')
        exit()
    else:
        boxes = np.loadtxt(io.BytesIO(boxstr.encode('utf-8')))
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
    return boxes


def label_str_to_numpy(labelstr):
    if strIsNaN(labelstr):
        # empty image
        print('error')
        exit()
    else:
        label = np.atleast_1d(np.loadtxt(io.BytesIO(labelstr.encode('utf-8'))).astype(int))
    return label


class wheatdataset(torch.utils.data.Dataset):
    def __init__(self,root,folder,transforms=None):
        self.transforms=[]
        if transforms!=None:
            self.transforms.append(transforms)
        self.root=root
        self.folder=folder
        box_data=pd.read_csv(os.path.join(root, "all_set.csv"))
        box_data['bbox'] = box_data['bbox'].apply(box_str_to_numpy)
        box_data['labels'] = box_data['labels'].apply(label_str_to_numpy)
        self.box_data = box_data
        self.imgs=list(os.listdir(os.path.join(root, self.folder)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path=os.path.join(os.path.join(self.root,self.folder),self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        df=self.box_data[self.box_data['im_name']==self.imgs[idx]]
        if df.shape[0]!=0:
            boxes = df['bbox'].values[0]
            labels = df['labels'].values[0]
        boxes = torch.from_numpy(boxes).float()


        for i in self.transforms:
            img=i(img)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        targets={}
        targets['boxes']=boxes
        targets['labels']=torch.from_numpy(labels).type(torch.int64)
        targets["image_id"] = image_id
        targets["area"] = area
        targets["iscrowd"] = iscrowd


        return img,targets



def view(images,labels,k,fname):
    figure = plt.figure(figsize=(30,30))
    images=list(images)
    labels=list(labels)
    for i in range(k):
        out=torchvision.utils.make_grid(images[i])
        ax = figure.add_subplot(2,2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
        if 'scores' in labels[i].keys():#when visualizing scores was not given but then in general is always a parameter
            keep = torchvision.ops.nms(boxes=labels[i]['boxes'], scores=labels[i]['scores'], iou_threshold=0.25)
            boxes_l = labels[i]['boxes'][keep]
            l = boxes_l.cpu().numpy()
        else:
            l=labels[i]['boxes'].cpu().numpy()
        #l[:,2]=l[:,2]-l[:,0]
        #l[:,3]=l[:,3]-l[:,1]
        for j in range(len(l)):
            #ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=5,edgecolor='black',facecolor='none'))
            ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),np.abs(l[j][2]-l[j][0]),
                                           np.abs(l[j][3]-l[j][1]),linewidth=5,edgecolor='black',facecolor='none'))
    plt.savefig(fname)
    plt.close()

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_faster_rcnn(path, device):

    model = get_model_instance_segmentation(4)
    model = model.to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    '''if train:
        transforms.append(T.RandomHorizontalFlip(0.5))'''
    return T.Compose(transforms)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = engine_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', engine_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = engine_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = engine_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

if __name__ == '__main__':
    root = '../data/FetchGenerativeEnv-v1'
    dataset = wheatdataset(root, folder='images', transforms=get_transform(train=True))
    dataset_test = wheatdataset(root, folder='images', transforms=get_transform(train=False))
    #torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])#:-1000])#todo change to 2100
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])#-1000:])#todo change to 2100
    data_loader_train = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                                    collate_fn=lambda x: list(zip(*x)))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False,
                                                   collate_fn=lambda x: list(zip(*x)))

    model = get_model_instance_segmentation(4)

    '''images, labels = next(iter(data_loader_train))
    view(images, labels, 4, fname='results/rcnn_test.png')'''

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)



    model_save_path = '../data/FetchGenerativeEnv-v1/model_rcnn.pth'
    optimizer_save_path = '../data/FetchGenerativeEnv-v1/optimizer_rcnn.pth'
    scheduler_save_path = '../data/FetchGenerativeEnv-v1/scheduler_rcnn.pth'
    model_save_path_ep = '../data/FetchGenerativeEnv-v1/model_rcnn_epoch_{}.pth'
    optimizer_save_path_ep = '../data/FetchGenerativeEnv-v1/optimizer_rcnn_epoch_{}.pth'
    scheduler_save_path_ep = '../data/FetchGenerativeEnv-v1/scheduler_rcnn_epoch_{}.pth'

    checkpoint = torch.load(model_save_path_ep.format(45))
    model.load_state_dict(checkpoint)
    it = 0
    test_iter = iter(data_loader_test)
    while it < 10:
        images, targets = next(test_iter)
        images = list(image.to(device) for image in images)
        model.eval()
        with torch.no_grad():
            output = model(images)
            view(images, output, min(4, len(images)), fname='results/rcnn_epoch_test_{}.png'.format(it))
        it +=1

    '''params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    torch.save(lr_scheduler.state_dict(), scheduler_save_path)
    model.train()
    total_epoches = 150

    for epoch in tqdm(range(total_epoches)):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, 40)
        if epoch % 5 == 0 or epoch == total_epoches - 1:
            images, targets = next(iter(data_loader_test))
            images = list(image.to(device) for image in images)
            model.eval()
            with torch.no_grad():
                output = model(images)
                view(images, output, min(4, len(images)), fname='results/rcnn_epoch_{}.png'.format(epoch))

            images, targets = next(iter(data_loader_train))
            images = list(image.to(device) for image in images)
            with torch.no_grad():
                output = model(images)
                view(images, output, min(4, len(images)), fname='results/rcnn_epoch_{}_train.png'.format(epoch))
            torch.save(model.state_dict(), model_save_path_ep.format(epoch))
            torch.save(optimizer.state_dict(), optimizer_save_path_ep.format(epoch))
            torch.save(lr_scheduler.state_dict(), scheduler_save_path_ep.format(epoch))'''


    '''torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    torch.save(lr_scheduler.state_dict(), scheduler_save_path)'''