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
warnings.filterwarnings('ignore')
#%%

def strIsNaN(s):
    return s != s
class wheatdataset(torch.utils.data.Dataset):
    def __init__(self,root,folder,transforms=None):
        self.transforms=[]
        if transforms!=None:
            self.transforms.append(transforms)
        self.root=root
        self.folder=folder
        box_data=pd.read_csv(os.path.join(root, "all_set.csv"))
        self.box_data = box_data
        self.imgs=list(os.listdir(os.path.join(root, self.folder)))
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img_path=os.path.join(os.path.join(self.root,self.folder),self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        df=self.box_data[self.box_data['im_name']==self.imgs[idx]]
        if df.shape[0]!=0:
            boxesstr=df['bbox'].values[0]

            if strIsNaN(boxesstr):
                #empty image
                print('error')
                exit()
                boxes = np.array([[0., 0., 0., 0.]])
                labels = np.array([0]).astype(int)
            else:
                boxes = np.loadtxt(io.BytesIO(boxesstr.encode('utf-8')))
                if boxes.ndim == 1:
                    boxes = np.expand_dims(boxes, axis=0)
                    labelsstr = df['labels'].values[0]
                    labels = np.atleast_1d(np.loadtxt(io.BytesIO(labelsstr.encode('utf-8'))).astype(int))
                else:
                    labelsstr=df['labels'].values[0]
                    labels=np.loadtxt(io.BytesIO(labelsstr.encode('utf-8'))).astype(int)
        for i in self.transforms:
            img=i(img)

        targets={}
        targets['boxes']=torch.from_numpy(boxes).double()
        targets['labels']=torch.from_numpy(labels).type(torch.int64)
        #targets['id']=self.imgs[idx].split('.')[0]
        return img.double(),targets


from matplotlib import patches
def view(images,labels,k,fname):
    figure = plt.figure(figsize=(30,30))
    images=list(images)
    labels=list(labels)
    for i in range(k):
        out=torchvision.utils.make_grid(images[i])
        ax = figure.add_subplot(2,2, i + 1)
        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
        if 'scores' in labels[i].keys():#when visualizing scores was not given but then in general is always a parameter
            keep = torchvision.ops.nms(boxes=labels[i]['boxes'], scores=labels[i]['scores'], iou_threshold=0.3)
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



if __name__ == '__main__':
    root = '../data/FetchGenerativeEnv-v1'
    dataset = wheatdataset(root, folder='images', transforms=torchvision.transforms.ToTensor())
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:2100])
    dataset_test = torch.utils.data.Subset(dataset, indices[2100:])
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=6, shuffle=True,
                                                    collate_fn=lambda x: list(zip(*x)))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=6, shuffle=False,
                                                   collate_fn=lambda x: list(zip(*x)))
    '''images, labels = next(iter(data_loader_train))
    view(images, labels, 4, fname='results/rcnn_test.png')'''

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 4  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)
    model = model.double()


    model_save_path = '../data/FetchGenerativeEnv-v1/model_rcnn.pth'
    optimizer_save_path = '../data/FetchGenerativeEnv-v1/optimizer_rcnn.pth'
    model_save_path_ep = '../data/FetchGenerativeEnv-v1/model_rcnn_epoch_{}.pth'
    optimizer_save_path_ep = '../data/FetchGenerativeEnv-v1/optimizer_rcnn_epoch_{}.pth'

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-6)
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
    model.train()
    total_epoches = 20

    for epoch in tqdm(range(total_epoches)):
        model.train()
        for images, targets in tqdm(data_loader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model = model.double()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()

            optimizer.zero_grad()
            optimizer.step()
            print("Loss = {:.4f} ".format(losses.item()))
        if epoch % 5 == 0 or epoch == total_epoches - 1:
            images, targets = next(iter(data_loader_test))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            model.eval()
            with torch.no_grad():
                output = model(images)
                view(images, output, min(4, len(images)), fname='results/rcnn_epoch_{}.png'.format(epoch))
            torch.save(model.state_dict(), model_save_path_ep.format(epoch))
            torch.save(optimizer.state_dict(), optimizer_save_path_ep.format(epoch))


    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)