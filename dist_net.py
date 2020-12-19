import torch
from torch import nn

class DistNet(nn.Module):
    def __init__(self, input_size, output_size,device, hidden_size=256):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.bb_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )



    def forward(self, x):
        y = self.bb_net(x)
        y = y.squeeze()
        y = torch.sigmoid(y)
        #ans_dict = {'distance':}
        ans_dict = {'is_infinite':y}
        return ans_dict