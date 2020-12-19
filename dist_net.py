import torch
from torch import nn

class DistNet(nn.Module):
    def __init__(self, input_size, output_size, val_infinite,device, hidden_size=256):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.val_infinite = torch.tensor(val_infinite).float()
        self.bb_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        self.d2 = nn.Sequential(
            nn.Linear(input_size+1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )



    def forward(self, x):
        is_infinite_logits = self.bb_net(x)
        is_infinite_pr = torch.sigmoid(is_infinite_logits)
        is_infinite_pr_flat = is_infinite_pr.squeeze()


        #ans_dict = {'distance':}
        ans_dict = {'is_infinite': is_infinite_pr_flat}
        return ans_dict