import torch
from torch import nn

class DistNet(nn.Module):
    def __init__(self, input_size, output_size, val_infinite,device, hidden_size=256):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.val_infinite = torch.tensor(val_infinite).float().to(device)



        self.common = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.pred_inf_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.pred_dist_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x_common = self.common(x)
        is_infinite_logits = self.pred_inf_net(x_common)
        is_infinite_pr = torch.sigmoid(is_infinite_logits)
        is_infinite_pr_flat = is_infinite_pr.squeeze()
        pred_dist = self.pred_dist_net(x_common)
        pred_dist = is_infinite_pr * self.val_infinite + (1. - is_infinite_pr) * pred_dist
        pred_dist = pred_dist.squeeze()


        ans_dict = {'is_infinite': is_infinite_pr_flat,
                    'distance': pred_dist}
        return ans_dict