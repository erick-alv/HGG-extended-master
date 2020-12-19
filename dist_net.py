import torch
from torch import nn

class DistNet(nn.Module):
    def __init__(self, input_size, output_size, val_infinite,device, hidden_size=256):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.val_infinite = torch.tensor(val_infinite).float().to(device)
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
        x_with_pr = torch.cat([x, is_infinite_pr], dim=1)
        pred_dist = self.d2(x_with_pr)
        pred_dist = pred_dist.squeeze()


        ans_dict = {'is_infinite': is_infinite_pr_flat,
                    'distance': pred_dist}
        return ans_dict