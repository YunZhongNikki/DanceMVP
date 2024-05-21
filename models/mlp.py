##############################################################################################################################################
#   This MLP Model is developed based on the source below
#   https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb#scrollTo=lAqzcW9XREvu
###############################################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = 'cuda:0'
#DEVICE = 'cpu'
class MLP(nn.Module):
    def __init__(self, device,input_dim, output_dim, project=True):
        super().__init__()

        self.device = device
        if project == True:
            # self.input_fc = nn.Linear(input_dim, 128)#258
            # self.hidden_fc = nn.Linear(128, 64)#256,128
            # self.output_fc = nn.Linear(64, output_dim)#128
            self.input_fc = nn.Linear(input_dim, 256)  # 258
            self.hidden_fc = nn.Linear(256, 256)  # 256,128
            self.output_fc = nn.Linear(256, output_dim)  # 128
        else:
            self.input_fc = nn.Linear(input_dim, output_dim)
            self.hidden_fc = nn.Linear(output_dim, output_dim)
            self.output_fc = nn.Linear(output_dim, output_dim)
            # self.input_fc = nn.Linear(input_dim, 256)
            # self.hidden_fc = nn.Linear(256, 128)
            # self.output_fc = nn.Linear(128, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        # x = [batch size, height * width]
        self.input_fc.to(DEVICE)
        self.hidden_fc.to(DEVICE)
        self.output_fc.to(DEVICE)
        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 256]
        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 128]
        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2
