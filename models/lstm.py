##############################################################################################################################################
#   This LSTM Model is developed based on the source below
#   https://stackoverflow.com/questions/65596522/lstm-for-time-series-prediction-failing-to-learn-pytorch
###############################################################################################################################################
import torch
import torch.nn as nn
from torch.autograd import Variable

DEVICE = 'cuda:0'
#DEVICE = 'cpu'
class LSTM(nn.Module):
    def __init__(self, device,input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.device = device

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        x = x.to(DEVICE)
        self.lstm.to(DEVICE)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))


        self.fc.to(DEVICE)
        out = self.fc(out[:, -1, :])
        # out.size() --> batchsize, output_dim
        return out
