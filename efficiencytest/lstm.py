import torch.nn as nn
from torch.autograd import Variable
import torch

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(3, 64, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(3, 64, self.hidden_size).cuda())
        x, (f_h, c_h) = self.lstm(x, (h_0, c_0))

        return self.linear(f_h[-1])
