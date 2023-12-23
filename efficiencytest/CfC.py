import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC
from torch.autograd import Variable
import torch

class CfCModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cfc_model = CfC(input_size, units=50, batch_first=True)
        self.hidden_size = hidden_size       
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.input_size, 64, self.hidden_size).cuda())
        output, hn = self.cfc_model(x, h0)

        return self.linear(hn[-1])
