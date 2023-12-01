import torch
from ncps.torch import CfC
from torch.utils.data import DataLoader, Dataset

rnn = CfC(20, 50)
x = torch.randn(2, 3, 20)
h0 = torch.zeros(2, 50)
output, hn = rnn(x, h0)


class BoundingBoxesDataset(Dataset):
    def __init__():



