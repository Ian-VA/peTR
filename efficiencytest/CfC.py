import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC

def get_cfc(in_features)
    wiring = AutoNCP(28, 1)
    cfc_model = CfC(in_features, wiring, batch_first=True)
    return cfc_model

