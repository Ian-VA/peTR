import matplotlib.pyplot as plt
import torch

epochs = []

for i in range(1, 51):
    epochs.append(i)

v1 = torch.load("vallosscfc.pt")
v2 = torch.load("vallosslstm.pt")


plt.plot(epochs, v1, label="CfC Validation Loss")
plt.plot(epochs, v2, label="LSTM Validation Loss")

plt.legend()
plt.show()
