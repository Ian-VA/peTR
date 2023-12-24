import torch
import torch.nn as nn
from dataprep import get_data, get_train
from rnn import RNN
import torch.optim as optim
from datetime import datetime
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.cuda.memory as memory
import pickle
from lstm import LSTMNet
from CfC import CfCModel
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import seaborn as sns
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        self.model.train()

        yhat, _ = self.model.forward(x)
        yhat = yhat.view_as(y)
        loss = self.loss_fn(yhat, y)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()

                yhat, _ = self.model(x_test)

                yhat = yhat.view_as(y_test)

                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=10, n_features=1):
        model_path = "cfc_save"
        #memory._record_memory_history()

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat, _ = self.model.forward(x_val)

                    yhat = yhat.view_as(y_val)

                    val_loss = self.loss_fn(yhat, y_val).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            #snap = memory._snapshot()
        
            #with open("snapcfc1.pickle", "wb") as f: pickle.dump(snap, f)

            #torch.save(model.state_dict(), model_path)

            
            print(f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}")

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")

        torch.save(self.train_losses, "traininglosscfc1.pt")
        torch.save(self.val_losses, "vallosscfc1.pt")

        print("saved losses")

        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

input_dim = len(get_train().columns)

output_dim = 1
hidden_dim = 64
layer_dim = 3
batch_size = 64
dropout = 0.2
n_epochs = 50
learning_rate = .01
weight_decay = 1e-6
train_loader, val_loader, test_loader = get_data()

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

wiring = AutoNCP(30, output_dim)

model = CfC(input_dim, wiring, batch_first=True)

# for drawing CfC network diagram
"""
sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=False, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show() 
"""
loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if torch.cuda.is_available():
    model.cuda()

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)

opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=50, n_features=input_dim)

opt.plot_losses()

predictions, values = opt.evaluate(test_loader, batch_size=1, n_features=input_dim)
