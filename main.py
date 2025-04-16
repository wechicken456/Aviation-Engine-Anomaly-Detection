#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import d2l.torch as d2l
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from pandas.api.types import is_numeric_dtype
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler

torch.manual_seed(0)


# In[4]:


dfs = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_excel(f"./data/flight_data_batch{i}.xlsx")
    dfs = pd.concat([dfs, df], ignore_index=True)


# In[5]:


for i in range(1, 1):
    df = pd.read_excel(f"./data/flight_data_batch6_part{i}.xlsx")
    dfs = pd.concat([dfs, df], ignore_index=True)


# In[6]:


# only visualizing a portion of data
#all_data = df.iloc[:9000]


all_data = dfs.groupby("label")

normal_groups = all_data.get_group(0)
bad_groups = all_data.get_group(1)    


# In[7]:


# compare normal vs bad flights
normal_flights = normal_groups.groupby("flight_id")
normal_flights = [f for _, f in normal_flights]
bad_flights = bad_groups.groupby("flight_id")
bad_flights = [f for _, f in bad_flights]


# In[8]:


class DataModule(d2l.HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, is_train=True):
        raise NotImplementedError


    def train_dataloader(self):
        return self.get_dataloader(is_train=True)


    def val_dataloader(self):
        return self.get_dataloader(is_train=False)


    def get_tensorloader(self, tensors, is_train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=is_train)

class PlaneData(DataModule):
    def __init__(self, data_frame, trunc_num = 5000, batch_size=64, 
                 num_steps = 50, train_val_ratio = 0.8, is_test = False, num_cols=None):
        super().__init__()
        self.save_hyperparameters()

        # only visualizing `trunc_num` data points/rows.
        # group by label (bad/normal flights), then group by flight_id, then sort data by timestamp
        all_data = data_frame.iloc[:trunc_num]

        start = time.time()
        all_data = all_data.groupby("label") 

        normal_groups = all_data.get_group(0)

        normal_flights = normal_groups.groupby("flight_id")
        normal_flights = [f for _, f in normal_flights] # a list now
    

        self.train_X = []
        self.val_X = []
        self.train_Y = []
        self.val_Y = []
        self.num_train = 0
        self.num_val = 0

        # sort each flight by time, then truncate to multiple of num_steps, then scale it separately from other flights
        # Then combine it into the training/val sets
        for flight in normal_flights:
            flight = flight.sort_values(by="time", kind="stable")
            flight = flight.drop(["time", "flight_id"], axis=1)
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            
            if num_cols:
                flight = flight.iloc[:, :num_cols]
            
            self.label_names = flight.columns.tolist()
            #scaler = StandardScaler()
            scaler = MinMaxScaler()
            scaler.fit(flight)

            scaled = scaler.transform(flight) # scale 
            scaled_tensor = torch.tensor(scaled, dtype=torch.float32)

            # Create input-output pairs
            X_seqs, Y_seqs = self.create_sequences(scaled_tensor)
            for i in range(1, X_seqs.shape[0]):
                assert(X_seqs[i][-1].equal(Y_seqs[i-1]))

            # split this flight into train/val sets
            num_train = int(len(X_seqs) * self.train_val_ratio)
            num_val = len(X_seqs) - num_train
            self.num_train += num_train
            self.num_val += num_val
            
            self.train_X.append(X_seqs[:num_train])
            self.train_Y.append(Y_seqs[:num_train])
            self.val_X.append(X_seqs[num_train:])
            self.val_Y.append(Y_seqs[num_train:])

        # After processing all flights, concatenate along the batch dimension
        self.train_X = torch.cat(self.train_X, dim=0)  # (total_sequences, num_steps - 1, num_features)
        self.train_Y = torch.cat(self.train_Y, dim=0)  # (total_sequences, num_features)
        self.val_X = torch.cat(self.val_X, dim=0)  # (total_sequences, num_steps - 1, num_features)
        self.val_Y = torch.cat(self.val_Y, dim=0)  # (total_sequences, num_features)
        
        # shape of X: (number of sequences, num_steps, # of features of the raw data) 
        # last dim is the number of different features (e.g. pitch, roll, etc) that each data point has
        # X is input, Y is label.
        # Y is the data point after X.
        print("train_X's shape: ", self.train_X.shape)
        print("train_Y's shape: ", self.train_Y.shape)
        print("val_X's shape: ", self.val_X.shape)
        print("val_Y's shape: ", self.val_Y.shape)
        end = time.time()
        print(f"Processing Time: {end - start}")

        
    # def create_sequences(self, flight: torch.Tensor):
    #     """
    #     Create input-output sequences from a single flight tensor.
        
    #     Args:
    #         flight (torch.Tensor): Tensor of shape (num data points, num_features)
        
    #     Returns:
    #         X (torch.Tensor): Input sequences of shape (num_sequences, num_steps - 1, num_features)
    #         Y (torch.Tensor): Targets of shape (num_sequences, num_features)
    #     """
    #     X, Y = [], []
    
    #     for i in range(flight.shape[0] - self.num_steps):
    #         seq = flight[i:i + self.num_steps]
    #         X.append(seq[:-1])  # first num_steps - 1 as input
    #         Y.append(seq[-1])   # last step as prediction target
    
    #     X = torch.stack(X)  # (num_sequences, num_steps - 1, num_features)
    #     Y = torch.stack(Y)  # (num_sequences, num_features)
    
    #     #print(f"create_sequences: X.shape = {X.shape}, Y.shape = {Y.shape}")
    #     return X, Y
    def create_sequences(self, flight: torch.Tensor):
        """
        Create input-output sequences from a single flight tensor.
        
        Args:
            flight (torch.Tensor): Tensor of shape (num data points, num_features)
        
        Returns:
            X (torch.Tensor): Input sequences of shape (num_sequences, num_steps - 1, num_features)
            Y (torch.Tensor): Targets of shape (num_sequences, num_features)
        """
        X, Y = [], []
        num_seqs = flight.shape[0] - self.num_steps
        if num_seqs <= 0:
            raise f"Not Enough Sequences. flight.shape[0] = {flight.shape[0]}, self.num_steps = {self.num_steps}"
        # broadcast then add. Shape of X_indices: (num_seqs, num_steps)
        # basically, X_indices[i] = [i, i + 1, ... , i + self.num_steps - 1] is the i-th sequence from the dataset
        X_indices = torch.arange(num_seqs)[:, None] + torch.arange(self.num_steps) 
        X = flight[X_indices] # Shape of X: (num_seqs, num_steps, num_features)
        Y = flight[self.num_steps:] # Shape of Y: (num_seqs, num_features)
        return X, Y
        

        # broadcast then add. Shape of X_indices: (num_seqs, num_steps)
        
    def get_dataloader(self, is_train=False):
        if self.is_test:
            # overfit 1 single batch to verify that we can reach the lowest training loss (0)
            X, Y = self.X[1240:1240+self.batch_size], self.Y[1240:1240+self.batch_size] # get a batch
            X = X.repeat(20, 1, 1)  # replicate this batch
            Y = Y.repeat(20, 1) # replicate this batch
            #print(X.shape, Y.shape)
            idx = slice(0, len(X)-self.batch_size) if is_train else slice(len(X)-self.batch_size, None) 
            return self.get_tensorloader([X, Y], is_train, idx)
        else:
            if is_train:
                return self.get_tensorloader([self.train_X, self.train_Y], is_train, slice(0, None))
            else:
                return self.get_tensorloader([self.val_X, self.val_Y], is_train, slice(0, None))



# In[9]:


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P, shape of P: (max sequence length, input embeddings dim = num_hiddens)
        self.P = torch.zeros((1, max_len, num_hiddens))
        # initialize the positional encoding
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 0::2 in the third dimension means "select every second element starting from index 0."
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # make sure the second dim matches
        X = X + self.P[:, X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionwiseFFN(nn.Module):
    def __init__(self, num_hiddens, num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(num_hiddens)
        self.gelu = nn.GELU()
        self.dense2 = nn.LazyLinear(num_outputs)

    def forward(self, X):
        return self.dense2(self.gelu(self.dense1(X)))
        
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_hiddens, num_hiddens_ffn, num_blks, num_hiddens_latent, 
                 num_heads=4, dropout=0.2, bias=True):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        # self.embed = nn.LazyLinear(num_hiddens) # project to higher dimension
        self.pos_encoding = PositionalEncoding(num_hiddens)
        self.blks = nn.Sequential()
        for i in range(self.num_blks):
            self.blks.add_module(f"blk#{i}", TransformerEncoder(num_hiddens, num_hiddens_ffn, num_heads, 
                                                                dropout, bias))
    
    def forward(self, X):
        X = self.pos_encoding(self.embed(X))
        
    

class Transformer(nn.Module):
    def __init__(self, num_hiddens, num_hiddens_ffn, num_blks, num_heads, dropout, bias=True):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        
    def forward(self, X):
        pass
    


# In[10]:


def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)


class RNN(d2l.Module):
    def __init__(self, num_features, num_hiddens, num_hiddens_ffn, num_hiddens_latent, 
                 num_lstm_layers, dropout=0.2, bias=True, lr = 0.1):
        
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_features, num_hiddens, num_layers = 2, batch_first = True)

    def init_weights(self):
        self.apply(init_normal)

    def forward(self, X, H_C = None):
        output, H_C = self.rnn(X, H_C)
        return output, H_C


# In[11]:


class MainModel(d2l.Module):
    def __init__(self, rnn, num_features, lr = 0.1, wd = 1e-5):
        """
        num_features: how many features are there? Is the last dim of a batch: (batch_size, num_steps, num_inputs) 
        num_hiddens: dim for each variable (e.g. torque, yaw, etc)
        num_hiddens_ffn: for transformer
        num_blks: # enc blocks for transformer 
        num_heads: # heads for transformer encoder block
        num_hiddens_latent: for AE
        """
        super().__init__()
        self.save_hyperparameters()
        self.rnn = rnn
        self.rnn.init_weights()
        #self.dense1 = nn.LazyLinear(int(rnn.num_hiddens / 2))
        self.dense2 = nn.LazyLinear(num_features)
        self.init_weights()
        self.train_loss, self.val_loss = [], []

    def init_weights(self):
        self.apply(init_normal)
        
    def forward(self, X):
        """
        `output`: final output of the each cell in the last layer, depth-wise
        `H`: final output of the last cell in a sequence in the batch, timestep-wise
        Check my notebook in Notion for a visualization of this
        output's shape: [batch size, num steps, hidden size)
        H's output: [batch size, hidden size]
        both output[:, -1, :] and h_n[-1, :, :] give (batch, hidden_size)
        """
        _, (H, _) = self.rnn(X)
        output = self.dense2(H[-1])
        #output = self.dense2(self.dense1(H[-1])) # we have 2 LSTM layers, take the last one
        #print(output.shape)
        return output

    def loss(self, Y_hat, Y):
        # right now Y_hat is the output of the dense after the RNN with dim (batch_size, 1, num_features)
        fn = nn.HuberLoss()
        #print(Y_hat.shape, Y.shape)
        return fn(Y_hat, Y)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return optimizer, scheduler

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])        
        self.plot('loss', l, train=True)
        return l
        
    def validation_step(self, batch):
        #print(len(batch), batch[0].shape, batch[-1].shape)
        l = self.loss(self(*batch[:-1]), batch[-1])        
        self.plot('loss', l, train=False)
        return l


# In[25]:


class Trainer(d2l.Trainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.val_losses = []
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

         
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim, self.scheduler = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
                self.scheduler.step(loss)
            self.train_batch_idx += 1
            self.train_losses.append(float(loss))
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
                self.val_losses.append(float(loss))
            self.val_batch_idx += 1


# In[13]:


num_steps = 50
data = PlaneData(dfs, trunc_num=-1, num_steps=num_steps, is_test=False, num_cols = None)
#print(data.Y.shape)


# In[26]:


num_features = data.train_X.shape[2] 
num_hiddens, num_hiddens_ffn, num_blks, num_hiddens_latent = 128, 128, 4, 64
num_lstm_layers = 2
num_heads, dropout, bias, lr = 4, 0.2, True, 5e-4
rnn = RNN(num_features, num_hiddens, num_hiddens_ffn, num_hiddens_latent, 
             num_lstm_layers, dropout, bias, lr)
model = MainModel(rnn, num_features, lr)

trainer = Trainer(max_epochs=50, num_gpus=0)
trainer.fit(model, data)


# In[ ]:


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
    
count_parameters(model)


# In[ ]:


# col_names = data.label_names

# # get 5 sequences, then flatten the first 2 dim out to be the pseudo-time dim so we can graph it.
# data_X = data.val_X[:8]
# data_Y = data.val_Y[:8]
# num_graphs = data_Y.shape[1]
# # t = torch.arange(data_X.shape[0]*data_X.shape[1])

# print(data_X.shape, data_Y.shape)
# for i in range(num_graphs):
#     fig, axes = plt.subplots(1, figsize=(13, 4), sharex="col", sharey="row")
#     plt.subplots_adjust(hspace=0.05, wspace=0.05)  # Reduced wspace from default (~0.2) to 0.1

#     # .reshape(-1, data.X.shape[2])
#     # for each label, plot it with its input
#     axes.set_ylabel(col_names[i])
#     axes.legend(["Input", "Label"])
#     num_steps = data_X.shape[1]
#     t_Y = -1
#     for j in range(data_Y.shape[0]):
#         t_X = torch.arange(start = t_Y + 1, end = t_Y + num_steps + 1) 
#         t_Y = t_Y + num_steps + 1
#         axes.plot(t_X, data_X[j, :, i]) # input sequence
#         axes.plot(t_Y, data_Y[j, i], color="red", marker='o', markersize=10) # label


# In[30]:
plt.savefig("losses.pdf")

with open("./losses.txt", "w") as f:
    f.write(", ".join([str(i) for i in trainer.train_losses]))
    f.write("\n")
    f.write(", ".join([str(i) for i in trainer.val_losses]))

