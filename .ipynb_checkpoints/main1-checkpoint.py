#!/usr/bin/env python
# coding: utf-8

# In[55]:


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
from sklearn.metrics import precision_recall_curve

torch.manual_seed(0)
plt.tight_layout()


# In[4]:


#plt.rcParams["figure.figsize"] = (45,35)


# In[5]:


dfs = pd.DataFrame()
for i in range(1, 6):
    df = pd.read_excel(f"./data/flight_data_batch{i}.xlsx")
    df.insert(len(df.columns.tolist()), "is_fail", 0)
    fail_df = pd.read_csv(f"./data/Failures/Failure_Events_in_Batch_{i}.csv")
    
    fails = set(zip(fail_df["flight_id"], fail_df["failure_time"]))
    df.loc[df.apply(lambda row: (row["flight_id"], row["time"]) in fails, axis=1), "is_fail"] = 1

    dfs = pd.concat([dfs, df], ignore_index=True)


# In[6]:


for i in range(1, 4):
    df = pd.read_excel(f"./data/flight_data_batch6_part{i}.xlsx")
    
    df.insert(len(df.columns.tolist()), "is_fail", 0)
    fail_df = pd.read_csv(f"./data/Failures/Failure_Events_in_Batch_6_Part_{i}.csv")
    
    fails = set(zip(fail_df["flight_id"], fail_df["failure_time"]))
    df.loc[df.apply(lambda row: (row["flight_id"], row["time"]) in fails, axis=1), "is_fail"] = 1
    
    dfs = pd.concat([dfs, df], ignore_index=True)    
    #dfs = pd.concat([dfs, df], ignore_index=False)


# In[7]:


(dfs["is_fail"] == 1).sum()


# In[143]:


class DataModule(d2l.HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, is_train=True):
        raise NotImplementedError


    def train_dataloader(self):
        return self.get_tensorloader([self.train_X, self.train_Y], is_train=True)

    def val_dataloader(self):
        return self.get_tensorloader([self.val_X, self.val_Y], is_train=False)
        
    def thresh_dataloader(self):
        return self.get_tensorloader([self.thresh_X, self.thresh_Y, self.thresh_labels], is_train=False)
        
    def test_dataloader(self):
        return self.get_tensorloader([self.test_X, self.test_Y, self.test_labels], is_train=False)


    def get_tensorloader(self, tensors, is_train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=is_train)

class PlaneData(DataModule):
    def __init__(self, data_frame, trunc_num = 5000, batch_size=64, 
                 num_steps = 50, train_val_ratio = 0.8, thresh_test_ratio = 0.5, is_test = False):
        """
        So the normal flights (without anomalies. label = 0) will be the training dataset
        Each normal flight is splitted into regression training and regression validation using `train__val_ratio`
        Then, we will use anomalous flights to determine a validation threshold, using the same spltiting with `thresh_test_ratio` instead
        So now the bad flights is splitted into threshold determination set and testing set 
        Finally, we use the testing dataset to compute all the metrics for a threshold
        The second dataset (thresholding and test dataset) will contain a labels tensor
        """
        super().__init__()
        self.save_hyperparameters()

        # only visualizing `trunc_num` data points/rows.
        # group by label (bad/normal flights), then group by flight_id, then sort data by timestamp
        all_data = data_frame.iloc[:trunc_num]

        start = time.time()
        flight_groups = all_data.groupby("flight_id") 
        flight_ids = list(flight_groups.groups.keys())

        normal_flight_ids = [fid for fid in flight_ids if flight_groups.get_group(fid)["label"].iloc[0] == 0]
        bad_flight_ids = [fid for fid in flight_ids if flight_groups.get_group(fid)["label"].iloc[0] == 1]
        num_thresh_flights = int(len(bad_flight_ids) * thresh_test_ratio)
        thresh_flight_ids = bad_flight_ids[:num_thresh_flights]
        test_flight_ids = bad_flight_ids[num_thresh_flights:]

        self.train_X = []
        self.val_X = []
        self.train_Y = []
        self.val_Y = []
        self.thresh_X, self.thresh_Y, self.thresh_labels = [], [], []
        self.test_X, self.test_Y, self.test_labels = [], [], []
        self.num_train = 0
        self.num_val = 0
        self.num_thresh = 0
        self.num_test = 0

        # sort each flight by time, then truncate to multiple of num_steps, then scale it separately from other flights
        # Then combine it into the training/val sets
        for fid in normal_flight_ids:
            flight = flight_groups.get_group(fid).sort_values(by="time", kind="stable")
            flight = flight.drop(["flight_id", "is_fail", "time"], axis=1)
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            
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

        # Same thing as the above loop, with `thresh_flight_ids` instead
        for fid in thresh_flight_ids:
            flight = flight_groups.get_group(fid).sort_values(by="time", kind="stable")
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            labels = flight["is_fail"].values # this line is different from the first loop
            flight = flight.drop(["flight_id", "is_fail", "time"], axis=1)

            self.label_names = flight.columns.tolist()
            #scaler = StandardScaler()
            scaler = MinMaxScaler()
            scaler.fit(flight)

            scaled = scaler.transform(flight) # scale 
            scaled_tensor = torch.tensor(scaled, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # Create input-output pairs
            X_seqs, Y_seqs, labels = self.create_sequences(scaled_tensor, labels_tensor)
            for i in range(1, X_seqs.shape[0]):
                assert(X_seqs[i][-1].equal(Y_seqs[i-1]))
            
            self.thresh_X.append(X_seqs)
            self.thresh_Y.append(Y_seqs)
            self.thresh_labels.append(labels)

         # Same thing as the above loop, with `test_flight_ids` instead
        for fid in thresh_flight_ids:
            flight = flight_groups.get_group(fid).sort_values(by="time", kind="stable")
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            labels = flight["is_fail"].values # this line is different from the first loop
            flight = flight.drop(["flight_id", "is_fail", "time"], axis=1)

            self.label_names = flight.columns.tolist()
            #scaler = StandardScaler()
            scaler = MinMaxScaler()
            scaler.fit(flight)

            scaled = scaler.transform(flight) # scale 
            scaled_tensor = torch.tensor(scaled, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # Create input-output pairs
            X_seqs, Y_seqs, labels = self.create_sequences(scaled_tensor, labels_tensor)
            for i in range(1, X_seqs.shape[0]):
                assert(X_seqs[i][-1].equal(Y_seqs[i-1]))
            
            self.test_X.append(X_seqs)
            self.test_Y.append(Y_seqs)
            self.test_labels.append(labels)
            
        # After processing all flights, concatenate along the batch dimension
        self.thresh_X = torch.cat(self.thresh_X, dim=0)  # (total_sequences, num_steps - 1, num_features)
        self.thresh_Y = torch.cat(self.thresh_Y, dim=0)  # (total_sequences, num_features)
        self.thresh_labels = torch.cat(self.thresh_labels, dim=0) # (total_sequences, 1)
        self.test_X = torch.cat(self.test_X, dim=0)  # (total_sequences, num_steps - 1, num_features)
        self.test_Y = torch.cat(self.test_Y, dim=0)  # (total_sequences, num_features)
        self.test_labels = torch.cat(self.test_labels, dim=0) # (total_sequences, 1)

        # shape of X: (number of sequences, num_steps, # of features of the raw data) 
        # last dim is the number of different features (e.g. pitch, roll, etc) that each data point has
        # X is input, Y is label.
        # Y is the data point after X.
        print("thresh_X's shape: ", self.thresh_X.shape)
        print("thresh_Y's shape: ", self.thresh_Y.shape)
        print("test_X's shape: ", self.test_X.shape)
        print("test_Y's shape: ", self.test_Y.shape)
        
        end = time.time()
        print(f"Processing Time: {end - start}")

    def create_sequences(self, flight: torch.Tensor, labels: torch.Tensor = None):
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
        if labels is None:
            return X, Y
        else:
            return X, Y, labels[self.num_steps:]
        

        # broadcast then add. Shape of X_indices: (num_seqs, num_steps)
        
    # def get_dataloader(self, X, Y, is_train=False):
    #     if is_train:
    #         return self.get_tensorloader([self.train_X, self.train_Y], is_train, slice(0, None))
    #     else:
    #         return self.get_tensorloader([self.val_X, self.val_Y], is_train, slice(0, None))



# In[144]:


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
    


# In[145]:


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


# In[146]:


class MainModel(d2l.Module):
    def __init__(self, rnn, num_features, threshold = 0.1, lr = 0.1, wd = 1e-5):
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
        fn = nn.MSELoss()
        #print(Y_hat.shape, Y.shape)
        return fn(Y_hat, Y)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch):
        Y_hat = self(*batch[:-1])
        Y = batch[-1]
        l = self.loss(Y_hat, Y)        
        self.plot('loss', l, train=True)
        return l
        
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        Y = batch[-1]
        l = self.loss(Y_hat, Y) 
        self.plot('loss', l, train=False)
        return l


# In[147]:


class Trainer(d2l.Trainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.val_losses = []
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.thresh_dataloader = data.thresh_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 15, 30],gamma = 0.1)
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            self.scheduler.step()
            plt.savefig("./losses.png")
            
        self.find_best_threshold()
        self.evaluate_thresholds()      

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

    def find_best_threshold(self):
        self.model.eval()
        
        max_errors_list, labels_list = [], []
        for batch in self.thresh_dataloader:
            X, Y, labels = self.prepare_batch(batch)
            Y_hat = self.model(X)
            # shape of errors: (batch_size, num_features). 
            # Didn't use the self.model.loss function here as it will use a mean reduction over all features. 
            errors = (Y_hat - Y) ** 2
            max_errors = errors.max(dim=1)[0] # for each batch, get the error of the feature that produces the greatest error
            max_errors_list.append(max_errors.detach().cpu().numpy())
            labels_list.append(labels.cpu().numpy())

        thresh_max_errors = np.concatenate(max_errors_list)
        thresh_labels = np.concatenate(labels_list)
        precision, recall, self.thresholds = precision_recall_curve(thresh_labels, thresh_max_errors, pos_label=1)
        f1_scores = 2*(precision*recall) / (precision + recall + 1e-9)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = self.thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]

        self.best_val_threshold_idx = best_threshold_idx
        self.best_threshold = best_threshold
        self.best_val_f1 = best_f1

        anomalies = thresh_max_errors > best_threshold
        tp = np.sum(anomalies & (thresh_labels == 1))
        fp = np.sum(anomalies & (thresh_labels == 0))
        tn = np.sum(~anomalies & (thresh_labels == 0))
        fn = np.sum(~anomalies & (thresh_labels == 1))
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        
        print(f"Optimal Threshold (Thresholding Set): {best_threshold:.6f}")
        print(f"Thresholding Set Metrics: Precision: {prec:.6f}, Recall: {rec:.6f}, F1: {best_f1:.6f}, Accuracy: {acc:.6f}")

    def evaluate_thresholds(self):
        """
        Find reconstruction errors for the test set. Then for each threshold in self.thresholds,
        compute all the metrics against that threshold and graph them
        """
        if self.best_val_f1 is None:
            raise "Please run find_best_threshold() first!"
        
        self.model.eval()
        max_errors_list, labels_list = [], []
        for batch in self.test_dataloader:
            X, Y, labels = self.prepare_batch(batch)
            Y_hat = self.model(X)
            # shape of errors: (batch_size, num_features). 
            # Didn't use the self.model.loss function here as it will use a mean reduction over all features. 
            errors = (Y_hat - Y) ** 2
            max_errors = errors.max(dim=1)[0] # for each batch, get the error of the feature that produces the greatest error
            max_errors_list.append(max_errors.detach().cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
        test_max_errors = np.concatenate(max_errors_list)
        test_labels = np.concatenate(labels_list)
        test_precision = []
        test_recall = []
        test_f1 = []
        test_accuracy = []
        for thresh in self.thresholds:
            anomalies = test_max_errors > thresh
            tp = np.sum(anomalies & (test_labels == 1))
            fp = np.sum(anomalies & (test_labels == 0))
            tn = np.sum(~anomalies & (test_labels == 0))
            fn = np.sum(~anomalies & (test_labels == 1))
            prec = tp / (tp + fp) if tp + fp > 0 else 0
            rec = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec + 1e-9) if prec + rec > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
            test_precision.append(prec)
            test_recall.append(rec)
            test_f1.append(f1)
            test_accuracy.append(acc)
            
        best_test_idx = np.argmax(test_f1)
        best_test_threshold = self.thresholds[best_test_idx]
        print(f"Best Test Set Threshold: {best_test_threshold:.6f}")
        print(f"Test Set Metrics at Best Threshold: Precision: {test_precision[best_test_idx]:.6f}, "
              f"Recall: {test_recall[best_test_idx]:.6f}, F1: {test_f1[best_test_idx]:.6f}, "
              f"Accuracy: {test_accuracy[best_test_idx]:.6f}")
        
        # Visualize metrics vs. thresholds
        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, test_precision, label="Precision", marker='.')
        plt.plot(self.thresholds, test_recall, label="Recall", marker='.')
        plt.plot(self.thresholds, test_f1, label="F1 Score", marker='.')
        plt.plot(self.thresholds, test_accuracy, label="Accuracy", marker='.')
        plt.axvline(x=self.best_threshold, color='red', linestyle='--', label=f"Optimal Threshold ({self.best_threshold:.6f})")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Metrics vs. Threshold on Test Set")
        plt.legend()
        plt.grid(True)
        plt.savefig("./metrics/metrics_vs_threshold.png")
        plt.close()

        # Precision-Recall Curve
        test_precision, test_recall, _ = precision_recall_curve(test_labels, test_max_errors, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(test_recall, test_precision, label="Precision-Recall Curve")
        plt.scatter(test_recall[best_test_idx], test_precision[best_test_idx], color='red',
                    label=f"Best Threshold ({best_test_threshold:.6f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Test Set)")
        plt.legend()
        plt.grid(True)
        plt.savefig("./metrics/precision_recall_curve_test.png")
        plt.close()
            


# In[148]:


num_steps = 50
data = PlaneData(dfs, trunc_num=-1, num_steps=num_steps, is_test=False)
#print(data.Y.shape)


# In[ ]:


num_features = data.train_X.shape[2] 
num_hiddens, num_hiddens_ffn, num_blks, num_hiddens_latent = 128, 128, 4, 64
num_lstm_layers = 2
num_heads, dropout, bias, lr = 4, 0.2, True, 1e-3
rnn = RNN(num_features, num_hiddens, num_hiddens_ffn, num_hiddens_latent, 
             num_lstm_layers, dropout, bias, lr)
model = MainModel(rnn, num_features, lr)

trainer = Trainer(max_epochs=70, num_gpus=2)
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


plt.savefig("./losses.png")
with open("./losses.txt", "w") as f:
    f.write(", ".join([str(i) for i in trainer.train_losses]))
    f.write("\n")
    f.write(", ".join([str(i) for i in trainer.val_losses]))


# In[1]:


net = nn.LazyLinear(5) # dummy net
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 7, 9, 15, 20, 25, 40, 45], gamma = 0.5)

def get_lr(optimizer, scheduler):
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()
    print(lr)
    return lr

d2l.plot(torch.arange(100), [get_lr(optimizer, scheduler)
                                  for t in range(100)])


# In[53]:


t = torch.tensor([[1,2,3, 4],
                  [7,1,9, 10],
                  [4,5,6, 2]])
t.max(dim=1)[0]


# In[62]:


p, r, t = precision_recall_curve([0, 1, 0, 1, 0, 0, 0, 1], [0.02, 0.08, 0.02, 0.09, 0.04, 0.04, 0.03, 0.07])


# In[63]:


p, r, t


# In[102]:


class DataModule(d2l.HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, is_train=True):
        raise NotImplementedError


    def train_dataloader(self):
        return self.get_tensorloader([self.train_X, self.train_Y], is_train=True)

    def val_dataloader(self):
        return self.get_tensorloader([self.val_X, self.val_Y], is_train=False)
        
    def thresh_dataloader(self):
        return self.get_tensorloader([self.thresh_X, self.thresh_Y, self.thresh_labels], is_train=False)
        
    def test_dataloader(self):
        return self.get_tensorloader([self.test_X, self.test_Y, self.test_labels], is_train=False)


    def get_tensorloader(self, tensors, is_train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=is_train)

class PlaneData(DataModule):
    def __init__(self, data_frame, trunc_num = 5000, batch_size=64, 
                 num_steps = 50, train_val_ratio = 0.8, thresh_test_ratio = 0.5, is_test = False):
        """
        So the normal flights (without anomalies. label = 0) will be the training dataset
        Each normal flight is splitted into regression training and regression validation using `train__val_ratio`
        Then, we will use anomalous flights to determine a validation threshold, using the same spltiting with `thresh_test_ratio` instead
        So now the bad flights is splitted into threshold determination set and testing set 
        Finally, we use the testing dataset to compute all the metrics for a threshold
        The second dataset (thresholding and test dataset) will contain a labels tensor
        """
        super().__init__()
        self.save_hyperparameters()

        # only visualizing `trunc_num` data points/rows.
        # group by label (bad/normal flights), then group by flight_id, then sort data by timestamp
        all_data = data_frame.iloc[:trunc_num]

        start = time.time()
        flight_groups = all_data.groupby("flight_id") 
        flight_ids = list(flight_groups.groups.keys())

        normal_flight_ids = [fid for fid in flight_ids if flight_groups.get_group(fid)["label"].iloc[0] == 0]
        bad_flight_ids = [fid for fid in flight_ids if flight_groups.get_group(fid)["label"].iloc[0] == 1]
        num_thresh_flights = int(len(bad_flight_ids) * thresh_test_ratio)
        thresh_flight_ids = bad_flight_ids[:num_thresh_flights]
        test_flight_ids = bad_flight_ids[num_thresh_flights:]

        self.train_X = []
        self.val_X = []
        self.train_Y = []
        self.val_Y = []
        self.thresh_X, self.thresh_Y, self.thresh_labels = [], [], []
        self.test_X, self.test_Y, self.test_labels = [], [], []
        self.num_train = 0
        self.num_val = 0
        self.num_thresh = 0
        self.num_test = 0

        # sort each flight by time, then truncate to multiple of num_steps, then scale it separately from other flights
        # Then combine it into the training/val sets
        for fid in normal_flight_ids:
            flight = flight_groups.get_group(fid).sort_values(by="time", kind="stable")
            flight = flight.drop(["flight_id", "is_fail", "label", "time"], axis=1)
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            
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

        # Same thing as the above loop, with `thresh_flight_ids` instead
        for fid in thresh_flight_ids:
            flight = flight_groups.get_group(fid).sort_values(by="time", kind="stable")
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            labels = flight["is_fail"].values # this line is different from the first loop
            flight = flight.drop(["flight_id", "is_fail", "label", "time"], axis=1)

            self.label_names = flight.columns.tolist()
            #scaler = StandardScaler()
            scaler = MinMaxScaler()
            scaler.fit(flight)

            scaled = scaler.transform(flight) # scale 
            scaled_tensor = torch.tensor(scaled, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # Create input-output pairs
            X_seqs, Y_seqs, labels = self.create_sequences(scaled_tensor, labels_tensor)
            for i in range(1, X_seqs.shape[0]):
                assert(X_seqs[i][-1].equal(Y_seqs[i-1]))
            
            self.thresh_X.append(X_seqs)
            self.thresh_Y.append(Y_seqs)
            self.thresh_labels.append(labels)

         # Same thing as the above loop, with `test_flight_ids` instead
        for fid in thresh_flight_ids:
            flight = flight_groups.get_group(fid).sort_values(by="time", kind="stable")
            flight = flight.iloc[:len(flight) - (len(flight) % self.num_steps)]
            labels = flight["is_fail"].values # this line is different from the first loop
            flight = flight.drop(["flight_id", "is_fail", "label", "time"], axis=1)

            self.label_names = flight.columns.tolist()
            #scaler = StandardScaler()
            scaler = MinMaxScaler()
            scaler.fit(flight)

            scaled = scaler.transform(flight) # scale 
            scaled_tensor = torch.tensor(scaled, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # Create input-output pairs
            X_seqs, Y_seqs, labels = self.create_sequences(scaled_tensor, labels_tensor)
            for i in range(1, X_seqs.shape[0]):
                assert(X_seqs[i][-1].equal(Y_seqs[i-1]))
            
            self.test_X.append(X_seqs)
            self.test_Y.append(Y_seqs)
            self.test_labels.append(labels)
            
        # After processing all flights, concatenate along the batch dimension
        self.thresh_X = torch.cat(self.thresh_X, dim=0)  # (total_sequences, num_steps - 1, num_features)
        self.thresh_Y = torch.cat(self.thresh_Y, dim=0)  # (total_sequences, num_features)
        self.thresh_labels = torch.cat(self.thresh_labels, dim=0) # (total_sequences, 1)
        self.test_X = torch.cat(self.test_X, dim=0)  # (total_sequences, num_steps - 1, num_features)
        self.test_Y = torch.cat(self.test_Y, dim=0)  # (total_sequences, num_features)
        self.test_labels = torch.cat(self.test_labels, dim=0) # (total_sequences, 1)

        # shape of X: (number of sequences, num_steps, # of features of the raw data) 
        # last dim is the number of different features (e.g. pitch, roll, etc) that each data point has
        # X is input, Y is label.
        # Y is the data point after X.
        print("thresh_X's shape: ", self.thresh_X.shape)
        print("thresh_Y's shape: ", self.thresh_Y.shape)
        print("test_X's shape: ", self.test_X.shape)
        print("test_Y's shape: ", self.test_Y.shape)
        
        end = time.time()
        print(f"Processing Time: {end - start}")

    def create_sequences(self, flight: torch.Tensor, labels: torch.Tensor = None):
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
        if labels is None:
            return X, Y
        else:
            return X, Y, labels[self.num_steps:]
        

        # broadcast then add. Shape of X_indices: (num_seqs, num_steps)
        
    # def get_dataloader(self, X, Y, is_train=False):
    #     if is_train:
    #         return self.get_tensorloader([self.train_X, self.train_Y], is_train, slice(0, None))
    #     else:
    #         return self.get_tensorloader([self.val_X, self.val_Y], is_train, slice(0, None))


class Trainer(d2l.Trainer):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        super().__init__(max_epochs)
        self.train_losses = []
        self.val_losses = []
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.thresh_dataloader = data.thresh_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 15, 30],gamma = 0.1)
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            self.scheduler.step()
            plt.savefig("./losses_without_labels.png")
            
        self.find_best_threshold()
        self.evaluate_thresholds()      

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

    def find_best_threshold(self):
        self.model.eval()
        
        max_errors_list, labels_list = [], []
        for batch in self.thresh_dataloader:
            X, Y, labels = self.prepare_batch(batch)
            Y_hat = self.model(X)
            # shape of errors: (batch_size, num_features). 
            # Didn't use the self.model.loss function here as it will use a mean reduction over all features. 
            errors = (Y_hat - Y) ** 2
            max_errors = errors.max(dim=1)[0] # for each batch, get the error of the feature that produces the greatest error
            max_errors_list.append(max_errors.detach().cpu().numpy())
            labels_list.append(labels.cpu().numpy())

        thresh_max_errors = np.concatenate(max_errors_list)
        thresh_labels = np.concatenate(labels_list)
        precision, recall, self.thresholds = precision_recall_curve(thresh_labels, thresh_max_errors, pos_label=1)
        f1_scores = 2*(precision*recall) / (precision + recall + 1e-9)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = self.thresholds[best_threshold_idx]
        best_f1 = f1_scores[best_threshold_idx]

        self.best_val_threshold_idx = best_threshold_idx
        self.best_threshold = best_threshold
        self.best_val_f1 = best_f1

        anomalies = thresh_max_errors > best_threshold
        tp = np.sum(anomalies & (thresh_labels == 1))
        fp = np.sum(anomalies & (thresh_labels == 0))
        tn = np.sum(~anomalies & (thresh_labels == 0))
        fn = np.sum(~anomalies & (thresh_labels == 1))
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        
        print(f"Optimal Threshold (Thresholding Set): {best_threshold:.6f}")
        print(f"Thresholding Set Metrics: Precision: {prec:.6f}, Recall: {rec:.6f}, F1: {best_f1:.6f}, Accuracy: {acc:.6f}")

    def evaluate_thresholds(self):
        """
        Find reconstruction errors for the test set. Then for each threshold in self.thresholds,
        compute all the metrics against that threshold and graph them
        """
        if self.best_val_f1 is None:
            raise "Please run find_best_threshold() first!"
        
        self.model.eval()
        max_errors_list, labels_list = [], []
        for batch in self.test_dataloader:
            X, Y, labels = self.prepare_batch(batch)
            Y_hat = self.model(X)
            # shape of errors: (batch_size, num_features). 
            # Didn't use the self.model.loss function here as it will use a mean reduction over all features. 
            errors = (Y_hat - Y) ** 2
            max_errors = errors.max(dim=1)[0] # for each batch, get the error of the feature that produces the greatest error
            max_errors_list.append(max_errors.detach().cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
        test_max_errors = np.concatenate(max_errors_list)
        test_labels = np.concatenate(labels_list)
        test_precision = []
        test_recall = []
        test_f1 = []
        test_accuracy = []
        for thresh in self.thresholds:
            anomalies = test_max_errors > thresh
            tp = np.sum(anomalies & (test_labels == 1))
            fp = np.sum(anomalies & (test_labels == 0))
            tn = np.sum(~anomalies & (test_labels == 0))
            fn = np.sum(~anomalies & (test_labels == 1))
            prec = tp / (tp + fp) if tp + fp > 0 else 0
            rec = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec + 1e-9) if prec + rec > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
            test_precision.append(prec)
            test_recall.append(rec)
            test_f1.append(f1)
            test_accuracy.append(acc)
            
        best_test_idx = np.argmax(test_f1)
        best_test_threshold = self.thresholds[best_test_idx]
        print(f"Best Test Set Threshold: {best_test_threshold:.6f}")
        print(f"Test Set Metrics at Best Threshold: Precision: {test_precision[best_test_idx]:.6f}, "
              f"Recall: {test_recall[best_test_idx]:.6f}, F1: {test_f1[best_test_idx]:.6f}, "
              f"Accuracy: {test_accuracy[best_test_idx]:.6f}")
        
        # Visualize metrics vs. thresholds
        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, test_precision, label="Precision", marker='.')
        plt.plot(self.thresholds, test_recall, label="Recall", marker='.')
        plt.plot(self.thresholds, test_f1, label="F1 Score", marker='.')
        plt.plot(self.thresholds, test_accuracy, label="Accuracy", marker='.')
        plt.axvline(x=self.best_threshold, color='red', linestyle='--', label=f"Optimal Threshold ({self.best_threshold:.6f})")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Metrics vs. Threshold on Test Set")
        plt.legend()
        plt.grid(True)
        plt.savefig("./metrics/metrics_vs_threshold_without_labels.png")
        plt.close()

        # Precision-Recall Curve
        test_precision, test_recall, _ = precision_recall_curve(test_labels, test_max_errors, pos_label=1)
        plt.figure(figsize=(8, 6))
        plt.plot(test_recall, test_precision, label="Precision-Recall Curve")
        plt.scatter(test_recall[best_test_idx], test_precision[best_test_idx], color='red',
                    label=f"Best Threshold ({best_test_threshold:.6f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Test Set)")
        plt.legend()
        plt.grid(True)
        plt.savefig("./metrics/precision_recall_curve_test_without_labels.png")
        plt.close()
            


# In[ ]:


num_features = data.train_X.shape[2] 
num_hiddens, num_hiddens_ffn, num_blks, num_hiddens_latent = 128, 128, 4, 64
num_lstm_layers = 2
num_heads, dropout, bias, lr = 4, 0.2, True, 1e-3
rnn = RNN(num_features, num_hiddens, num_hiddens_ffn, num_hiddens_latent, 
             num_lstm_layers, dropout, bias, lr)
model = MainModel(rnn, num_features, lr)

trainer = Trainer(max_epochs=70, num_gpus=8)
trainer.fit(model, data)

