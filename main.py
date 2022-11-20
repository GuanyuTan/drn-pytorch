import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DRN
import time
from drn import Ldjs
from config.config import get_cfg_defaults
from torch.utils.data import TensorDataset, DataLoader
import logging
from torch.utils.tensorboard.writer import SummaryWriter

# TODO Create experiment directory
# TODO Write logging to /path_to_directory/train.log
# logging.basicConfig(level=logging.INFO, format=)
device = "cuda" if torch.cuda.is_available() else "cpu"
config = get_cfg_defaults()
config.freeze()
q, hidden_q = config.MODEL.Q, config.MODEL.HIDDEN_Q
batch_size = config.TRAINING.BATCH_SIZE
epochs = config.TRAINING.N_EPOCH
np.random.seed()
torch.random.seed()
data_dir = config.DATASET.DATA_DIR
train_x = np.loadtxt(data_dir+'train_x.dat')
train_y = np.loadtxt(data_dir+'train_y.dat')
test_x = np.loadtxt(data_dir+'test_x.dat')
test_y = np.loadtxt(data_dir+'test_y.dat')
train_x = train_x[:config.TRAINING.N_TRAIN].reshape((-1, 1, q))
train_y = train_y[:config.TRAINING.N_TRAIN].reshape((-1, 1, q))
test_x = test_x[:config.TRAINING.N_TEST].reshape((-1, 1, q))
test_y = test_y[:config.TRAINING.N_TEST].reshape((-1, 1, q))
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


model = DRN(cfg=config)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.TRAINING.LEARNING_RATE)
writer = SummaryWriter()
for epoch in range(epochs):
    running_mse = []
    running_djsl = []
    for i, data in enumerate(train_dataloader, 0):
        x, labels = data
        optimizer.zero_grad()
        y_pred = model(x)
        Djs_loss = Ldjs(labels, y_pred)
        Djs_loss.backward()
        with torch.no_grad():
            Mse_loss = mse_loss(labels, y_pred)
            running_mse.append(Mse_loss.item())
        optimizer.step()
        running_djsl.append(Djs_loss.item())
    mse = np.mean(running_mse)
    djsl = np.mean(running_djsl)
    writer.add_scalar('Train MSE Loss', mse, epoch)
    writer.add_scalar('Train Shanon-Jenson Divergence Loss', djsl, epoch)
    with torch.no_grad():
        mse = []
        djsl = []
        for data in test_dataloader:
            x, labels = data
            y_pred = model(x)
            Djs_loss = Ldjs(labels, y_pred)
            Mse_loss = mse_loss(labels, y_pred)
            mse.append(Mse_loss.item())
            djsl.append(Djs_loss.item())
        mse = np.mean(mse)
        djsl = np.mean(djsl)
        writer.add_scalar('Test MSE Loss', mse, epoch)
        writer.add_scalar('Test Shanon-Jenson Divergence Loss', djsl, epoch)
writer.flush()
            
        





