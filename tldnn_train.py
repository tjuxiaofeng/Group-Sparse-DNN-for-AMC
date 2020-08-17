import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import time
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from utils import Regularization
from data import RML2016b
torch.__version__

# Parameter definition
BATCH_SIZE = 1024
EPOCHS = 1000
weight_decay = 10**-4.5
R_TYPE = 5  # 1 for L1 norm, 2 for L2 norm, 3 for GL norm, 4 for SGL norm, 5 for TGL norm
LEARN_RATE = 0.001
MODEL1_PATH = 'checkpoints/LDNN_1000.pth'
MODEL2_PATH = 'checkpoints/ConfNet.pth'
Is_load_main_model = 1  # 0 for no load, 1 for load trained model from MODEL1_PATH
Is_load_conf_model = 0  # 0 for no load, 1 for load trained model from MODEL2_PATH
Is_train_main_model = 0  # 0 for use trained main_model form MODEL1_PATH, 1 for trained main_model from scratch
Bh = 0.75
Bl = 0.20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # try to use GPU
print(DEVICE)


#  Initialization
def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.1)
        m.bias.data.zero_()


def train(model, device, train_loader, optimizer, epoch, reg_loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if weight_decay > 0:
            loss = loss + reg_loss(model)
        else:
            print("no regularization")
        loss.backward()
        optimizer.step()
        time_now = time.time()
        if(batch_idx+1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime={:.2f}s'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time_now-time_begin))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    time_epoch = time.time()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%),time:{:.2f}s\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), time_epoch-time_epoch0))


#  Define confusion matrix calculation function
def confusion_matrix(model, conf_matrix):
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            predict = torch.argmax(output, 1)
            correct += predict.eq(target.view_as(predict)).sum().item()
            c = predict[0]
            for t, p in zip(target, predict):
                conf_matrix[t, p] += 1
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    return conf_matrix


def converged_net(data):
    out_main = model_main(data)
    predict_main = torch.argmax(out_main, 1)
    predict_fine = predict_main
    for i in range(0, predict_fine.size(0) - 1):
        if predict_main[i] in conf_num:
            out_conf = model_conf(data)
            predict_conf = torch.argmax(out_conf, 1)
            predict_fine[i] = conf_num[int(predict_conf[i])]
    return predict_fine


# Step1: train the main network
train_dataset = RML2016b(train=True)
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2)
test_dataset = RML2016b(train=False)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2)

#  Model and optimizer
model_main = models.DnnNet0(len(test_dataset.classes))
model_main.apply(weight_init)
model_main = model_main.to(DEVICE)
if Is_load_main_model == 1:
    model_main.load(MODEL1_PATH)

#  Regularization
reg_loss_main = 0
if weight_decay > 0:
    reg_loss_main = Regularization(model_main, weight_decay, p=R_TYPE).to(DEVICE)
else:
    print("no regularization")

optimizer_main = optim.Adam(model_main.parameters(), lr=LEARN_RATE)
scheduler_main = optim.lr_scheduler.MultiStepLR(optimizer_main, milestones=[500, 800], gamma=0.1)


if Is_train_main_model == 1:
    print("Step1: train the main network")
    time_begin = time.time()
    for epoch in range(1, EPOCHS + 1):
        time_epoch0 = time.time()
        train(model_main, DEVICE, train_loader, optimizer_main, epoch, reg_loss_main)
        scheduler_main.step()
        test(model_main, DEVICE, test_loader)
else:
    print("Step1: use the trained network as the main network")

model_main.save('checkpoints/MNet_LDNN.pth')

# Converged network: describe the confusion matrix from the trained main_model
print('Converged network: the training phase')
classes = test_dataset.classes
conf_matrix = np.zeros((len(classes), len(classes)))
conf_matrix = confusion_matrix(model_main, conf_matrix)
confusion_num = []
confusion_class = []
for i in range(0, len(classes)):
    for j in range(0, len(classes)):
        if i == j:
            if conf_matrix[i, j] < Bh:
                confusion_num.append(i)
                confusion_class.append(classes[i])
        if i != j:
            if conf_matrix[i, j] > Bl:
                confusion_num.append(i)
                confusion_class.append(classes[i])
                confusion_num.append(j)
                confusion_class.append(classes[j])
conf_num = np.unique(confusion_num)
conf_num = conf_num.tolist()
conf_class = np.unique(confusion_class)
conf_class = conf_class.tolist()
print("The confusion classes:", conf_class)

# Step2: fine training
print('Step2: fine training')
fine_train_dataset = RML2016b(train=True, fine_train=True, conf_class=conf_class)
fine_train_loader = DataLoader(fine_train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2)
fine_test_dataset = RML2016b(train=False, fine_train=True, conf_class=conf_class)
fine_test_loader = DataLoader(fine_test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2)

model_conf = models.DnnNet0(len(conf_num))
# model_conf.apply(weight_init)
model_conf = model_main.to(DEVICE)
if Is_load_conf_model == 1:
    model_main.load(MODEL2_PATH)

#  Regularization
reg_loss_conf = 0
if weight_decay > 0:
    reg_loss_conf = Regularization(model_conf, weight_decay, p=R_TYPE).to(DEVICE)
else:
    print("no regularization")

optimizer_conf = optim.Adam(model_conf.parameters(), lr=LEARN_RATE)
scheduler_conf = optim.lr_scheduler.MultiStepLR(optimizer_conf, milestones=[500, 800], gamma=0.1)

time_begin = time.time()
for epoch in range(1, EPOCHS + 1):
    time_epoch0 = time.time()
    train(model_conf, DEVICE, fine_train_loader, optimizer_conf, epoch, reg_loss_conf)
    scheduler_conf.step()
    test(model_conf, DEVICE, fine_test_loader)

model_conf.save('checkpoints/CNet_LDNN.pth')

# Converged network
print('Converged network: the testing phase')
with torch.no_grad():
    correct = 0
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        predict = converged_net(data)
        correct += predict.eq(target.view_as(predict)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))