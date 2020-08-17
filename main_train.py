import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import time
from torch.utils.data import DataLoader
from torch import nn
from utils import Regularization
from data import RML2016b
torch.__version__

# Parameter definition
BATCH_SIZE = 1024
EPOCHS = 1000
weight_decay = 10**-4.5
R_TYPE = 4  # 1 for L1 norm, 2 for L2 norm, 3 for GL norm, 4 for SGL norm, 5 for TGL norm
LEARN_RATE = 0.001
MODEL_PATH = 'checkpoints/TL_R01_01_00_stft.pth'
Is_load_model = 0  # 0 for no load, 1 for load trained model from MODEL_PATH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # try to use GPU
print(DEVICE)

#  Download the dataset
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


#  Initialization
def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.1)
        m.bias.data.zero_()


#  Model and optimizer
model = models.DnnNet0()
model.apply(weight_init)
model = model.to(DEVICE)
if Is_load_model == 1:
    model.load(MODEL_PATH)

# loss_fun = torch.nn.MSELoss()
# loss_fun = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 800], gamma=0.1)

#  Regularization
reg_loss = 0
if weight_decay > 0:
    reg_loss = Regularization(model, weight_decay, p=R_TYPE).to(DEVICE)
else:
    print("no regularization")


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if weight_decay > 0:
            loss = loss + reg_loss(model)
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
    lr_now = optimizer.state_dict()['param_groups'][0]['lr']
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), L-rate:{:}, time:{:.2f}s\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), lr_now, time_epoch-time_epoch0))


time_begin = time.time()
for epoch in range(1, EPOCHS + 1):
    time_epoch0 = time.time()
    train(model, DEVICE, train_loader, optimizer, epoch)
    scheduler.step()
    test(model, DEVICE, test_loader)

model.save('checkpoints/SGL-NN_1000.pth')
