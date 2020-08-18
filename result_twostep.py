import torch
import matplotlib.pyplot as plt
import numpy as np
import models
import itertools
from torch.utils.data import DataLoader
from tabulate import tabulate
from data import RML2016b
torch.__version__

# 定义参数
BATCH_SIZE = 1024
MODEL_PATH1 = 'checkpoints/LDNN1.pth'
MODEL_PATH2 = 'checkpoints/LDNN2.pth'
Bh = 0.75
Bl = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


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


def confusion_matrix(model, conf_matrix, converge=False):
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            if converge:
                predict = model(data)
            else:
                output = model(data)
                predict = torch.argmax(output, 1)
            correct += predict.eq(target.view_as(predict)).sum().item()
            for t, p in zip(target, predict):
                conf_matrix[t, p] += 1
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        if converge:
            print('\nTest set: Converged Accuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        else:
            print('\nTest set: Main_model Accuracy: {}/{} ({:.2f}%)\n'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    return conf_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : confusion matrix
        - classes : the classes
        - normalize : True:show as percentage, False:show as norm
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_weight(model_weight, model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            model_weight.append(weight)
    return model_weight


def model_sparse(model):
    weight_list = []
    weight_list = get_weight(weight_list, model)
    count = 0
    sparsity_weights = []
    sparsity_neurons = []
    parameter_nums = []
    for name, w in weight_list:
        w = w.data.cpu().numpy()
        # w = abs(w)
        parameters = [w.round(decimals=3).ravel().nonzero()[0].shape[0]]
        sparsity = [1 - (w.round(decimals=3).ravel().nonzero()[0].shape[0] / w.size)]
        neurons = [w.round(decimals=3).sum(axis=0).nonzero()[0].shape[0]]
        sparsity_weights.append(sparsity)
        sparsity_neurons.append(neurons)
        parameter_nums.append(parameters)
        count = count + 1
    names = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
    table_data = [['Sparsity [%]'] + np.mean(sparsity_weights, axis=1).round(decimals=4).tolist(),
                  ['Parameters'] + np.mean(parameter_nums, axis=1).round(decimals=4).tolist(),
                  ['Neurons'] + np.mean(sparsity_neurons, axis=1).round(decimals=4).tolist()]
    print(tabulate(table_data, headers=['Level']+names, tablefmt='fancy_grid'))


test_dataset = RML2016b(train=False)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2)

model_main = models.DnnNet0()
model_main = model_main.to(DEVICE)
model_main.load(MODEL_PATH1)

model_conf = models.DnnNet0()
model_conf = model_conf.to(DEVICE)
model_conf.load(MODEL_PATH2)


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


print("The sparse of model1")
model_sparse(model_main)
print("The sparse of model2")
model_sparse(model_conf)
classes = test_dataset.classes
conf_matrix = np.zeros((len(classes), len(classes)))
conf_matrix = confusion_matrix(converged_net, conf_matrix, converge=True)  # 计算混淆矩阵
plot_confusion_matrix(conf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')  # 绘制混淆矩阵



