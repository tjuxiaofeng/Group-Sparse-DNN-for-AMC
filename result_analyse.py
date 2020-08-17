import torch
import matplotlib.pyplot as plt
import numpy as np
import models
import itertools
from torch.utils.data import DataLoader
from tabulate import tabulate
from data import RML2016b
torch.__version__


BATCH_SIZE = 1024
MODEL_PATH = 'checkpoints/SGL-NN_1000.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # try to use GPU
print(DEVICE)

test_dataset = RML2016b(train=False)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2)

model = models.DnnNet0()
model = model.to(DEVICE)
model.load(MODEL_PATH)


#  Define confusion matrix calculation function
def confusion_matrix(conf_matrix):
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
        print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return conf_matrix


#  Define confusion matrix drawing function
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


def get_weight(model_weight):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            model_weight.append(weight)
    return model_weight


#  Define the network sparsity acquisition function
def model_sparse():
    weight_list = []
    weight_list = get_weight(weight_list)
    count = 0
    sparsity_weights = []
    sparsity_neurons = []
    parameter_nums = []
    sum_parameters = 0
    for name, w in weight_list:
        w = w.data.cpu().numpy()
        parameters = [w.round(decimals=3).ravel().nonzero()[0].shape[0]]
        sparsity = [1 - (w.round(decimals=3).ravel().nonzero()[0].shape[0] / w.size)]
        neurons = [w.round(decimals=3).sum(axis=0).nonzero()[0].shape[0]]
        sparsity_weights.append(sparsity)
        sparsity_neurons.append(neurons)
        parameter_nums.append(parameters)
        sum_parameters = sum_parameters + np.array(parameters)
        count = count + 1
    print('# generator parameters:', sum_parameters)
    names = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
    table_data = [['Sparsity [%]'] + np.mean(sparsity_weights, axis=1).round(decimals=4).tolist(),
                  ['Parameters'] + np.mean(parameter_nums, axis=1).round(decimals=4).tolist(),
                  ['Neurons'] + np.mean(sparsity_neurons, axis=1).round(decimals=4).tolist()]
    print(tabulate(table_data, headers=['Level']+names, tablefmt='fancy_grid'))


#  print the result
model_sparse()
classes = test_dataset.classes
conf_matrix = np.zeros((len(classes), len(classes)))
conf_matrix = confusion_matrix(conf_matrix)
plot_confusion_matrix(conf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')

