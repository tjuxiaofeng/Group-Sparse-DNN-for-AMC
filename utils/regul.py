import torch
import numpy


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=0):
        '''
        :param model: model
        :param weight_decay: Regularization parameters
        '''
        super(Regularization, self).__init__()
        if p == 1:
            print("Using L1-NN penalty")
        elif p == 2:
            print("Using L2-NN penalty")
        elif p == 3:
            print("Using G-L1-NN penalty")
        elif p == 4:
            print("Using SG-L1-NN penalty")
        elif p == 5:
            print("Using TLDNN penalty")
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        Specify operating mode
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        Get a list of model weights
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        Compute tensor norm
        :param weight_list:
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l_reg1 = 0
            l_reg2 = 0
            if p == 1:
                l_reg1 = torch.norm(w, p=p)
            elif p == 2:
                l_reg1 = torch.norm(w, p=p)
            elif p == 3:
                temp = torch.norm(w, p=2, dim=0)
                l_reg1 = torch.norm(temp, p=1)
            elif p == 4:
                temp = torch.norm(w, p=2, dim=0)
                l_reg1 = numpy.sqrt(w.size(0))*torch.norm(temp, p=1)
                l_reg2 = torch.norm(w, p=1)
            elif p == 5:
                temp2 = torch.norm(w, p=2, dim=0)
                temp1 = torch.norm(w, p=2, dim=1)
                l_reg2 = torch.norm(temp2, p=1)
                l_reg1 = torch.norm(temp1, p=1)

            reg_loss = reg_loss + l_reg2 + l_reg1

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        Print the weight information
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
