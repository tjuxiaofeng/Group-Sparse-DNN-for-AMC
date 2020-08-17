from torch import nn
from .BasicModule import BasicModule
import torch.nn.functional as F


class DnnNet0(BasicModule):
    def __init__(self, output_len=10):
        self.output_len = output_len
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256, 500),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(500, 400),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(400, 300),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(300, 200),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(200, 100),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(100, self.output_len),
        )

    def forward(self, x):
        x = x.reshape(-1, 256)
        out = self.classifier(x)
        out = F.log_softmax(out, dim=1)
        return out
