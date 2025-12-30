# -*- coding: utf-8 -*-
# model.py
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, dim=10, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)
