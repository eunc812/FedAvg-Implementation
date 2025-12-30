# -*- coding: utf-8 -*-
# client.py
import torch
import copy

class Client:
    def __init__(self, model, dataloader, lr=0.01, device="cpu"):
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_weights(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def local_update(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                x = x.view(x.size(0), -1)
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

        return copy.deepcopy(self.model.state_dict()), len(self.dataloader.dataset)
