import torch
from Synthetic_data_generator import data_loader

# custom Trainer

class CustomCaseTrainer:
    def __init__(self, model,loss, optim):
        self.model = model
        self.loss = loss
        self.optim = optim

    def __call__(self,data_iter, epochs = 10):
        for epoch in range(epochs):
            for X, y in data_iter:
                l = self.loss(self.model(X), y)
                self.optim.zero_grad()
                l.backward()
                self.optim.step()
        return self.model

# optimization function
def sgd(params,lr,batch_size):
  with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()