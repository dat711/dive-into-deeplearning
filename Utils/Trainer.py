import torch
from .Synthetic_data_generator import data_loader


class CustomCaseTrainer:
    def __init__(self, model,loss, optim):
        self.model = model
        self.loss = loss
        self.optim = optim

    def __call__(self,features,labels,batchsize,lr = 1e-2, epochs = 10,logging = False):
        for epoch in range(epochs):
            for X, y in data_loader(batchsize,features,labels):
                l = self.loss(self.model.forward(X), y)
                # Compute gradient on `l` with respect to [`w`, `b`]
                l.sum().backward()
                self.optim([self.model.w,self.model.b], lr, batchsize) # Update parameters using their gradient
            if logging:
                with torch.no_grad():
                    train_l = self.loss(self.model.forward(features), labels)
                    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        return self.model

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    print(params)
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def sqr_loss(y_hat, y):
    """
    :param X: Features
    :param y: true label
    :return: square loss of the model output
    """

    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2