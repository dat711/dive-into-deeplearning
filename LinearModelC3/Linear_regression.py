import torch

class CustomLinear:
    def __init__(self, w = None, bias = None):
        """
        :param w: weight initialize
        :param bias: bias initialize
        """
        self.w = w if w else torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        self.b = bias if bias else torch.zeros(1)

    def forward(self, X):
        """
        :param X: features
        :return: output
        """
        return torch.matmul(X, self.w) + self.b

    def sqr_loss(self,X, y):
        """
        :param X: Features
        :param y: true label
        :return: square loss of the model output
        """
        y_hat = self.forward(X)
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2



