import torch

class LinearLayer(object):

    def __init__(self, in_features, out_features, bias=True):

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = torch.Tensor(in_features, out_features)
        self.bias = torch.Tensor(out_features)
        self.init_params()

        self.cache = None
        self.grad_weight = None
        self.grad_bias = None

    def init_params(self, std=1.):
        self.weight = std * torch.randn_like(self.weight)

        if self.use_bias:
            self.bias = torch.randn_like(self.bias)
        else:
            self.bias = torch.zeros_like(self.bias)

    def forward(self, x):
        self.cache = x
        y = torch.addmm(self.bias, x, self.weight)

        return y

    def backward(self, dupstream):
        x = self.cache
        dx = torch.mm(dupstream, self.weight.T)
        self.grad_weight = torch.mm(x.T, dupstream)
        self.grad_bias = torch.sum(dupstream, dim=0)

        return dx
