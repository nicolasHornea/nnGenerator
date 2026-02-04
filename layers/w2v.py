#Although W2Vec is a tiny NN in this project we treat it as a layer
import torch
import torch.nn.functional as F 
from linear import LinearLayer
class W2VecLayer(object):

    def __init__(self):
        self.n_embeddings = None
        self.V = None 
        self.context_size = None
        
        self.l1 = None
        self.l2 = None

    def init_params(self, V , n_embeddings, context_size):

        self.V = V
        self.n_embeddings = n_embeddings
        self.context_size = context_size
        self.l1 = LinearLayer(V, n_embeddings)
        self.l2 = LinearLayer(n_embeddings, V)

    def forward(self, X):
        X = torch.tensor(X,dtype=torch.float32).reshape(1,-1)
        self.cache = X
        h = self.l1.forward(X)
        u = self.l2.forward(h)

        self.y_pred = F.softmax(u, dim=0)

        return self.y_pred
    
    def backward(self,dupsteam_u):
        X = self.cache
        dW1 = self.l2.backward(dupsteam_u)
        dW = self.l1.backward(dW1)
        self.grad_W1 = dW1
        self.grad_W = dW

        return [dW1,dW]

