#Although W2Vec is a tiny NN in this project we treat it as a layer
import torch
import torch.nn.functional as F 

class W2VecLayer(object):

    def __init__(self):
        self.n_embeddings = None
        self.V = None 
        self.context_size = None
        self.W = None
        self.W1 = None

        self.words = []
        self.word_idx = {}

    def init_params(self, V , n_embeddings, context_size, data):

        self.V = V
        self.n_embeddings = n_embeddings
        self.context_size = context_size
        self.words = data.words
        self.word_idx = data.word_idx

        self.W = torch.randn(V, n_embeddings) * 0.01
        self.W1 = torch.randn(n_embeddings, V) * 0.01

    def forward(self, X):
        X = torch.tensor(X,dtype=torch.float32).reshape(-1,1)
        self.cache = X
        self.h = torch.mm(self.W.T, X)
        self.u = torch.mm(self.W1.T,self.h)
        self.y_pred = F.softmax(self.u, dim=0)

        return self.y_pred
    
    def backward(self,dupsteam_h, dupsteam_u):
        X = self.cache
        dW1 = torch.mm(self.h, dupsteam_u.T)
        dW = torch.mm(X, dupsteam_h.T)

        self.grad_W1 = dW1
        self.grad_W = dW

        return [dW1,dW]

