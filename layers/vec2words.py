#Although W2Vec is a tiny NN in this project we treat it as a layer
import torch
import torch.nn.functional as F 
import linear.py
import ntlk
from ntlk.corpus import stopwords

class W2VecLayer(object):

    def __init__(self):
        self.n_embeddings = None
        self.V = None 
        self.context_size = None
        
        l1 = None
        l2 = None
        self.words = []
        self.word_idx = {}

    def init_params(self, V , n_embeddings, context_size, data):

        self.V = V
        self.n_embeddings = n_embeddings
        self.context_size = context_size
        self.words = data.words

        for i in range(len(self.words)): 
            self.words_idx[data[i]] = i
        self.l1 = LinearLayer(V, n_embeddings)
        self.l2 = LinearLayer(n_embeddings, V)

    def forward(self, X):
        X = torch.tensor(X,dtype=torch.float32).reshape(-1,1)
        self.cache = X
        h = self.l1.forward(X)
        u = self.l2.forward(h)

        self.y_pred = F.softmax(u, dim=0)

        return self.y_pred
    
    def backward(self,dupsteam_h, dupsteam_u):
        X = self.cache
        dW1 = self.l1.backward(dupsteam_h)
        dW = self.l2.backward(dupsteam_u)
        self.grad_W1 = dW1
        self.grad_W = dW

        return [dW1,dW]

