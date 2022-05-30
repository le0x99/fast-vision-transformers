import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
import torchvision

class Attention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads

        self.W_k    = nn.Linear(emb, emb, bias=False)
        self.W_q = nn.Linear(emb, emb, bias=False)
        self.W_v  = nn.Linear(emb, emb, bias=False)
        self.W_u = nn.Linear(emb, emb)

    def forward(self, X):

        b, t, e = X.size()
        h = self.heads
        # chunksize of e, i.e. head dim
        s = e // h
        # query, key, value model
        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)
        # split
        K = K.view(b, t, h, s)
        Q = Q.view(b, t, h, s)
        V = V.view(b, t, h, s)
        # prepare for dot product and scale (pbloem)
        K = K.transpose(1,2).contiguous().view(b * h, t, s) / (e ** (1/4))
        Q = Q.transpose(1,2).contiguous().view(b * h, t, s) / (e ** (1/4))
        V = V.transpose(1,2).contiguous().view(b * h, t, s) / (e ** (1/4))

        W = Q@K.transpose(1,2)
        W = F.softmax(W, dim=2)

        #assert W.size() == (b*h, t, t)

        Y = W@V
        Y = Y.view(b, h, t, s)

        # re-arange and unify heads 
        Y = Y.transpose(1, 2).contiguous().view(b, t, s * h)
        Y = self.W_u(Y)
        return Y
    
class Transformer(nn.Module):

    def __init__(self, emb=2048, heads=32,dropout=0.25,ff_hidden_mult=2):
        super().__init__()
        self.attention = Attention(emb, heads=heads)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x
    


