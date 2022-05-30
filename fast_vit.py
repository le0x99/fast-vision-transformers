import torch
import torch.nn as nn
import torch.nn.functional as F


from base_models import *

class Encoder(nn.Module):
    def __init__(self, N, G):
        super().__init__()
        self.T = G ** 2
        self.N2 = N // G
        self.emb_dim = 3*(N // G)**2
        
    @torch.no_grad()
    def forward(self, X):
        N2 = self.N2
        X = X.unfold(2, N2, N2).unfold(3, N2, N2)
        X = X.flatten(2,3)
        X = X.transpose(1,2)
        X = X.flatten(2,4)
        return X
    

class PatchDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.sampler = torch.distributions.binomial.Binomial(1, torch.tensor([p]))
    @torch.no_grad()
    def forward(self, X):
        B, T, E = X.size()
        idx = self.sampler.sample(sample_shape=(B * T,)).bool().ravel()
        X.view(B * T, E)[idx] = 0.
        return X

    

class FastVisionTransformer(nn.Module):
    def __init__(self, IMAGE_N, N_CLASSES, G, PDO, DO, N_HEADS,
                 MLP_HEAD=True, MLP_MULT=2, FF_MULT=2, N_BLOCKS=4,AUTO_ENCODE=True, EMB_DIM=None):
        super().__init__()
        self.G = G
        self.T = G ** 2
        self.pdo = PatchDropout(PDO) if PDO > 0. else nn.Identity()
        self.pdo_p = PDO
        self.encoder = Encoder(IMAGE_N, G) if AUTO_ENCODE else nn.Identity()
        self.patch_emb_dim = self.encoder.emb_dim if AUTO_ENCODE else EMB_DIM
        self.head = nn.Sequential(
            nn.Linear(self.patch_emb_dim, self.patch_emb_dim * MLP_MULT),
            nn.ReLU(inplace=True),
            nn.Linear(self.patch_emb_dim * MLP_MULT, N_CLASSES) ) if MLP_HEAD else nn.Linear(self.patch_emb_dim, N_CLASSES)
        self.contextualize = [Transformer(self.patch_emb_dim,
                                                  N_HEADS,
                                                  DO,
                                                  FF_MULT) for _ in range(N_BLOCKS)]
        self.contextualize = nn.Sequential(*self.contextualize)
        self.pos_emb = nn.Embedding(self.T, self.patch_emb_dim)
        
    def total_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def forward(self, X):
        X = self.encoder(X)
        B, T, E = X.size()
        positions = torch.arange(T, device=self.device)
        positions = self.pos_emb(positions)[None, :, :].expand(B, T, E)
        X += positions
        X = self.pdo(X)
        X = self.contextualize(X)
        Y = X.mean(1)
        Y = self.head(Y)
        return Y
        
    def eval_mode(self):
        self.pdo = nn.Identity()
        self.eval()
    
    def train_mode(self):
        self.pdo = PatchDropout(self.pdo_p)
        self.train()

