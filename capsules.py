import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def squash(vector, dim=-1):
    """Activation function for capsule"""
    sj2 = (vector**2).sum(-1, keepdim=True)
    sj = sj2 ** 0.5
    return(
        sj2/(1.0+sj2)*vector/sj
    )

def find_probability(t):
    """Compute softmaxed length of vector"""
    return(F.softmax((t.squeeze(-1)**2).sum(-1)*0.5, dim=-1))

def get_a_capsule(u,n,k):
    """Return the contents of a capsule"""
    return(u.squeeze(-1)[n,k,:])


class PrimaryCapsuleLayer(nn.Module):
    
    def make_conv(self):
        """Build a primary capsule which is just 2d-convolution"""
        return(
            nn.Conv2d(
                self.in_ch, self.out_ch, 
                kernel_size=self.kernel_size, 
                stride=self.stride, padding=0
            )
        )
    
    def __init__(
        self, n_capsules=8, n_routes=32*6*6, in_ch=256, 
        out_ch=32, kernel_size=9, stride=2, cuda=True
    ):
        super(PrimaryCapsuleLayer, self).__init__()
        self.n_capsules = n_capsules
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_capsules = n_capsules
        self.n_routes = n_routes
        self.capsules = nn.ModuleList([])
        for a in range(self.n_capsules):
            self.capsules.append(self.make_conv())
            
    def forward(self, x):
        """Compute outputs of capsules, reshape and squash"""
        out = torch.stack([a(x) for a in self.capsules], dim=1).view(
            x.size(0), self.n_routes, -1
        )
        return(squash(out))
    
    
class SecondaryCapsuleLayer(nn.Module):
    
    def __init__(self, n_capsules=10, n_iter=3, n_routes=32*6*6, in_ch=8, out_ch=16, cuda=True):
        super(SecondaryCapsuleLayer, self).__init__()
        self.n_capsules = n_capsules
        self.n_iter = n_iter
        self.n_routes = n_routes
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.W = nn.Parameter(
            torch.randn(1, self.n_routes, self.n_capsules, self.out_ch, self.in_ch)
        )
        self.cuda = cuda
    
    def forward(self, x):
        """Perform routing by agreement"""
        batch_size = x.size(0)
        u_hat = torch.matmul(
            torch.cat([self.W]*x.size(0), dim=0),
            torch.stack([x]*self.n_capsules, dim=2).unsqueeze(4)
        )
        b_ij = Variable(torch.zeros(1, self.n_routes, self.n_capsules, 1))
        b_ij = b_ij.cuda() if self.cuda else b_ij
        for a in range(self.n_iter):
            c_ij = torch.cat(
                [F.softmax(b_ij, dim=1)]*x.size(0), dim=0
            ).unsqueeze(4)
            s_j = (c_ij*u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)
            if a < self.n_iter - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j]*self.n_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
        return(v_j.squeeze(1))
    
    
class RegularizingDecoder(nn.Module):
    
    def __init__(self, dims=[16,512,1024,784]):
        super(RegularizingDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(inplace=True),
            nn.Linear(dims[2], dims[3]),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Feed-forward reconstructor"""
        return(self.decoder(x))
    

class CapsuleLoss(nn.Module):
    
    def __init__(self, ms=(0.9, 0.1), l=0.5, adjustment=0.0005, n_classes=10, cuda=True):
        super(CapsuleLoss, self).__init__()
        self.ms = ms
        self.l = l
        self.m_p = ms[0]
        self.m_n = ms[1]
        self.a = adjustment
        self.n = n_classes
        self.cudap = cuda
        self.reconstruction_loss = nn.MSELoss()
        self.ones = torch.eye(self.n)
        if self.cudap:
            self.ones = self.ones.cuda()
        
    def margin_loss(self, x, mask):
        a = mask*F.relu(self.m_p-find_probability(x))
        b = self.l*(1-mask)*F.relu(find_probability(x)-self.m_n)
        return((a+b).sum(dim=1).mean())
        
    def forward(self, internal, inp, real_classes, reconstructions):
        """Compute margin and reconstruction losses"""
        mask = self.ones.index_select(0, real_classes)
        ml = self.margin_loss(internal, mask)
        inp = inp.detach()
        reconstructions = reconstructions.detach()
        rl = self.reconstruction_loss(
            inp, reconstructions
        )
        return(ml+self.a*rl)