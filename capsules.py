import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax


def anomaly_scores(lengths, inp, reconstruction, normal_class=0, anomaly_class=1):
    """Anomaly scores based on https://arxiv.org/pdf/1909.02755.pdf"""
    difference = lengths.T[normal_class] - lengths.T[anomaly_class]
    return(difference, np.sum((inp-reconstruction)**2, 1))
    
def normality_scores(lengths, inp, reconstruction):
    """Normality scores based on https://arxiv.org/pdf/1907.06312.pdf"""
    return(
        softmax(lengths, axis=1).max(1), 
        np.sum((inp-reconstruction)**2/np.sum(inp**2), 1)
    )

def squash(vector, dim=-1):
    """Activation function for capsule"""
    sj2 = (vector**2).sum(dim, keepdim=True)
    sj = sj2 ** 0.5
    return(
        sj2/(1.0+sj2)*vector/sj
    )

def make_y(labels, n_classes):
    masked = torch.eye(n_classes)
    masked = masked.cuda() if torch.cuda.is_available() else masked
    masked = masked.index_select(dim=0, index=labels)
    return(masked)


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
        self, n_capsules=8, in_ch=256, out_ch=32, kernel_size=9, stride=2
    ):
        super(PrimaryCapsuleLayer, self).__init__()
        self.n_capsules = n_capsules
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_capsules = n_capsules
        self.capsules = nn.ModuleList([])
        for a in range(self.n_capsules):
            self.capsules.append(self.make_conv())
            
    def forward(self, x):
        """Compute outputs of capsules, reshape and squash"""
        out = torch.cat([a(x).view(x.size(0), -1, 1) for a in self.capsules], dim=-1)
        return(squash(out))
        
    
class SecondaryCapsuleLayer(nn.Module):
    
    def __init__(self, n_capsules=10, n_iter=3, n_routes=32*6*6, in_ch=8, out_ch=16):
        super(SecondaryCapsuleLayer, self).__init__()
        self.n_capsules = n_capsules
        self.n_iter = n_iter
        self.n_routes = n_routes
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.W = nn.Parameter(
            torch.randn(self.n_capsules, self.n_routes, self.in_ch, self.out_ch)
        )
    
    def forward(self, x):
        P = x[None, :, :, None, :] @ self.W[:, None, :, :, :]
        L = torch.zeros(*P.size())
        L = L.cuda() if torch.cuda.is_available() else L
        for i in range(self.n_iter):
            probabilities = F.softmax(L, dim=2)
            out = squash((probabilities*P).sum(dim=2, keepdim=True))
            if i != self.n_iter - 1:
                L = L + (P*out).sum(dim=-1, keepdim=True)
        out = out.squeeze().transpose(1,0)
        if x.shape[0] == 1:
            out = out.reshape(1, *out.shape).transpose(2,1)
        return(out)
    
    
class RegularizingDecoder(nn.Module):
    
    def __init__(self, dims=[160,512,1024,784]):
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
    
    def __init__(
        self, ms=(0.9, 0.1), l=0.5, adjustment=0.0005, only_normals=False, 
        normal_class=0, shift=False, scale=False
    ):
        super(CapsuleLoss, self).__init__()
        self.l = l
        self.m_p = ms[0]
        self.m_n = ms[1]
        self.a = adjustment
        self.scale = scale
        self.shift = shift
        self.only_normals = only_normals
        self.normal_class = normal_class
        self.reconstruction_loss = nn.MSELoss(reduction="none")
            
    def forward(self, labels, inputs, classes, reconstructions):
        left = F.relu(self.m_p - classes, inplace=True) ** 2
        right = F.relu(classes - self.m_n, inplace=True) ** 2
        margin_loss = (labels*left + self.l*(1-labels)*right).sum()
        if self.only_normals:
            reconstructions = reconstructions[labels.argmax(1) == self.normal_class]
            inputs = inputs[labels.argmax(1) == self.normal_class]
        if self.shift:
            inputs = inputs+1
        if self.scale:
            inputs = inputs/inputs.max()
        reconstruction_loss = self.reconstruction_loss(
            reconstructions, inputs
        ).sum()
        return((margin_loss+self.a*reconstruction_loss)/inputs.size(0))