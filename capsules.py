import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax


def anomaly_scores(lengths, inp, reconstruction, normal_class=0, anomaly_class=1):
    """Anomaly scores based on https://arxiv.org/pdf/1909.02755.pdf"""
    difference = lengths.T[normal_class] - lengths.T[anomaly_class]
    return(difference, np.sum((inp-reconstruction)**2, 1))
    
def normality_scores(lengths, inp, reconstruction, use_softmax=True):
    """Normality scores based on https://arxiv.org/pdf/1907.06312.pdf"""
    u = softmax(lengths, axis=1).max(1) if use_softmax else lengths.max(1)
    return(
        u, -np.sum((inp-reconstruction)**2, 1)/(np.sum((inp**2), 1)**0.5)
    )

def squash(vector, dim=-1):
    """Activation function for capsule"""
    sj2 = (vector**2).sum(dim, keepdim=True)
    sj = sj2 ** 0.5
    return(
        sj2/(1.0+sj2)*vector/sj
    )

def make_y(labels, n_classes, use_cuda=True):
    masked = torch.eye(n_classes)
    masked = masked.cuda() if torch.cuda.is_available() and use_cuda else masked
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
    
    def __init__(
        self, n_capsules=10, n_iter=3, n_routes=32*6*6, in_ch=8, out_ch=16, 
        return_couplings=False, cuda=True
    ):
        super(SecondaryCapsuleLayer, self).__init__()
        self.n_capsules = n_capsules
        self.n_iter = n_iter
        self.n_routes = n_routes
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rc = return_couplings
        self.W = nn.Parameter(
            torch.randn(self.n_capsules, self.n_routes, self.in_ch, self.out_ch)
        )
        self.cuda = cuda
    
    def forward(self, x):
        P = x[None, :, :, None, :] @ self.W[:, None, :, :, :]
        L = torch.zeros(*P.size())
        L = L.cuda() if torch.cuda.is_available() and self.cuda else L
        for i in range(self.n_iter):
            probabilities = F.softmax(L, dim=2)
            out = squash((probabilities*P).sum(dim=2, keepdim=True))
            if i != self.n_iter - 1:
                L = L + (P*out).sum(dim=-1, keepdim=True)
        out = out.squeeze().transpose(1,0)
        if x.shape[0] == 1:
            out = out.reshape(1, *out.shape).transpose(2,1)
        if self.rc:
            return(out, probabilities.cpu().data.numpy())
        else:
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
    
    
class HitOrMissLayer(nn.Module):
    
    def __init__(
        self, in_ch=256*6*6, out_ch=32, n_classes=10
    ):
        """Hit or Miss layer from Hitnet paper (https://arxiv.org/pdf/1806.06519.pdf)"""
        super(HitOrMissLayer, self).__init__()
        self.hom = nn.Sequential(
            nn.Linear(in_ch, n_classes*out_ch),
            nn.BatchNorm1d(n_classes*out_ch),
            nn.Sigmoid()
        )
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_classes = n_classes
    
    def forward(self, x):
        xx = x.reshape(x.shape[0], -1)
        return(self.hom(xx).reshape(xx.shape[0], self.n_classes, self.out_ch))

    
def mask_hom(hom, ys):
    """Masking function for Hitnet"""
    def make_mask(y):
        return(y.repeat(hom.shape[-1], 1).permute(1, 0))
    mask = torch.stack([make_mask(a) for a in ys])
    mask = mask.cuda() if torch.cuda.is_available() else mask
    return(hom*mask)


class CentripetalLoss(nn.Module):
    
    def __init__(
        self, m_hit=0.1, h_step=0.1, v_step=0.2, m_miss=0.9, caps_dimension=16,
        use_ghosts=False, adjustment=0.005
    ):
        """Loss for HitOrMiss capsules"""
        super(CentripetalLoss, self).__init__()
        self.m_hit = m_hit
        self.h_step = h_step
        self.v_step = v_step
        self.caps_dimension = caps_dimension
        self.use_ghosts = use_ghosts
        self.m_miss = m_miss
        self.reconstruction_loss = nn.MSELoss(reduction="mean")
        self.adjustment = adjustment
        
    def forward(self, predictions, true_labels, inputs, reconstructions):
        floored = ((predictions-self.m_hit)/self.h_step).floor()
        zero = torch.Tensor(0)
        heaviside = torch.sign(F.relu(predictions-self.m_hit))
        L1_1 = (floored*(floored+1))*self.v_step*self.h_step*0.5
        L1_2 = (floored+1)*self.v_step*(predictions-self.m_hit-floored*self.h_step)
        L1 = heaviside*(L1_1+L1_2)
        r_factor = 0.5*(self.caps_dimension**0.5)
        m_miss_r = r_factor-self.m_miss
        predictions_r = r_factor-predictions
        floored = ((predictions_r-m_miss_r)/self.h_step).floor()
        heaviside = torch.sign(F.relu(predictions_r-m_miss_r))
        L2_1 = (floored*(floored+1))*self.h_step*self.v_step*0.5
        L2_2 = (floored+1)*self.v_step*(predictions_r-m_miss_r-floored*self.h_step)
        L2 = heaviside*(L2_1+L2_2)
        L_cp = true_labels*L1+0.5*(1-true_labels)*L2
        L_rec = self.reconstruction_loss(
            inputs.reshape(inputs.shape[0], -1), reconstructions
        )
        L_final = L_cp+self.adjustment*L_rec
        return(L_final.sum(1).mean())