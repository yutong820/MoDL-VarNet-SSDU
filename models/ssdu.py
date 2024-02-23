import torch
import torch.nn as nn

from utils import r2c, c2r
from models import mri

    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.scale_factor = scale_factor
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.scale_factor
        out += residual
        return out
    
class ResNet(nn.Module):
    def __init__(self, num_blocks=15, scale_factor=0.1):
        super().__init__()
        self.initial_conv = nn.Conv2d(2, 64, 3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(64, 64, scale_factor=scale_factor)
            for _ in range(num_blocks)
        ])
    
        self.final_conv = nn.Conv2d(64, 2, 3, padding=1)
        
    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

#CG algorithm ======================
class myAtA(nn.Module): # implement AHA+λI 
    """
    performs DC step
    """
    def __init__(self                                                                                                                          , csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol), coil sensivity maps
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam # regularizaion 

        self.A = mri.SenseOp(csm, mask)

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im_u = self.A.adj(self.A.fwd(im))
        return im_u + self.lam * im

def myCG(AtA, rhs): # solve Ax=b, AtA is AHA+λI, rhs is b
    """
    performs CG algorithm
    :AtA: a class object that contains csm, mask and lambda and operates forward model
    """
   
    rhs = r2c(rhs, axis=1) # nrow, ncol
    x = torch.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rTr = torch.sum(r.conj()*r).real
    while i < 10 and rTr > 1e-10:
        Ap = AtA(p)
        alpha = rTr / torch.sum(p.conj()*Ap).real
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return c2r(x, axis=1)

class data_consistency(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(self, z_k, x0, csm, mask):# denoised image z_k, initial reconstruct x0
        rhs = x0 + self.lam * z_k # (2, nrow, ncol)
        AtA = myAtA(csm, mask, self.lam)
        rec = myCG(AtA, rhs) #rhs is the combine of initial recon and denoised
        return rec

class SSDU(nn.Module):
    def __init__(self, k_iters):
        super().__init__()
        self.k_iters = k_iters
        self.dw = ResNet()
        self.dc = data_consistency()
        
    def forward(self, x0, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """

        x_k = x0.clone()
        for k in range(self.k_iters):
            # cnn denoiser
            z_k = self.dw(x_k) # (2, nrow, ncol)
            # data consistency
            x_k = self.dc(z_k, x0, csm, mask) # (2, nrow, ncol)
        return x_k
    