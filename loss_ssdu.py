import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedL1L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, v):
       
        if torch.is_complex(u):
            v = torch.complex(v, torch.zeros_like(v))
        else:
            u = torch.complex(u, torch.zeros_like(u))
       
        l1_loss = F.l1_loss(u.abs(), v.abs(), reduction='none')
        l2_loss = F.mse_loss(u.abs(), v.abs(), reduction='none')
       
        norm_u_l1 = torch.norm(u.abs(), p=1)
        norm_u_l2 = torch.norm(u.abs(), p=2)
       
        loss = (torch.sum(l2_loss) / norm_u_l2) + (torch.sum(l1_loss) / norm_u_l1)
        return loss
