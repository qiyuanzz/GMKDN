import torch.nn as nn
import torch.nn.functional as F

class COL_KD(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, beta=1.0, gamma=1.0):
        super(COL_KD, self).__init__()
        self.T = T
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        inter_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)

        p_s = F.log_softmax(y_s.T/self.T, dim=1)
        p_t = F.softmax(y_t.T/self.T, dim=1)
        intra_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2) 
        loss = self.beta * inter_loss + self.gamma * intra_loss

        return loss