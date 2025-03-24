from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def batch_normalize(logit):
    mean = torch.mean(logit)
    stdv = torch.std(logit)
    return (logit - mean) / (1e-7 + stdv)

class DistillKL_logit_stand(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, temp):
        super(DistillKL_logit_stand, self).__init__()
        self.temp = temp
    def forward(self, y_s, y_t):
        T = self.temp
        
        KD_loss = 0
        y_s = batch_normalize(y_s)/T
        y_t = batch_normalize(y_t)/T
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y_s, dim=1),
                                F.softmax(y_t, dim=1)) * T * T
        
        return KD_loss
