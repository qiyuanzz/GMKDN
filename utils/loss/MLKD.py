# from termios import CEOL
# from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from ._base import Distiller

# 
# import torch
# import torch.nn as nn

def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class KD_ours(nn.Module):
    def __init__(self, cfg):
        super(KD_ours, self).__init__()
        self.temperature = cfg.temp
        self.kd_loss_weight = 1

    def forward(self, f_s, f_t):


        pred_teacher_weak = F.softmax(f_t.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()
        # losses
        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            f_s,
            f_t,
            self.temperature,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            f_s,
            f_t,
            3.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            f_s,
            f_t,
            5.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            f_s,
            f_t,
            2.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            f_s,
            f_t,
            6.0,
            # reduce=False
        ) * mask).mean())


        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            f_s,
            f_t,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            f_s,
            f_t,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            f_s,
            f_t,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            f_s,
            f_t,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            f_s,
            f_t,
            6.0,
        ) * class_conf_mask).mean())

        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            f_s,
            f_t,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            f_s,
            f_t,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            f_s,
            f_t,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            f_s,
            f_t,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            f_s,
            f_t,
            6.0,
        ) * mask).mean())

        loss = loss_kd_weak + loss_cc_weak + loss_bc_weak
        
        return  loss