#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_coattn.py
@Time    :   2022/07/07 16:43:59
@Author  :   Innse Xu 
@Contact :   innse76@gmail.com
'''

# Here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils.model_utils import *



###########################
### MCAT Implementation ###
###########################
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        # x = x.unsqueeze(0)
        ## x: N x L
        # x = x.squeeze(0)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights( A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N



class ABMIL_SNN(nn.Module):
    def __init__(self, fusion='bilinear', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, topk=1024,
                dropout=0.25):
        super(ABMIL_SNN, self).__init__()
        self.fusion = fusion
        self.topk = topk
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.dropout = dropout

        feature_dim = 256


        self.wsi_net = nn.Sequential(nn.Linear(1024, feature_dim), 
                                    nn.ReLU(), 
                                    nn.Dropout(0.25))

        self.cnv = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )
        self.mrna = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )
        self.coattn_mrna = MultiheadAttention(embed_dim=256, num_heads=1)
        self.coattn_cnv = MultiheadAttention(embed_dim=256, num_heads=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256*4, nhead=8, dim_feedforward=256*4, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # path_mrna_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=256, dropout=dropout, activation='relu')
        # self.path_mrna_transformer = nn.TransformerEncoder(path_mrna_encoder_layer, num_layers=2)

        # path_cnv_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=256, dropout=dropout, activation='relu')
        # self.path_cnv_transformer = nn.TransformerEncoder(path_cnv_encoder_layer, num_layers=2)

        # mrna_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=256, dropout=dropout, activation='relu')
        # self.mrna_transformer = nn.TransformerEncoder(mrna_encoder_layer, num_layers=2)

        # cnv_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=256, dropout=dropout, activation='relu')
        # self.cnv_transformer = nn.TransformerEncoder(cnv_encoder_layer, num_layers=2)

        self.multi_fc = nn.Sequential(
            nn.Linear(256*4, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention_path = Attention_Gated(L=256, D=128, K=1)
        self.attention_snn = Attention_Gated(L=256, D=128, K=1)
        self.attention = Attention_Gated(L=256, D=128, K=1)


        self.mm_multi = BilinearFusion(dim1=feature_dim, dim2=feature_dim, scale_dim1=8, scale_dim2=8, mmhid=feature_dim)
        self.mm_omic = BilinearFusion(dim1=feature_dim, dim2=feature_dim, scale_dim1=8, scale_dim2=8, mmhid=feature_dim)
        ### Classifier
        self.classifier = nn.Linear(feature_dim, n_classes)


    def forward(self, **kwargs):
        x_path = kwargs['x_path'] # (num_patch, 1024)
        x_mrna = kwargs['x_mrna'].reshape(-1, 512) # (1, num_omic)
        x_cnv = kwargs['x_cnv'].reshape(-1, 512)

        h_path_bag = self.wsi_net(x_path)
        h_mrna_bag = self.mrna(x_mrna)
        h_cnv_bag = self.cnv(x_cnv)

        AA_path = self.attention_path(h_path_bag)
        h_path_bag = torch.mm(AA_path, h_path_bag)

        AA_mrna = self.attention_snn(h_mrna_bag)
        h_mrna_bag = torch.mm(AA_mrna, h_mrna_bag)

        AA_cnv = self.attention_snn(h_cnv_bag)
        h_cnv_bag = torch.mm(AA_cnv, h_cnv_bag)

        h_omic_bga = self.mm_omic(h_cnv_bag, h_mrna_bag).view(-1, 256)
        fusion = self.mm_multi(h_omic_bga, h_path_bag).view(-1, 256)

        
        logits = self.classifier(fusion.reshape(1, -1)) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        return  logits, Y_prob, Y_hat, fusion

class ABMIL(nn.Module):
    def __init__(self, L=256, D=128, K=1, n_classes=2, dropout=0):
        super(ABMIL, self).__init__()
        
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, n_classes, dropout)
        self._fc1 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        init_max_weights(self)
    def forward(self, **kwargs): ## x: N x L
        h  = kwargs['x_path'] #[B, n, 1024]
        h = self._fc1(h) #[B, n, 256]
        x = h.squeeze(0)
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        logits = self.classifier(afeat) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits, Y_prob, Y_hat, afeat



class omic_transformer(nn.Module):
    def __init__(self, n_classes: int=2, dropout: float=0.25, topk=1024) -> None:
        super(omic_transformer, self).__init__()
        feature_dim = 256
        self.n_classes =  n_classes
        self.dropout = dropout
        self.topk = topk
        self.fc1 = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )
        # encoder_layer = nn.TransformerEncoderLayer(d_model=256*2, nhead=8, dim_feedforward=256*2, dropout=dropout, activation='relu')
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attention = Attention_Gated(L=256, D=128, K=1)
        self.attention_rmna = Attention_Gated(L=256, D=128, K=1)
        self.mm = BilinearFusion(dim1=feature_dim, dim2=feature_dim, scale_dim1=8, scale_dim2=8, mmhid=feature_dim)
        self.classifier = nn.Linear(feature_dim, n_classes)

        init_max_weights(self)

    def forward(self, **kwargs):
        x_mrna = kwargs['x_mrna'].reshape(-1, 512) 
        x_cnv = kwargs['x_cnv'].reshape(-1, 512)
        x_mrna = self.fc1(x_mrna)
        x_cnv = self.fc2(x_cnv)

        AA_mrna = self.attention(x_mrna)
        mrna = torch.mm(AA_mrna, x_mrna)

        AA_cnv = self.attention(x_cnv)
        cnv = torch.mm(AA_cnv, x_cnv)



        afeat = self.mm(cnv, mrna).view(-1, 256)
        logits = self.classifier(afeat)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)
        return logits, Y_prob, Y_hat, afeat
