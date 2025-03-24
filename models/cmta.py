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


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

class CMTA(nn.Module):
    def __init__(self, fusion='moe', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, topk=1024,
                dropout=0.25):
        super(CMTA, self).__init__()
        self.fusion = fusion
        self.topk = topk
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.dropout = dropout

        feature_dim = 256

        # self.n_clusters = n_clusters
        ### FC Layer over WSI bag
        # fc = [nn.Linear(1024, 512), nn.ReLU()]
        # fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(nn.Linear(1024, feature_dim), 
                                    nn.ReLU(), 
                                    nn.Dropout(0.25))

        self.snn = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False)
        )

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=feature_dim)
        # Decoder
        self.pathomics_decoder = Transformer_P(feature_dim=feature_dim)

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=feature_dim, num_heads=1)
        # G->P Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=feature_dim, num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=feature_dim)
        # Decoder
        self.genomics_decoder = Transformer_G(feature_dim=feature_dim)


        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(feature_dim*2, 512), nn.ReLU(), nn.Linear(512, feature_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=feature_dim, dim2=feature_dim, scale_dim1=8, scale_dim2=8, mmhid=feature_dim)

        
        ### Classifier
        self.classifier = nn.Linear(feature_dim, n_classes)


    def forward(self, **kwargs):
        x_path = kwargs['x_path'] # (num_patch, 1024)
        # x_omic = kwargs['x_omic'].reshape(-1, 512) # (1, num_omic)
        x_mrna = kwargs['x_mrna'].reshape(-1, 512)
        x_cnv = kwargs['x_cnv'].reshape(-1, 512)
        x_omic = torch.cat([x_mrna, x_cnv], axis=0)
        h_path_bag = self.wsi_net(x_path).unsqueeze(0) ### path embeddings are fed through a FC layer [1, n, 512]
        h_omic_bag = self.snn(x_omic).unsqueeze(0) ### omic embeddings are fed through a FC layer [1, n, 512]

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            h_path_bag)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            h_omic_bag)  # cls token + patch tokens

        # cross-omics attention
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        # decoder
        # pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        # genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ),
                    dim=1,
                )
            )  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
            )  # take cls token to make prediction
        elif self.fusion == 'moe':
            all_feat = torch.cat(((cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2, (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2), 1)
            weights = self.gating_network(all_feat)
            fusion = weights @ torch.cat(((cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2, (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2), 0)
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))
        
        logits = self.classifier(fusion.reshape(1, -1)) ## K x num_cls
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        return  logits, Y_prob, Y_hat, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_genomics_encoder, cls_token_genomics_decoder

