import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange

from thop import profile


class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.
    ):
        """
        Attention Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. Rearrange
        4. LayerNorm
        5. Linear
        6. Rearrange
        7. Softmax
        8. Dropout
        9. Rearrange
        10. Linear
        11. Dropout
        
        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Rearrange input tensor
        3. Apply non-linearity to the input
        4. Rearrange input tensor
        5. Apply softmax to the input
        6. Apply dropout to the input
        7. Rearrange input tensor
        8. Apply non-linearity to the input
        
        """
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        #layer norm
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)

        #sftmx
        self.attend = nn.Softmax(dim = -1)

        #dropout
        self.dropout = nn.Dropout(dropout)

        #projections, split from x -> q, k, v
        self.to_qkv = nn.Linear(
            dim, 
            inner_dim * 3, 
            bias = False
        )
        
        #project out
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #apply layernorm to x
        x = self.norm(x)

        #apply linear layer to x
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        #rearrange x to original shape
        q, k, v = map(
            lambda t: rearrange(
                t, 
                'b n (h d) -> b h n d', 
                h = self.heads
            ), qkv)

        # #normalize key and values, known QK Normalization
        k = self.norm_k(k)
        v = self.norm_v(v)
        
        # attn
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            #Flash Attention
            out = F.scaled_dot_product_attention(q, k, v)
            
            #dropout
            out = self.dropout(out)

            #rearrange to original shape
            out = rearrange(out, 'b h n d -> b n (h d)')

            #project out
            return self.to_out(out)

class HydraAttention(nn.Module):
    def __init__(self, dim=1024, output_layer='linear', dropout=0.0, heads = 8):
        super(HydraAttention, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.d_model = dim
        self.out = nn.Linear(dim, dim) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 
        self.heads = heads
        self.to_qkv = nn.Linear(
            dim, 
            dim * 3, 
            bias = False
        )
        self.sig= nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gl = nn.GELU()
    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(
            lambda t: rearrange(
                t, 
                'b n (h d) -> b h n d', 
                h = self.heads
            ), qkv)
        
        q = self.sig(q)
        k = self.tanh(k)
        #q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
        #k = k / torch.norm(k, p=2, dim=-1, keepdim=True)
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        k = k.transpose(-1, -2)
        kvw = torch.matmul(k, v)
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out = torch.matmul(q, kvw)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out)
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=1024):
        super().__init__()
        self.norm = norm_layer(dim)
        # self.attn = NystromAttention(
        #     dim = dim,
        #     dim_head = dim//8,
        #     heads = 8,
        #     num_landmarks = dim//2,    # number of landmarks
        #     pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #     residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #     dropout=0.1
        # )
        self.attn = Attention(dim=dim)
    def forward(self, x):
        x = x + self.attn((x))
        return x
class TransLayer1(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=1024):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = HydraAttention(dim=dim)
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
class DAMLN(nn.Module):
    def __init__(self, n_classes=2, ds=512):
        super(DAMLN, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, ds), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, ds))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=ds)
        self.layer2 = TransLayer(dim=ds)
        self.layer3 = TransLayer1(dim=ds)
        self.layer4 = TransLayer1(dim=ds)
        self.norm = nn.LayerNorm(ds)
        self._fc2 = nn.Linear(ds, self.n_classes)

    def forward(self, **kwargs):

        h = kwargs['x_path'].unsqueeze(0).float() #[B, n, 1024]
        h = self._fc1(h)
        h = self.layer1(h)
        h = self.layer2(h)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.norm(h)[:,0]
        
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        return logits, Y_prob, Y_hat

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = DAMLN().cuda()
    #print(model.eval())
    #results_dict = model(data = data)
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)