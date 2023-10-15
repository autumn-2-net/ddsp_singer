from .base_attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F
class ffn(nn.Module):
    def __init__(self,dim,hideen_dim,kernel_size):
        super().__init__()
        self.ffn_1 = nn.Conv1d(
            dim, hideen_dim, kernel_size, padding=kernel_size // 2
        )
        self.kernel_size=kernel_size


        self.ffn_2 = nn.Linear(hideen_dim, dim)
        self.act=nn.GELU()

    def forward(self, x:torch.tensor):#x B T C
        x=self.act(self.ffn_1(x.transpose(1,2))#* self.kernel_size ** -0.5
                   )
        # x = x * self.kernel_size ** -0.5
        return self.ffn_2(x.transpose(1,2))

class attn_lay(nn.Module): #pre norm
    def __init__(self,dim, heads=4, dim_head=64,hideen_dim=None,kernel_size=9):
        super().__init__()
        if hideen_dim is None:
            hideen_dim=dim*4
        self.ffn=ffn(dim=dim,hideen_dim=hideen_dim,kernel_size=kernel_size)
        self.attn=Attention(dim, heads=heads, dim_head=dim_head,)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self,x,mask=None):
        # if mask is not None:
        #     x=x.masked_fill(~mask.unsqueeze(-1), 0)
        x=self.attn(self.layer_norm1(x),mask=mask )+x
        # if mask is not None:
        #     x=x.masked_fill(~mask.unsqueeze(-1), 0)
        x=self.ffn(self.layer_norm2(x))+x
        # if mask is not None:
        #     x=x.masked_fill(~mask.unsqueeze(-1), 0)
        return x
