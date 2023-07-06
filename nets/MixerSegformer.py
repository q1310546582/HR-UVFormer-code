import numpy as np
from einops import einops
from torch import nn
import torch

class MLPBlock(nn.Module):
    def __init__(self, mlp_dim:int, hidden_dim:int, dropout = 0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)
    def forward(self,x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x

class Mixer_struc(nn.Module):
    def __init__(self, patches: int , token_dim: int, dim: int,channel_dim: int,dropout = 0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches # 100
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(self.patches,token_dim,self.dropout)
        self.MLP_block_chan = MLPBlock(dim,channel_dim,self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self,x):
        out = self.LayerNorm(x)
        out = einops.rearrange(out, 'b n d -> b d n')
        out = self.MLP_block_token(out)
        out = einops.rearrange(out, 'b d n -> b n d')
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2+=out
        return out2


class MixerSegformer(nn.Module):
    def __init__(self, backbone, input_shape, patch_size, token_hidden_dim, channel_hidden_dim, num_classes, num_blocks):
        super(MixerSegformer, self).__init__()
        self.num_classes = num_classes
        self.window_size =input_shape//patch_size
        self.n_patches =self.window_size **2
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.patch_size = patch_size
        self.embbeding_dim = self.backbone.embedding_dim
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=self.n_patches, token_dim=token_hidden_dim,channel_dim=channel_hidden_dim, dim=self.embbeding_dim) for i in range(num_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(self.embbeding_dim)
        self.pred_seg   = nn.Conv2d(self.embbeding_dim, num_classes,1)

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, self.embbeding_dim))  # (1,100,768*2)

    def forward(self,x):
        out = self.embedding(x)
        b, n, c = out.size()
        out = out + self.position_embeddings
        # out = einops.rearrange(out,"n c h w -> n (h w) c")
        for block in self.blocks:
            out = block(out)
        out = self.Layernorm1(out)
        # out = out.mean(dim = 1) # out（n_sample,dim）

        out = out.view(b, self.window_size, self.window_size, self.embbeding_dim).permute(0,3,1,2)
        result = self.pred_seg(out) # (B,num_classes,H,W)

        return [None, result]

    def embedding(self, images):
        patches = []
        steps = int(np.sqrt(self.n_patches))
        for row in range(steps):
            for col in range(steps):
                # (B,3,200,200)
                patch_image = images[:,:,row * self.patch_size: (row + 1) * self.patch_size, col * self.patch_size: (col + 1) * self.patch_size]
                _ = self.backbone(patch_image)
                patch_feature  =  self.backbone.decode_head.backbone_output # (B,2*embedding_dim)
                patches.append(patch_feature.unsqueeze(0))
        x = torch.cat(patches,dim=0).transpose(0,1) # (n_patches,B,2*embedding_dim) => (B,n_patches,2*embedding_dim)
        return x
