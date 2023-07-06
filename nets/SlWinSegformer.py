# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from nets.segformer import SegFormer
from osgeo import gdal

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 #
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1)) # (Q*K^T/d^0.5)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        return x, q

class TransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=3,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.merge_conv = nn.Conv2d(dim, dim, self.window_size, groups=dim)

    def forward(self, window_image):
        H, W = self.input_resolution

        x = window_image
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.selfAttn(x)

        x = self.mergeConv(x)

        return x

    def selfAttn(self, x):

        shortcut = x
        x = self.norm1(x) # LN
        x,q = self.attn(x)  # B,N,C
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x))) # B N C

        return x

    def mergeConv(self, x):

        x = x.transpose(1, 2)  # B C N

        x = x.view(-1, self.dim, self.input_resolution[0], self.input_resolution[1])  # B C H W
        x = self.merge_conv(x) # B C 1 1
        x = x.squeeze(-1).transpose(1,2) # B 1 C

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.norm = norm_layer(self.dim)
        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=(window_size, window_size),
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        self.downsample = None
        self.window_size = window_size

    def forward(self, image):
        # block 3*3 slide local attention

        for blk in self.blocks:
            B, N, C = image.size()
            image = self.zero_padding(image, h=self.window_size//2, w=self.window_size//2)
            outputs_image = []

            # 3*3 sliding window local attention
            for row in range(self.input_resolution[0]):
                for col in range(self.input_resolution[1]):
                    slide_window_image = image[:, :, row:row+self.window_size, col:col+self.window_size].contiguous() # B C window_size window_size

                    slide_window_image = slide_window_image.view(B, C, self.window_size**2).transpose(1,2) # B window_size**2 C

                    fusion_feature = blk(slide_window_image) # B 1 C

                    outputs_image.append(fusion_feature)

            image    = torch.cat(outputs_image, dim = 1) # B num_windows C

        return image

    def zero_padding(self, x, h=1, w=1):
        B, N, C = x.size()
        x = x.transpose(1, 2)  # B C N
        x = x.view(-1, self.dim, self.input_resolution[0], self.input_resolution[1])  # B C H W
        zero_pad_1 = torch.zeros([B, C, self.input_resolution[0], w], dtype=torch.float32)
        zero_pad_2 = torch.zeros([B, C, h, self.input_resolution[1] + 2*w], dtype=torch.float32)
        zero_pad_1 = zero_pad_1.cuda() if x.device.type == 'cuda' else zero_pad_1
        zero_pad_2 = zero_pad_2.cuda() if x.device.type == 'cuda' else zero_pad_2
        x = torch.cat([zero_pad_1, x, zero_pad_1], dim=3)
        x = torch.cat([zero_pad_2, x, zero_pad_2], dim=2)  # # B C H+2 W+2
        return x

class SlWinSegformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 3
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
    """

    def __init__(self, backbone, input_shape=2000, patch_size=200, num_classes=3,
                 depths=2, num_heads=3,
                 window_size=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_depth = depths
        self.num_heads = num_heads
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embed_dim = self.backbone.embedding_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.input_shape = input_shape
        self.patch_size = patch_size
        # split image into non-overlapping patches
        self.num_patches = (self.input_shape // self.patch_size)**2
        self.patches_resolution = self.input_shape // self.patch_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layer
        layer = BasicLayer(dim=int(self.embed_dim),
                           input_resolution=(self.patches_resolution,
                                             self.patches_resolution),
                           depth=self.num_depth,
                           num_heads=self.num_heads,
                           window_size=window_size,
                           mlp_ratio=self.mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate,
                           drop_path=0.,
                           norm_layer=norm_layer,)
        self.block_layer = layer

        self.norm1 = norm_layer(self.embed_dim)
        self.norm2 = norm_layer(self.embed_dim)
        self.pred_seg   = nn.Conv2d(self.embed_dim, num_classes,1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.embedding(x)
        x = self.norm1(x)
        b, n, c = x.size()
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.block_layer(x)

        x = self.norm2(x)  # B N C
        x = x.transpose(1, 2).contiguous()  # B C N

        x = x.view(b, self.embed_dim, self.patches_resolution, self.patches_resolution) # B C H W
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pred_seg(x)

        return [None, x]


    def embedding(self, images):
        patches = []

        B,C,H,W = images.shape
        if H != self.input_shape or W != self.input_shape:
            images   = F.interpolate(images, size=(self.input_shape, self.input_shape),
                                        mode='bicubic', align_corners=True)

            print(f'warning: input size not equal to input_shape:{self.input_shape}, interpolated!')

        for row in range(self.patches_resolution):
            for col in range(self.patches_resolution):
                # (B,3,200,200)
                patch_image = images[:,:,row * self.patch_size: (row + 1) * self.patch_size, col * self.patch_size: (col + 1) * self.patch_size]


                _ = self.backbone(patch_image)
                patch_feature  =  self.backbone.decode_head.backbone_output # (B,embedding_dim)
                patches.append(patch_feature.unsqueeze(0))

        patches = torch.cat(patches,dim=0).transpose(0,1) # (n_patches,B,embedding_dim) => (B,n_patches,embedding_dim)

        return patches

if __name__ == "__main__":
    backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi='b3', classification=True,
                               segmentation=True, input_shape=200)
    model = SlWinSegformer(backbone, input_shape=2000, patch_size=200, window_size=4, num_classes=3)
    params = sum([param.numel() for param in model.parameters()])
    print("%.2fM" % (params/1e6))
    x = torch.randn([1,3,2000,2000], dtype=torch.float32)
    x = model(x)
    print(x[1].shape)