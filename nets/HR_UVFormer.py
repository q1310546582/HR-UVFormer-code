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
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, query = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        if query != None:
            q = query
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

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


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
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        self.attn1 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim//2, act_layer=act_layer, drop=drop)

        self.mlpfusion = Mlp(in_features=dim*2, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)

        self.merge_conv1 = nn.Conv2d(dim, dim, self.window_size, groups=dim)
        self.merge_conv2 = nn.Conv2d(dim, dim, self.window_size, groups=dim)

    def forward(self, window_image, window_building):
        H, W = self.input_resolution

        if window_image != None and window_building != None:
            B, L, C = window_image.shape
            assert L == H * W, "input feature has wrong size"
            attn_image, q1 = self.selfAttn1(window_image,query = None) # B N C
            attn_building, _  = self.selfAttn2(window_building,query = None)
            image_attn_build, _ = self.selfAttn2(attn_building,query = q1) # B N C

            x = torch.cat([attn_image, image_attn_build], dim=-1) # B N 2C
            x = self.mlpfusion(x) + attn_image # B N C

            attn_building = self.merge_conv1(attn_building.view(B,H,W,self.dim).permute(0,3,1,2))
            attn_building = attn_building.squeeze(-1).transpose(1,2)

        else:
            x = window_image
            attn_building = None
            B, L, C = x.shape  # (B,9,768)
            assert L == H * W, "input feature has wrong size"
            x, _ = self.selfAttn1(x)

        x = x.view(B,H,W,self.dim).permute(0,3,1,2) # B C N => B C H W
        x = self.merge_conv2(x) # B C 1 1
        x = x.squeeze(-1).transpose(1,2) # B 1 C

        return x, attn_building # B 1 C

    def selfAttn1(self, x, query = None):

        shortcut = x
        x = self.norm1(x) # LN
        x,q = self.attn1(x, query)  # B,N,C
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp1(self.norm2(x))) # B N C

        return x,q

    def selfAttn2(self, x, query = None):
        shortcut = x
        x = self.norm3(x) # LN
        x,q = self.attn2(x, query)  # B,N,C
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp2(self.norm4(x))) # B N C

        return x, q

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

        self.window_size = window_size

    def forward(self, image, building):
        # block 3*3 slide local attention
        i = 1

        for blk in self.blocks:
            B, N, C = image.size()
            image = self.zero_padding(image, h=self.window_size//2, w=self.window_size//2)
            outputs_image = []

            if building is not None:
                building = self.zero_padding(building, h=self.window_size//2, w=self.window_size//2)
                outputs_build = []

            # 3*3 slide local attention
            for row in range(self.input_resolution[0]):
                for col in range(self.input_resolution[1]):
                    slide_window_image = image[:, :, row:row+self.window_size, col:col+self.window_size].contiguous() # B C window_size window_size

                    slide_window_image = slide_window_image.view(B, C, self.window_size**2).transpose(1,2) # B window_size**2 C

                    if building is not None:
                        slide_window_building = building[:, :, row:row + self.window_size,
                                                col:col + self.window_size].contiguous()

                        slide_window_building = slide_window_building.view(B, C, self.window_size ** 2).transpose(1, 2)
                    else:
                        slide_window_building = None

                    fusion_feature,window_building = blk(slide_window_image, slide_window_building) # B 1 C

                    outputs_image.append(fusion_feature)

                    if window_building is not None:
                        outputs_build.append(window_building)

            image    = torch.cat(outputs_image, dim = 1) # B num_windows C
            if building is not None:
                building = torch.cat(outputs_build, dim = 1)

            i += 1
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

""" overview structure"""
class HR_UVFormer(nn.Module):
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

    def __init__(self, backbone, input_shape=2000,  patch_size=200, num_classes=3,
                 depths=2, num_heads=3,
                 window_size=3, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False,
                 building_bone=None, freeze_backbone = True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_depth = depths
        self.num_heads = num_heads
        self.backbone = backbone
        freeze_backbone = False
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.building_bone = building_bone
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
        self.norm3 = norm_layer(self.embed_dim)
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

    def forward_features(self, image, building):
        image, building = self.embedding(image, building)
        image = image.detach()
        image = self.norm1(image)
        b, n, c = image.size()
        if self.ape:
            image = image + self.absolute_pos_embed
        image = self.pos_drop(image) # Dropout

        if building is not None:
            building = self.norm2(building)
            if self.ape:
                building = building + self.absolute_pos_embed
            building = self.pos_drop(building)

        fusion = self.block_layer(image, building)

        fusion = self.norm3(fusion)  # B N C
        fusion = fusion.transpose(1, 2).contiguous()  # B C N

        fusion = fusion.view(b, self.embed_dim, self.patches_resolution, self.patches_resolution) # B C H W

        return fusion

    def forward(self, image, building = None):
        x = self.forward_features(image, building)
        x = self.pred_seg(x)

        return [None, x]

    def embedding(self, images, building):
        patches = []
        buildings = []
        if building is not None:
            assert images.shape[2]*images.shape[3] - building.shape[2]*building.shape[3] == 0, "input a and b have different size"
        B,C,H,W = images.shape
        if H != self.input_shape or W != self.input_shape:
            images   = F.interpolate(images, size=(self.input_shape, self.input_shape),
                                        mode='bicubic', align_corners=True)
            if building != None:
                building = F.interpolate(building, size=(self.input_shape, self.input_shape),
                                            mode='bicubic', align_corners=True)
            print(f'warning: input size not equal to input_shape:{self.input_shape}, interpolated!')

        for row in range(self.patches_resolution):
            for col in range(self.patches_resolution):
                # (B,3,200,200)
                patch_image = images[:,:,row * self.patch_size: (row + 1) * self.patch_size, col * self.patch_size: (col + 1) * self.patch_size]

                _ = self.backbone(patch_image)
                patch_feature  =  self.backbone.decode_head.backbone_output # (B,embedding_dim)
                patches.append(patch_feature.unsqueeze(0))

                if self.building_bone is not None and building is not None:
                    patch_building = building[:, :, row * self.patch_size: (row + 1) * self.patch_size,
                                     col * self.patch_size: (col + 1) * self.patch_size]
                    _ = self.building_bone(patch_building)
                    building_feature = self.building_bone.decode_head.backbone_output
                    buildings.append(building_feature.unsqueeze(0))
                else:
                    buildings = None

        patches = torch.cat(patches,dim=0).transpose(0,1) # (n_patches,B,embedding_dim) => (B,n_patches,embedding_dim)
        if buildings is not None:
            buildings = torch.cat(buildings,dim=0).transpose(0,1)
        return patches, buildings


if __name__ == "__main__":
    backbone_image = SegFormer(num_classes_seg=2, num_classes_cls=3, phi='b3', classification=True,
                               segmentation=True, input_shape=200)
    backbone_build = SegFormer(num_classes_seg=2, num_classes_cls=3, phi='b0', classification=True,
                               segmentation=False, input_shape=200, input_channels = 4)
    model = HR_UVFormer(backbone_image, input_shape=2000, num_classes=3,
                        building_bone=backbone_build)

    params1 = sum([param.numel() for param in model.parameters()])
    params2 = sum([param.numel() for param in model.backbone.parameters()])
    params3 = sum([param.numel() for param in model.building_bone.parameters()])
    print("%.2fM" % (params1/1e6),"%.3fM" % (params2/1e6),"%.2fM" % (params3/1e6))
    for name, param in model.named_parameters():
        print(name)
        print("%.2fM" % (param.numel()/1e6))
        print("-----------------------------------")

    x1 = torch.randn([1,3,2000,2000], dtype=torch.float32)
    x2 = torch.randn([1, 4, 2000, 2000], dtype=torch.float32)
    output = model(x1, x2)
    print(output[1].shape)