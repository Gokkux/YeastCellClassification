import torch.nn as nn
import torch
from typing import List, Tuple, Union
from mmcv.cnn import ConvModule
import torch.nn.functional as F
# EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# https://arxiv.org/pdf/2405.06880

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='gelu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shift = shift_module(shift_size=1)
        self.up_dwc = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(self.in_channels, self.in_channels, 
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.shift(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class EUCB_MultiKernel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
    upsample_kernel_size: int,
    upsample_stride: int,
    upsample_padding: int,
    kernel_sizes: List[int] = [3, 5], 
    stride: int = 1, activation: str = 'gelu'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shift = shift_module(shift_size=1)

        # 1. ConvTranspose2d 用于上采样
        self.conv_transpose = nn.ConvTranspose2d(self.in_channels, self.in_channels, 
                                                   kernel_size=upsample_kernel_size, stride=upsample_stride, padding=upsample_padding)

        # 2. 并行多核分支
        self.branches = nn.ModuleList()
        #self.branch_shifts = nn.ModuleList()
        for k_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=k_size, stride=stride, 
                          padding=k_size // 2, groups=self.in_channels, bias=False), # Depthwise Conv
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True), # Pointwise Conv
                nn.BatchNorm2d(self.in_channels), # BN 放在 DWConv+PWConv 之后
                act_layer(activation, inplace=True) # 激活函数
            )
            self.branches.append(branch)
            #self.branch_shifts.append(shift_module(shift_size=1))
        # 3. 最终的 PWConv，用于调整通道数到 out_channels
        self.final_pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 上采样
        x = self.conv_transpose(x)

        # 2. 并行多核分支处理
        branch_outputs = []
        #shifted_branch_output = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # for i, branch in enumerate(self.branches):
        #     branch_outputs = branch(x)
        #     shifted_output = self.branch_shifts[i](branch_outputs)
        #     shifted_branch_output.append(shifted_output)

        # 3. 分支融合 (相加)
        fused_x = sum(branch_outputs)
        #fused_x = sum(shifted_branch_output)
        if fused_x.shape[1] >= 4 and fused_x.shape[1] % 4 == 0: # 检查通道数是否满足要求
            fused_x = self.shift(fused_x)
        else:
            pass
        #fused_x = self.shift(fused_x)
        # 4. 通道混洗
        fused_x = channel_shuffle(fused_x, self.in_channels)

        # 5. 最终的 PWConv 调整通道
        out = self.final_pwc(fused_x)

        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class LocalGatedLinear(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.GELU):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels, in_channels*2, 1, norm_cfg=None, act_cfg=None
        )
        self.dwconv = ConvModule(
           in_channels, in_channels, 1, groups=in_channels, norm_cfg=None, act_cfg=None
        )
        # self.dwconv = ConvModule(
        #     in_channels, in_channels, 1, padding=1, groups=in_channels, norm_cfg=None, act_cfg=None
        # )

        self.act = act_layer('gelu')
        self.conv2 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=None, act_cfg=None
        )

    def forward(self, x: torch.Tensor):
        x_proj = self.conv1(x)
        x_main, x_gate = x_proj.chunk(2, dim=1)

        x_main_processed = self.dwconv(x_main)
        x_gated = self.act(x_main_processed) * x_gate

        out = self.conv2(x_gated)
        return out
    
class ChannelAttention_LocalGating(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int=16, conv_cfg: dict=None, act_cfg: dict=None):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 替换之前的 MLP，使用 LocalGatedLinear
        # LocalGatedLinear 的 out_channels 应该是 channels
        self.local_gated_mlp = LocalGatedLinear(channels, channels, act_layer=act_layer) # 假设 act_layer 是你的通用激活函数

        # 这里需要一个 MLP 来从融合后的池化结果中学习最终权重
        # 这个 MLP 的输入仍然是 [B, C, 1, 1]，输出是 [B, C, 1, 1]
        self.mlp_for_weights = nn.Sequential(
            ConvModule(
                channels, channels // reduction_ratio, 1, 
                conv_cfg=conv_cfg, norm_cfg=None, act_cfg=act_cfg
            ),
            ConvModule(
                channels // reduction_ratio, channels, 1, 
                conv_cfg=conv_cfg, norm_cfg=None, act_cfg=None
            )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        combined_features = avg_out + max_out
        attention_weights_raw = self.local_gated_mlp(combined_features)

        # processed_x = self.local_gated_mlp(x)
        # avg_out = self.avg_pool(x)
        # max_out = self.max_pool(x)
        # combined_features = avg_out + max_out
        # attention_weights_raw = self.mlp_for_weights(combined_features)

        attention_weights = self.sigmoid(attention_weights_raw)

        filtered_feature_map = x * attention_weights

        return filtered_feature_map


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LMM(nn.Module):
    def __init__(self, channels, kernel_size: List[int]=[3,5], activation: str='gelu'):
        super(LMM, self).__init__()
        self.channels = channels
        dim = self.channels
        self.kernel_size = kernel_size
        self.activation = activation
        # conv
        self.branches = nn.ModuleList()
        for k_size in kernel_size:
            branch = nn.Sequential(
                # DWConv (Depthwise Convolution)
                nn.Conv2d(dim, dim, kernel_size=k_size, stride=1, 
                          padding=k_size // 2, groups=dim, bias=False), 
                #nn.BatchNorm2d(dim), # 对应 DWConv_Block 内部的 BN
                #act_layer(activation, inplace=True), # 对应 DWConv_Block 内部的 Act (注意 inplace 兼容性)
                
                # PWC (Pointwise Convolution)
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True), # bias=True 对应 PWC_Block
                nn.BatchNorm2d(dim), # 对应 PWC_Block 内部的 BN
                act_layer(activation, inplace=True) # 对应 PWC_Block 内部的 Act
            )
            self.branches.append(branch)
    
        self.reweight = Mlp(dim, dim // 8, dim * 3)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        N, C, H, W = x.shape
        #identity = x

        branch_outputs = []
        for branch_module in self.branches:
            branch_outputs.append(branch_module(x))

        x_add = sum(branch_outputs) + x
        att = F.adaptive_avg_pool2d(x_add, output_size=1)
        att_1 = F.adaptive_avg_pool2d(x_add, output_size=1)
        att_total = att + att_1
        att_total = self.reweight(att_total).reshape(N, C, 3).permute(2, 0, 1)
        #att_total = self.swish(att_total).unsqueeze(-1).unsqueeze(-1)
        att_total = F.softmax(att_total, dim=0).unsqueeze(-1).unsqueeze(-1)

        # --- MODIFICATION: 加权融合所有分支和原始 x ---
        # 提取各个分支的权重
        weighted_results = []
        for i in range(len(self.kernel_size)):
            weighted_results.append(branch_outputs[i] * att_total[i])
        
        # 原始 x 的权重是最后一个
        weighted_results.append(x * att_total[len(self.kernel_size)])

        x_att = sum(weighted_results) # 将所有加权后的结果相加
        #out = identity + x_att

        return x_att
        #return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        #self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups=n_feats)
        # 新增
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1 ,1)), requires_grad=True)
    def forward(self, x):
        short_cut = x.clone()
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        #--------------------------
        # a_3x3 = self.DWConv2(a) 
        # a_7x7 = self.DWConv1(a) 
        # a_fused = a_3x3 + a_7x7
        # x = x * a_fused 
        #---------------------------
        x = self.Conv2(x)
        return x * self.scale + short_cut

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, conv_cfg: dict = None, norm_cfg: dict = None, act_cfg: dict = None):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1 # 确保填充正确

        # 通道维度上的平均和最大池化
        # 结果是 [B, 1, H, W]
        self.compress_conv = ConvModule(
            2, 1, kernel_size=kernel_size, stride=1, padding=padding,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None # 最后一层不加激活
        )
        self.conv_path_avg = ConvModule(1, 1, kernel_size=kernel_size, stride=1, padding=padding, 
        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='gelu'))
        self.conv_path_max = ConvModule(1, 1, kernel_size=kernel_size, stride=1, padding=padding,
        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=dict(type='gelu'))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        # 1. 通道维度的平均池化和最大池化
        # 结果形状是 [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        processed_avg = self.conv_path_avg(avg_out)
        processed_max = self.conv_path_max(max_out)
        # 2. 拼接结果
        # [B, 2, H, W]
        #combined = torch.cat([avg_out, max_out], dim=1)
        spatial_attention_map = processed_avg + processed_max
        # 3. 通过卷积生成空间注意力图
        # [B, 1, H, W]
        #spatial_attention_map = self.compress_conv(combined)

        # 4. Sigmoid 激活得到空间权重
        spatial_weights = self.sigmoid(spatial_attention_map)

        # 5. 将权重与原特征图相乘 (逐元素调制)
        filtered_feature_map = x * spatial_weights

        return filtered_feature_map


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, conv_cfg: dict = None, act_cfg: dict = None):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP (或 ConvModule 序列)
        # 输入通道数是 channels，因为 avg_pool 和 max_pool 都是 [B, C, 1, 1]
        # 输出通道数是 channels
        self.mlp = nn.Sequential(
            # 第一个 ConvModule: 降维
            ConvModule(
                channels, channels // reduction_ratio, 1, 
                conv_cfg=conv_cfg, 
                norm_cfg=None, # 通常在注意力分支中不使用 BN
                act_cfg=dict(type='ReLU')
            ),
            # 第二个 ConvModule: 升维
            ConvModule(
                channels // reduction_ratio, channels, 1, 
                conv_cfg=conv_cfg, 
                norm_cfg=None, # 通常在注意力分支中不使用 BN
                act_cfg=None   # 最后一层不加激活，因为后面是 Sigmoid
            )
        )
        
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        # 1. 全局平均池化和全局最大池化
        avg_out = self.avg_pool(x) # [B, C, 1, 1]
        max_out = self.max_pool(x) # [B, C, 1, 1]

        # 2. 通过共享 MLP 学习权重
        # 移除了 1x1 的空间维度，使得 MLP 接受 [B, C]
        # 但 ConvModule 期望 4D 输入，所以不 squeeze，让 ConvModule 自己处理 1x1
        avg_out_processed = self.mlp(avg_out) # [B, C, 1, 1]
        max_out_processed = self.mlp(max_out) # [B, C, 1, 1]
        # 3. 融合池化结果
        # 逐元素相加
        combined_features = avg_out_processed + max_out_processed

        # 4. Sigmoid 激活得到权重
        attention_weights = self.sigmoid(combined_features) # [B, C, 1, 1]

        # 5. 将权重与原特征图相乘 (逐元素调制)
        filtered_feature_map = x * attention_weights # [B, C, H, W]

        return filtered_feature_map


class MaskHead(nn.Module):
    def __init__(self, hidden_dim: int, feats1_stride: int, 
                 upsample_kernel_size: int = 8, upsample_stride: int = 4):
        super().__init__()
        
        self.feats1_stride = feats1_stride
        self.hidden_dim = hidden_dim

        # 1. 第一个卷积模块 (通道不变，空间不变)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.SyncBatchNorm(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 2. 插入 EUCB_MultiKernel 上采样模块
        # 它将 hidden_dim 通道输入，输出 hidden_dim 通道，并进行上采样
        # upsample_factor (例如 4) 和 upsample_kernel_size (例如 8)
        self.upsampler = EUCB_MultiKernel(
            in_channels=hidden_dim,   # 输入通道数是 hidden_dim
            out_channels=hidden_dim,  # 输出通道数仍然是 hidden_dim
            kernel_sizes=[3, 5],      # Depthwise Conv 的核大小
            stride=1,                 # Depthwise Conv 的步长
            upsample_stride=upsample_stride, # 总上采样倍数，例如 4
            upsample_kernel_size=upsample_kernel_size, # 转置卷积核大小
            upsample_padding = 2,
            activation='gelu'         # 激活函数类型
        )

        # 3. 最后一个 1x1 卷积，将通道数从 hidden_dim 降到 1
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # 验证一下 upsample_factor 和 feats1_stride 是否匹配
        if feats1_stride != upsample_stride:
            raise ValueError(f"feats1_stride ({feats1_stride}) must match upsample_factor ({upsample_factor}) "
                             "for MaskHead to correctly resize to img_size.")

    def forward(self, feats1):
        # 输入 feats1 形状: [B, hidden_dim, H_feats1, W_feats1]
        
        # 1. 初始处理
        x = self.conv1(feats1)
        x = self.bn1(x)
        x = self.relu1(x)
        # 此时 x 的形状仍是 [B, hidden_dim, H_feats1, W_feats1]

        # 2. 上采样
        x = self.upsampler(x)
        # 经过 upsampler 后，x 的形状变为 [B, hidden_dim, H_feats1 * upsample_factor, W_feats1 * upsample_factor]
        # 这应该等于 [B, hidden_dim, img_size, img_size]

        # 3. 最终的 1x1 卷积压缩通道
        x = self.final_conv(x)
        # 此时 x 的形状变为 [B, 1, img_size, img_size]

        return x

from einops import rearrange
class DilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias = False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=1),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(1,3), stride=stride, padding=(0,1)),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.edge_enhancer = EdgeEnhancer(in_dim=out_planes, norm=nn.BatchNorm2d) 
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x0,x1,x2], 1)
        out = self.ConvLinear(out)
        short = (self.shortcut(x))
        out = out * self.scale + short
        out = self.relu(out)
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ParallelEdgeFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 norm_layer: nn.BatchNorm2d, 
                 act_layer: nn.GELU, 
                 kernel_sizes: List[int] = [3, 5], # DWConv 并行分支的核大小
                 bias: bool = False,
                 # EdgeEnhancer 相关参数
                 include_edge_branch: bool = True,
                ): 
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_edge_branch = include_edge_branch

        self.dwconv_branches = nn.ModuleList() # 重命名，与 edge_branch 区分
        for k_size in kernel_sizes:
            branch = DWwConv(in_channels, in_channels, kernel_size=k_size, 
                            stride=1, padding=k_size // 2, bias=bias, 
                            norm_layer=norm_layer, act_layer=act_layer)
            self.dwconv_branches.append(branch)
        
        # 边缘增强并行分支
        self.edge_branch = nn.Identity() # 默认不进行额外处理
        if self.include_edge_branch:
            # EdgeEnhancer 的 in_dim 应该与输入 in_channels 匹配
            self.edge_branch = EdgeEnhancer(in_dim=in_channels, norm=norm_layer)
        
        # 融合所有分支输出的 1x1 卷积
        # 拼接后的通道数 = len(kernel_sizes) * in_channels + (in_channels if include_edge_branch else 0)
        total_concat_channels = len(kernel_sizes) * in_channels
        if self.include_edge_branch:
            total_concat_channels += in_channels # 加上 EdgeEnhancer 分支的通道

        self.fusion_conv_1x1 = nn.Conv2d(total_concat_channels, out_channels, 
                                          kernel_size=1, stride=1, bias=bias)
        
        # 残差连接的投影
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, H, W]

        # 1. DWConv 并行分支处理
        branch_outputs = []
        for branch in self.dwconv_branches:
            branch_outputs.append(branch(x)) 
        
        # 2. 边缘增强并行分支处理
        if self.include_edge_branch:
            edge_output = self.edge_branch(x) # 边缘分支的输出
            branch_outputs.append(edge_output) # 将边缘分支添加到输出列表
        
        # 3. 分支融合 (拼接)
        fused_features = torch.cat(branch_outputs, dim=1) 

        # 4. 融合后的 1x1 卷积
        out_main_path = self.fusion_conv_1x1(fused_features) # [B, out_channels, H, W]

        # 5. 残差连接 (将原始输入 x 投影到 out_channels 后与增强后的主路径相加)
        out = out_main_path + self.shortcut(x) 
        
        return out


# --- DWConv 模块 ---
class DWwConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, # kernel_size 是位置参数
                 norm_layer: nn.BatchNorm2d, # 传入的是类引用，例如 nn.BatchNorm2d
                 act_layer: nn.ReLU, # 传入的是类引用，例如 nn.ReLU 或 nn.GELU
                 stride: int = 1, padding: int = None, 
                 dilation: int = 1, bias: bool = False, 
                 inplace_act: bool = True): # 激活函数是否原地操作
        super().__init__()
        
        # 确保 padding 被正确设置
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        
        # 深度卷积层：输入通道和输出通道都是 in_channels，groups=in_channels
        # 严格意义上，深度卷积层不会改变通道数
        # 如果你希望 DWConv 可以改变通道数，那么在深度卷积后需要一个 1x1 的逐点卷积 (pointwise conv)
        # 但在你的 ParallelDWConvFusion 语境中，DWConv 通常只做空间操作，不改变通道数
        # 所以 out_channels 在这里应该和 in_channels 相同，或者至少 DWConv 层本身输出 in_channels
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, 
                                        kernel_size=kernel_size, stride=stride, 
                                        padding=padding, dilation=dilation, 
                                        groups=in_channels, # <--- 深度卷积的关键
                                        bias=bias)
        
        # 归一化层
        self.norm = norm_layer(in_channels) if norm_layer else nn.Identity()
        self.act = act_layer()

        # 如果 in_channels != out_channels，你需要一个额外的 1x1 卷积来投影通道
        # 在 ParallelDWConvFusion 中，每个分支的 in_channels 和 out_channels 都应该是相同的 (in_channels)
        # 所以这里通常不需要额外的投影
        if in_channels != out_channels:
            raise ValueError(f"DWConv is typically channel-preserving. Expected in_channels == out_channels, but got {in_channels} != {out_channels}. "
                             "If you intend to change channels, please add a pointwise convolution explicitly.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, conv_cfg: dict = None, act_cfg: dict = None):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp_for_weights = GSAU(n_feats=channels)
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        # 1. 全局平均池化和全局最大池化
        avg_out = self.avg_pool(x) # [B, C, 1, 1]
        max_out = self.max_pool(x) # [B, C, 1, 1]

        combined_features = avg_out + max_out 
        # 3. 融合池化结果
        # 逐元素相加
        #combined_features = avg_out_processed + max_out_processed

        # 4. Sigmoid 激活得到权重
        attention_weights_raw = self.mlp_for_weights(combined_features)
        attention_weights = self.sigmoid(attention_weights_raw)
        #attention_weights = self.sigmoid(combined_features) # [B, C, 1, 1]

        # 5. 将权重与原特征图相乘 (逐元素调制)
        filtered_feature_map = x * attention_weights # [B, C, H, W]

        return filtered_feature_map

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4):
        super().__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        #print(f"FusionConv init: dim for channel_attention is {dim}")
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x_fused = self.down(x)
        res = x_fused
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        
        x_fused_c = self.channel_attention(x_fused)
        #x_out = self.up(x_fused_s) * x_fused_c + res
        x_out = self.up(x_fused_c + x_fused_s + res)
        
        return x_out


import numbers
class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8
        
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        
        x=x[:,:,:h,:w]
        
        return x

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    
class LayerNorm2(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class EVS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(EVS, self).__init__()
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)
        self.norm = LayerNorm2(dim)
    def forward(self,x):
        x = x + self.ffn(self.norm(x))
        return x


import scipy.fftpack

def dct_2d(x):
    x = x.cpu().numpy()
    return torch.tensor(scipy.fftpack.dct(scipy.fftpack.dct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')).to(torch.float32)

def idct_2d(x):
    x = x.cpu().numpy()
    return torch.tensor(scipy.fftpack.idct(scipy.fftpack.idct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')).to(torch.float32)

def batch_dct(x):
    B, C, H, W = x.shape
    return torch.stack([
        torch.stack([
            dct_2d(x[b, c]) for c in range(C)
        ]) for b in range(B)
    ])

def batch_idct(x):
    B, C, H, W = x.shape
    return torch.stack([
        torch.stack([
            idct_2d(x[b, c]) for c in range(C)
        ]) for b in range(B)
    ])

def high_pass_filter(dct_map, threshold=0.3):
    B, C, H, W = dct_map.shape
    mask = torch.ones_like(dct_map)
    cx, cy = int(H * threshold), int(W * threshold)
    mask[:, :, :cx, :cy] = 0
    return dct_map * mask

class HighFrequencyEnhancementFFT(nn.Module):
    def __init__(self, in_channels, threshold=0.3):
        super().__init__()
        self.threshold = threshold
        self.conv_spatial = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) # 修正输入通道
        self.final_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) # 修正输入通道

        self.ca_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1), # 降维
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1) # 升维
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. FFT -> 高频 -> iFFT
        fft_map = torch.fft.fft2(x, norm='ortho')
        fft_map_shifted = torch.fft.fftshift(fft_map, dim=(-2, -1))

        # 创建高通滤波器掩码
        mask = torch.zeros_like(fft_map_shifted)
        cx, cy = H // 2, W // 2 # 频谱中心
        th_h, th_w = int(cy * self.threshold), int(cx * self.threshold)
        mask[:, :, cy - th_h : cy + th_h, cx - th_w : cx + th_w] = 1 # 低通区域为 1
        mask = 1 - mask # 反转为高通

        high_freq_fft = fft_map_shifted * mask
        high_freq_fft = torch.fft.ifftshift(high_freq_fft, dim=(-2, -1))
        hf_feat = torch.fft.ifft2(high_freq_fft, norm='ortho').real # 取实部

        # 2. Channel & Spatial Attention (逻辑不变)
        gap = F.adaptive_avg_pool2d(hf_feat, 1)
        gmp = F.adaptive_max_pool2d(hf_feat, 1)
        #ch_attn = torch.sigmoid(self.conv_channel(gap + gmp))
        ch_attn_logits = self.ca_mlp(gap) + self.ca_mlp(gmp)
        ch_attn = torch.sigmoid(ch_attn_logits)
        x_cp = x * ch_attn

        # sp_attn = torch.sigmoid(self.conv_spatial(hf_feat))
        # x_sp = x * sp_attn
        
        # 4. 融合 (可以尝试不同的融合方式)
        # 方案 A: 原始 + 通道增强 + 空间增强
        out = self.final_conv(x_cp) + self.final_conv2(x)
        
        # 方案 B: 将高频特征也作为输入
        #out = self.final_conv(torch.cat([x, hf_feat], dim=1)) # 拼接原始特征和高频特征

        return out


class HighFrequencyFFT(nn.Module):
    def __init__(self, in_channels, threshold=0.3, k=16, reduction_ratio=16):
        super().__init__()
        self.threshold = threshold
        self.conv_channel = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.ca_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1), # 降维
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1) # 升维
        )
        self.ca_mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1), # 降维
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1) # 升维
        )
        self.gap = nn.AdaptiveAvgPool2d((k, k))
        self.gmp = nn.AdaptiveMaxPool2d((k, k))
        self.relu = nn.ReLU()
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.mlp_gap = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.mlp_gap_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.mlp_gmp_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.mlp_gmp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(in_channels // 16 * 2, in_channels),
            nn.Sigmoid()
        )
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)    

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. FFT -> 高频 -> iFFT
        fft_map = torch.fft.fft2(x, norm='ortho')
        fft_map_shifted = torch.fft.fftshift(fft_map, dim=(-2, -1))

        # 创建高通滤波器掩码
        mask = torch.zeros_like(fft_map_shifted)
        cx, cy = H // 2, W // 2 # 频谱中心
        th_h, th_w = int(cy * self.threshold), int(cx * self.threshold)
        mask[:, :, cy - th_h : cy + th_h, cx - th_w : cx + th_w] = 1 # 低通区域为 1
        mask = 1 - mask # 反转为高通

        high_freq_fft = fft_map_shifted * mask
        high_freq_fft = torch.fft.ifftshift(high_freq_fft, dim=(-2, -1))
        hf_feat = torch.fft.ifft2(high_freq_fft, norm='ortho').real # 取实部

        # 2. Channel & Spatial Attention (逻辑不变)
        gap_feat = self.relu(self.gap(hf_feat)) # (B, C, k, k)
        gmp_feat = self.relu(self.gmp(hf_feat)) # (B, C, k, k)
        
        # b. "sum across each channel to generate two one-dimensional feature vectors"
        vec_gap = gap_feat.sum(dim=(-2, -1)) # (B, C)
        vec_gmp = gmp_feat.sum(dim=(-2, -1)) # (B, C)
        
        # c. "passed through separate 1x1 group convolutions"
        score_gap = self.mlp_gap(vec_gap) # (B, hidden)
        score_gmp = self.mlp_gmp(vec_gmp) # (B, hidden)
        
        # d. "concatenated and passed through another 1x1 group convolution"
        combined_scores = torch.cat([score_gap, score_gmp], dim=1) # (B, 2*hidden)
        channel_weights = self.final_mlp(combined_scores) # (B, C)
        
        # e. 应用通道注意力
        x_cp = x * channel_weights.unsqueeze(-1).unsqueeze(-1)

        # --- Spatial Path (SP) ---
        spatial_mask = self.conv_spatial(hf_feat)
        x_sp = x * spatial_mask
        
        # --- 最终融合 ---
        fused_feat = x_cp + x_sp
        out = self.final_conv(fused_feat)
        
        return out 

from pytorch_wavelets import DWTForward, DWTInverse

class HighFrequencyEnhancementWavelet(nn.Module):
    def __init__(self, in_channels, wavelet='haar', threshold=0.3):
        """
        使用离散小波变换 (DWT) 提取高频特征。

        Args:
            in_channels (int): 输入通道数。
            wavelet (str): 使用的小波基函数，例如 'haar', 'db1', 'db2' 等。
            threshold (float, optional): 如果提供，可以对高频子带进行软阈值去噪。
        """

        super().__init__()
        self.in_channels = in_channels
        self.threshold = threshold

        # 初始化前向和反向小波变换模块
        # J=1 表示只进行一级分解
        # mode='zero' 表示边界填充方式
        self.dwt = DWTForward(J=1, mode='zero', wave=wavelet)
        self.idwt = DWTInverse(mode='zero', wave=wavelet)
        reduced_channels = in_channels // 4 # 确保 in_channels 是 3 的倍数，或者调整
        
        self.lh_proj = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.hl_proj = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.hh_proj = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        # 注意力生成部分的模块 (与之前类似)
        hf_feat_channels = reduced_channels * 3
        self.conv_spatial = nn.Conv2d(hf_feat_channels, 1, kernel_size=1) # 输入是拼接后的 LH, HL, HH

        self.ca_mlp = nn.Sequential(
            nn.Conv2d(hf_feat_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1)
        )
        
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x 形状: (B, C, H, W)

        # 1. DWT 分解
        # Yl: 低频近似 (LL), 形状 (B, C, H/2, W/2)
        # Yh: 高频细节列表，包含 [LH, HL, HH]
        # 每个高频子带的形状是 (B, C, H/2, W/2)
        Yl, Yh = self.dwt(x)
        
        # 提取三个高频子带
        lh, hl, hh = Yh[0][:,:,0,:,:], Yh[0][:,:,1,:,:], Yh[0][:,:,2,:,:]
        
        # (可选) 对高频子带进行软阈值去噪
        if self.threshold is not None:
            lh = F.softshrink(lh, self.threshold)
            hl = F.softshrink(hl, self.threshold)
            hh = F.softshrink(hh, self.threshold)
            
        # 将三个高频子带拼接起来，作为“高频特征”
        # hf_feat 形状: (B, C*3, H/2, W/2)
        lh = self.lh_proj(lh) # (B, C/3, H/2, W/2)
        hl = self.hl_proj(hl) # (B, C/3, H/2, W/2)
        hh = self.hh_proj(hh) # (B, C/3, H/2, W/2)
        hf_feat = torch.cat([lh, hl, hh], dim=1)

        # 2. Channel & Spatial Attention
        # 注意：现在 hf_feat 的尺寸是 (H/2, W/2)，通道数是 C*3
        
        # Channel Attention
        # 对 hf_feat 进行池化
        gap = F.adaptive_avg_pool2d(hf_feat, 1)
        gmp = F.adaptive_max_pool2d(hf_feat, 1)
        # ca_mlp 的输入通道数需要是 C*3
        ch_attn_logits = self.ca_mlp(gap) + self.ca_mlp(gmp)
        ch_attn = torch.sigmoid(ch_attn_logits) # (B, C, 1, 1)
        x_cp = x * ch_attn
        
        # Spatial Attention
        # conv_spatial 的输入通道数需要是 C*3
        # 并且其输出的空间注意力图尺寸是 H/2, W/2
        # 我们需要将其上采样以匹配原始 x 的尺寸
        sp_attn_low_res = torch.sigmoid(self.conv_spatial(hf_feat)) # (B, 1, H/2, W/2)
        sp_attn = F.interpolate(sp_attn_low_res, size=x.shape[2:], mode='bilinear', align_corners=True)
        x_sp = x * sp_attn
        
        # 3. 融合
        out = self.final_conv(x_cp + x_sp)
        
        return out


class Adaptive_GSAU_ChannelAttention(nn.Module):
    def __init__(self, channels: int, 
                 reduction_ratio = 16,
                 target_gsau_size: int = 8 # GSAU 期望的最佳输入尺寸
                ):
        """
        一个自适应的、基于 GSAU 的通道注意力模块。
        它会根据输入特征图的尺寸，动态计算局部池化的大小，
        以确保 GSAU 作用在一个接近 target_gsau_size 的特征图上。
        GSAU 在这里完全替代了传统的 MLP 来生成通道权重。
        """
        super().__init__()
        
        self.target_gsau_size = target_gsau_size
        
        # 1. GSAU 模块
        self.gsau_processor = GSAU(n_feats=channels)
        
        # 2. 全局池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 3. 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # --- 步骤 1: 动态计算池化参数并进行局部池化 ---
        # 目标是 H_in / pool_kernel_size ≈ target_gsau_size
        if H > self.target_gsau_size:
            # `max(1, ...)` 确保池化核至少为 1
            # 使用 round() 来找到最接近的整数池化核大小
            pool_kernel_size_h = max(1, round(H / self.target_gsau_size))
            pool_kernel_size_w = max(1, round(W / self.target_gsau_size))
            
            # 使用 F.avg_pool2d 和 F.max_pool2d 进行动态池化
            # 我们需要 kernel_size 和 stride 都等于计算出的值
            local_avg_out = F.avg_pool2d(x, 
                                         kernel_size=(pool_kernel_size_h, pool_kernel_size_w), 
                                         stride=(pool_kernel_size_h, pool_kernel_size_w))
            local_max_out = F.max_pool2d(x, 
                                         kernel_size=(pool_kernel_size_h, pool_kernel_size_w), 
                                         stride=(pool_kernel_size_h, pool_kernel_size_w))
            
            feature_for_gsau = local_avg_out + local_max_out
        else:
            # 如果输入特征图已经小于或等于目标尺寸，则不进行局部池化
            feature_for_gsau = x
        
        # 步骤 2: GSAU 增强
        gsau_out = self.gsau_processor(feature_for_gsau)
        
        # 步骤 3: 全局池化，提取通道描述符
        global_context = self.global_avg_pool(gsau_out) + self.global_max_pool(gsau_out)
        
        # 步骤 4: Sigmoid 激活得到最终权重
        attention_weights = self.sigmoid(global_context)
        
        # 步骤 5: 应用权重
        return x * attention_weights


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels)

class WFD(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.dwt = HaarWavelet(channels, grad=False)
        self.dim = channels

    def forward(self, x):
        haar = self.dwt(x, rev=False)
        a = haar.narrow(1, 0, self.dim) 
        h = haar.narrow(1, self.dim, self.dim) 
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)
        return a, h+v+d

class WFD_Fusion_Block(nn.Module): # <--- 我们用这个名字来代表你说的 WFD 融合模块
    def __init__(self, channels):
        super().__init__()
        
        # 1. 在内部使用 WaveletDecomposition
        self.decomposition = WFD(channels=channels)

        # 2. 融合层
        # 接收拼接后的特征 (来自 x_h, a_l, d_l)
        self.fusion_conv = nn.Conv2d(channels * 3, channels, kernel_size=1)
        
    def forward(self, x_l: torch.Tensor, x_h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_l (torch.Tensor): 空间分辨率更高的特征 (要被分解的)
            x_h (torch.Tensor): 空间分辨率更低的特征
        """
        # 1. 对高分辨率特征 x_l 进行分解
        # a_l, d_l 的尺寸都与 x_h 匹配
        a_l, d_l = self.decomposition(x_l)
        
        # 2. 将 x_h 与分解后的 a_l, d_l 拼接
        fused_feat = torch.cat([x_h, a_l, d_l], dim=1)
        
        # 3. 通过 1x1 卷积进行融合
        out = self.fusion_conv(fused_feat)
        
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 辅助模块：可微分的 2D-DCT ---
# 这个模块对于整个设计的可训练性至关重要
class Dct2d(nn.Module):
    def __init__(self, norm='ortho'):
        super().__init__()
        self.norm = norm

    def dct_1d(self, x):
        """ Discrete Cosine Transform 1D """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        # Use torch.fft.fft for differentiability
        Vc = torch.fft.fft(v)
        k = -torch.arange(N, device=x.device, dtype=x.dtype) * (torch.pi / (2 * N))
        W_k = torch.exp(1j * k)
        V = Vc * W_k
        V = 2 * V.real
        if self.norm == 'ortho':
            V[:, 0] /= torch.sqrt(torch.tensor(N)) * 2
            V[:, 1:] /= torch.sqrt(torch.tensor(N / 2)) * 2
        return V.view(*x_shape)

    def idct_1d(self, X):
        """ Inverse Discrete Cosine Transform 1D """
        x_shape = X.shape
        N = x_shape[-1]
        X_v = X.contiguous().view(-1, N) / 2
        if self.norm == 'ortho':
            X_v[:, 0] *= torch.sqrt(torch.tensor(N))
            X_v[:, 1:] *= torch.sqrt(torch.tensor(N / 2))
        k = torch.arange(N, device=X.device, dtype=X.dtype) * (torch.pi / (2 * N))
        W_k = torch.exp(1j * k)
        V_k = X_v * W_k
        # Use torch.fft.ifft
        v_k = torch.fft.ifft(V_k)
        x = torch.zeros_like(X_v)
        x[:, ::2] = v_k[:, :N - N // 2].real
        x[:, 1::2] = v_k.flip([1])[:, :N // 2].real
        return x.view(*x_shape)
        
    def forward(self, x, rev=False):
        if not rev:
            # DCT-2D is separable, apply 1D DCT on rows then columns
            x = self.dct_1d(x)
            x = self.dct_1d(x.transpose(-1, -2)).transpose(-1, -2)
            return x
        else:
            # iDCT-2D
            x = self.idct_1d(x)
            x = self.idct_1d(x.transpose(-1, -2)).transpose(-1, -2)
            return x

# --- 2. 补全你的模块 ---

class DctSpatialInteraction(nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.dct_transform = Dct2d(norm='ortho')

    def _compute_weight(self, H, W, ratio):
        """ Creates a high-pass or low-pass filter mask. """
        # 这是一个高通滤波器，与你之前的 HFE 类似
        mask = torch.ones(H, W)
        height_cutoff = int(H * ratio)
        width_cutoff = int(W * ratio)
        if height_cutoff > 0 and width_cutoff > 0:
            mask[:height_cutoff, :width_cutoff] = 0 # Low-frequency area set to 0
        return mask

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. DCT
        dct_map = self.dct_transform(x) # [B, C, H, W]
        
        # 2. Compute and apply weight
        # 权重是 2D 的，需要在 B 和 C 维度上广播
        weight = self._compute_weight(H, W, self.ratio).to(x.device)
        weight = weight.view(1, 1, H, W) # expand_as is not needed with broadcasting
        
        dct_filtered = dct_map * weight
        
        # 3. iDCT
        spatial_attention_map = self.dct_transform(dct_filtered, rev=True)
        
        # 4. Modulate input feature
        # 这里的 spatial_attention_map 包含了增强后的高频信息
        # 通常会用 sigmoid 激活，并与原始 x 相乘
        return x * torch.sigmoid(spatial_attention_map)


class DctChannelInteraction(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_size=(4, 4), ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.h, self.w = pool_size
        self.ratio = ratio
        self.dct_transform = Dct2d(norm='ortho')
        self.relu = nn.ReLU()

        # 1x1 卷积用于通道融合
        self.channel1x1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.channel2x1 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def _compute_weight(self, H, W, ratio):
        mask = torch.ones(H, W)
        height_cutoff = int(H * ratio)
        width_cutoff = int(W * ratio)
        if height_cutoff > 0 and width_cutoff > 0:
            mask[:height_cutoff, :width_cutoff] = 0
        return mask

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. DCT
        dct_map = self.dct_transform(x)
        
        # 2. Compute and apply weight
        weight = self._compute_weight(H, W, self.ratio).to(x.device)
        weight = weight.view(1, 1, H, W)
        dct_filtered = dct_map * weight
        
        # 3. iDCT
        hf_feat = self.dct_transform(dct_filtered, rev=True)
        
        # 4. Pool and generate channel weights
        amaxp = F.adaptive_max_pool2d(hf_feat, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(hf_feat, output_size=(self.h, self.w))
        
        amaxp_relu = self.relu(amaxp)
        aavgp_relu = self.relu(aavgp)
        
        # c. 空间维度求和，得到 (B, C) 的向量
        sum_amaxp = torch.sum(amaxp_relu, dim=(-2, -1))
        sum_aavgp = torch.sum(aavgp_relu, dim=(-2, -1))
        sum_amaxp_4d = sum_amaxp.unsqueeze(-1).unsqueeze(-1)
        sum_aavgp_4d = sum_aavgp.unsqueeze(-1).unsqueeze(-1)

        # 将池化结果送入 MLP
        amaxp_vec = self.channel1x1(sum_amaxp_4d)
        aavgp_vec = self.channel1x1(sum_amaxp_4d)
        
        channel_logits = self.channel2x1((amaxp_vec + aavgp_vec))
        
        # 5. Modulate input feature
        return x * torch.sigmoid(channel_logits)


class high_freq_perception_module(nn.Module):
    def __init__(self, in_channels, ratio_spatial=0.5, ratio_channel=0.5):
        super().__init__()
        
        # 1. 实例化空间和通道交互模块
        self.spatial = DctSpatialInteraction(ratio=ratio_spatial)
        self.channel = DctChannelInteraction(in_channels=in_channels, ratio=ratio_channel)
        
        # 2. 最后的融合层
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, in_channels), # GroupNorm 的 num_groups 需要能整除 in_channels
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 1. 并行计算
        spatial_enhanced = self.spatial(x)
        channel_enhanced = self.channel(x)
        
        # 2. 融合并输出
        # 将两个增强后的特征相加
        fused = spatial_enhanced + channel_enhanced
        
        # 通过最后的卷积层
        out = self.out(fused)
        
        # (可选) 添加残差连接
        return fused

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. FFT 版本的空间注意力 ---

class FftSpatialInteraction(nn.Module):
    def __init__(self, ratio_inner=0.2, ratio_outer=0.8):
        super().__init__()
        self.ratio_inner = ratio_inner  # 比如 0.2：中心低频屏蔽
        self.ratio_outer = ratio_outer  # 比如 0.8：外围高频屏蔽

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. FFT
        fft_map = torch.fft.fft2(x, norm='ortho')
        fft_map_shifted = torch.fft.fftshift(fft_map, dim=(-2, -1))

        # 2. 创建中频掩码
        mask = torch.zeros_like(fft_map_shifted)  # 先全设为 0
        center_h, center_w = H // 2, W // 2
        half_h_inner = int(center_h * self.ratio_inner)
        half_w_inner = int(center_w * self.ratio_inner)
        half_h_outer = int(center_h * self.ratio_outer)
        half_w_outer = int(center_w * self.ratio_outer)

        # 设置中频区域为 1（一个中间环带）
        mask[:, :, center_h - half_h_outer : center_h + half_h_outer,
                  center_w - half_w_outer : center_w + half_w_outer] = 1
        mask[:, :, center_h - half_h_inner : center_h + half_h_inner,
                  center_w - half_w_inner : center_w + half_w_inner] = 0

        # 3. 掩码滤波
        fft_filtered = fft_map_shifted * mask
        fft_filtered_ishifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        spatial_attention_map = torch.fft.ifft2(fft_filtered_ishifted, norm='ortho').real

        # 4. Modulate input feature
        return x * torch.sigmoid(spatial_attention_map)



# --- 2. FFT 版本的通道注意力 ---

class FftChannelInteraction(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, ratio_inner=0.2, ratio_outer=0.8):
        super().__init__()
        self.in_channels = in_channels
        #self.h, self.w = pool_size
        self.ratio_inner = ratio_inner
        self.ratio_outer = ratio_outer
        self.relu = nn.ReLU()

        hidden_features = in_channels // reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. FFT → 中频筛选 → iFFT
        fft_map = torch.fft.fft2(x, norm='ortho')
        fft_map_shifted = torch.fft.fftshift(fft_map, dim=(-2, -1))

        mask = torch.zeros_like(fft_map_shifted)
        center_h, center_w = H // 2, W // 2

        half_h_inner = int(center_h * self.ratio_inner)
        half_w_inner = int(center_w * self.ratio_inner)
        half_h_outer = int(center_h * self.ratio_outer)
        half_w_outer = int(center_w * self.ratio_outer)

        # 创建中频保留区域（环形）
        mask[:, :, center_h - half_h_outer:center_h + half_h_outer,
                    center_w - half_w_outer:center_w + half_w_outer] = 1
        mask[:, :, center_h - half_h_inner:center_h + half_h_inner,
                    center_w - half_w_inner:center_w + half_w_inner] = 0

        fft_filtered = fft_map_shifted * mask
        fft_filtered_ishifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        hf_feat = torch.fft.ifft2(fft_filtered_ishifted, norm='ortho').real

        # 2. 通道注意力：对中频特征池化 + MLP
        amaxp = F.adaptive_max_pool2d(hf_feat, output_size=(1, 1))  # [B, C, 1, 1]
        aavgp = F.adaptive_avg_pool2d(hf_feat, output_size=(1, 1))

        amaxp_relu = self.relu(amaxp)
        aavgp_relu = self.relu(aavgp)

        sum_amaxp = amaxp.view(B, C)
        sum_aavgp = aavgp.view(B, C)

        channel_scores = sum_amaxp + sum_aavgp  # [B, C]
        channel_logits = self.mlp(channel_scores)  # [B, C]

        channel_weights = self.sigmoid(channel_logits).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 3. Modulate input
        return x * channel_weights


# --- 3. 顶层模块 (现在基于 FFT) ---

class high_freq_perception_module_fft(nn.Module):
    def __init__(self, in_channels, 
                 ratio_spatial_inner=0.3, ratio_spatial_outer=0.7,
                 ratio_channel_inner=0.3, ratio_channel_outer=0.7,
                 num_groups_norm=16):
        super().__init__()
        
        # ✅ 1. 空间注意力模块：中频保留
        self.spatial = FftSpatialInteraction(
            ratio_inner=ratio_spatial_inner, 
            ratio_outer=ratio_spatial_outer
        )
        
        # ✅ 2. 通道注意力模块：中频保留 + 全局 GAP/GMP
        self.channel = FftChannelInteraction(
            in_channels=in_channels, 
            ratio_inner=ratio_channel_inner, 
            ratio_outer=ratio_channel_outer
        )
        
        # ✅ 3. 处理 GroupNorm 的可整除性
        if in_channels % num_groups_norm != 0:
            num_groups_norm = 1 
            while in_channels % (num_groups_norm * 2) == 0 and num_groups_norm * 2 <= 32:
                num_groups_norm *= 2

        # ✅ 4. 融合卷积输出（可换成更强的 block）
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups_norm, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. 并行两个高频增强模块
        spatial_enhanced = self.spatial(x)
        channel_enhanced = self.channel(x)
        
        # 2. 融合 + 卷积
        fused = spatial_enhanced + channel_enhanced
        out = self.out(fused)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class FftBandInteraction(nn.Module):
    """
    一个通用的频带交互模块，可以配置为低通、带通或高通滤波器。
    它只处理空间注意力部分。
    """
    def __init__(self, mode='mid', ratio_inner=0.2, ratio_outer=0.8):
        """
        Args:
            mode (str): 'low', 'mid', or 'high'.
            ratio_inner (float): 内环半径比例.
            ratio_outer (float): 外环半径比例.
        """
        super().__init__()
        assert mode in ['low', 'mid', 'high'], "Mode must be 'low', 'mid', or 'high'"
        self.mode = mode
        self.ratio_inner = ratio_inner
        self.ratio_outer = ratio_outer

    def forward(self, x):
        B, C, H, W = x.shape
        fft_map_shifted = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        
        mask = torch.zeros_like(x, dtype=torch.float32)
        center_h, center_w = H // 2, W // 2

        if self.mode == 'low':
            # 低通：只保留中心区域
            h_r, w_r = int(center_h * self.ratio_outer), int(center_w * self.ratio_outer)
            mask[:, :, center_h - h_r : center_h + h_r, center_w - w_r : center_w + w_r] = 1
        elif self.mode == 'high':
            # 高通：只保留外围区域
            h_r, w_r = int(center_h * self.ratio_inner), int(center_w * self.ratio_inner)
            mask[:, :, :, :] = 1
            mask[:, :, center_h - h_r : center_h + h_r, center_w - w_r : center_w + w_r] = 0
        elif self.mode == 'mid':
            # 带通（中频）：保留一个环带
            h_r_outer, w_r_outer = int(center_h * self.ratio_outer), int(center_w * self.ratio_outer)
            h_r_inner, w_r_inner = int(center_h * self.ratio_inner), int(center_w * self.ratio_inner)
            mask[:, :, center_h - h_r_outer : center_h + h_r_outer, center_w - w_r_outer : center_w + w_r_outer] = 1
            mask[:, :, center_h - h_r_inner : center_h + h_r_inner, center_w - w_r_inner : center_w + w_r_inner] = 0
        
        # 掩码滤波和逆变换
        # 注意：这里我们让掩码作用于复数张量上
        fft_filtered = fft_map_shifted * mask.to(fft_map_shifted.device)
        spatial_attention_map = torch.fft.ifft2(torch.fft.ifftshift(fft_filtered, dim=(-2, -1)), norm='ortho').real
        
        # 返回增强后的特征，而不是原始特征的调制
        # 这样更方便后续融合
        return x * torch.sigmoid(spatial_attention_map)

class MultiBandPerceptionModule(nn.Module):
    def __init__(self, in_channels, num_groups_norm=16):
        super().__init__()
        
        # --- 1. 定义三个并行的空间频带分支 ---
        self.spatial_low = FftBandInteraction(mode='low', ratio_outer=0.2)
        self.spatial_mid = FftBandInteraction(mode='mid', ratio_inner=0.2, ratio_outer=0.8)
        self.spatial_high = FftBandInteraction(mode='high', ratio_inner=0.8)
        
        # --- 2. 学习如何融合三个空间分支的输出 ---
        # 我们将三个分支的输出拼接起来，然后通过一个1x1卷积进行融合
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # --- 3. 定义通道注意力模块 (我们保持原样，因为它内部已经很强大) ---
        # 为了简化，我们暂时只使用原始的中频通道注意力
        # 也可以按照空间分支的思路，为通道注意力也创建多频带版本
        self.channel_attention = FftChannelInteraction(
            in_channels=in_channels,
            ratio_inner=0.2,
            ratio_outer=0.8
        )
        
        # --- 4. 最终的融合层 ---
        # 处理 GroupNorm 的可整除性
        if in_channels % num_groups_norm != 0:
            num_groups_norm = 1 
            while in_channels % (num_groups_norm * 2) == 0 and num_groups_norm * 2 <= 32:
                num_groups_norm *= 2
                
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups_norm, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- 空间多频带处理 ---
        s_low = self.spatial_low(x)
        s_mid = self.spatial_mid(x)
        s_high = self.spatial_high(x)
        
        # 将三个空间特征拼接
        s_combined = torch.cat([s_low, s_mid, s_high], dim=1) # -> [B, C*3, H, W]
        
        # 通过1x1卷积进行自适应融合
        spatial_enhanced = self.spatial_fusion(s_combined) # -> [B, C, H, W]
        
        # --- 通道注意力处理 ---
        channel_enhanced = self.channel_attention(x)
        
        # --- 最终融合 ---
        # 融合经过多频带空间增强的特征和通道增强的特征
        fused = spatial_enhanced + channel_enhanced
        
        # 通过最后的卷积层
        out = self.out_conv(fused)
        
        return out

class EnhancedChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, 
                 shift_size: int = 1, shuffle_groups: int = 4, 
                 conv_cfg: dict = None, act_cfg: dict = None):
        super(EnhancedChannelAttention, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # --- 新增的模块 ---
        # 确保通道数可以被4整除以使用 shift_module
        # 确保通道数可以被 shuffle_groups 整除
        if channels % 4 != 0:
            # 如果不能被4整除，则不使用shift，或者调整设计
            self.shifter = nn.Identity()
            print(f"Warning: Channels ({channels}) not divisible by 4. Disabling shift_module.")
        else:
            self.shifter = shift_module(shift_size)
            
        self.shuffle_groups = shuffle_groups
        # -------------------

        # 原始的注意力计算组件
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            ConvModule(channels, channels // reduction_ratio, 1, conv_cfg=conv_cfg, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(channels // reduction_ratio, channels, 1, conv_cfg=conv_cfg, norm_cfg=None, act_cfg=None)
        )
        # 注意：这里我简化了原始代码，因为你的原始代码有两个MLP，看起来有些冗余
        self.mlp_for_weights = GSAU(n_feats=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存原始输入，用于最终的乘法
        identity = x

        # --- 核心修改：在计算权重前，对输入特征进行增强 ---
        # 1. 应用空间位移
        x_shifted = self.shifter(x)
        
        # 2. 应用通道重排
        x_shuffled = channel_shuffle(x_shifted, self.shuffle_groups)
        
        # --- 使用增强后的特征 x_shuffled 来计算注意力权重 ---
        # 3. 全局池化
        avg_out = self.avg_pool(x_shuffled)
        max_out = self.max_pool(x_shuffled)

        # 4. 通过共享 MLP 学习权重
        #avg_out_processed = self.mlp(avg_out)
        #max_out_processed = self.mlp(max_out)

        # 5. 融合池化结果并激活
        attention_weights = self.sigmoid(self.mlp_for_weights(avg_out + max_out))

        # --- 将计算出的“智能”权重应用到原始输入上 ---
        # 6. 将权重与原始特征图相乘
        filtered_feature_map = identity * attention_weights

        return filtered_feature_map