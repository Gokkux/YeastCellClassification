# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from .EUCB import high_freq_perception_module_fft, high_freq_perception_module, Adaptive_GSAU_ChannelAttention, HighFrequencyEnhancementWavelet, FusionConv, EUCB_MultiKernel, LMM, GSAU, DilatedMDTA, EdgeEnhancer, FEM, ParallelEdgeFusion, EVS, HighFrequencyEnhancementFFT, HighFrequencyFFT

class AdaptiveFPNFusionBlock(nn.Module):
    """
    Adaptive Feature Pyramid Network Fusion Block.
    This block adaptively fuses two feature maps (lateral and top-down)
    using a channel-wise attention mechanism, inspired by MFEblock.
    """
    def __init__(self, in_channels: int, out_channels: int, conv_cfg: dict = None, norm_cfg: dict = None, act_cfg: dict = None):
        super(AdaptiveFPNFusionBlock, self).__init__()
        
        # 确保输入输出通道数一致，因为这里是融合后直接相加到当前层
        assert in_channels == out_channels, "in_channels must be equal to out_channels for FPN fusion block"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gap = nn.AdaptiveAvgPool2d(1) # Global Average Pooling for channel-wise stats
        self.sigmoid = nn.Sigmoid()        # Activation for weights
        self.softmax = nn.Softmax(dim=2)   # Softmax over branches for competition

        # Sub-networks to predict channel weights for each input feature map
        # Using ConvModule for consistency with FPN's configuration
        # No activation in the final layer of these weight prediction branches
        self.se_current = nn.Sequential(
            ConvModule(in_channels, in_channels // 4, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels // 4, in_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        )
        self.se_top_down = nn.Sequential(
            ConvModule(in_channels, in_channels // 4, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(in_channels // 4, in_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        )

    def forward(self, lateral_feat: Tensor, top_down_feat: Tensor) -> Tensor:
        """
        Args:
            lateral_feat (Tensor): Feature map from the current lateral connection.
                                   Shape: [B, C, H, W]
            top_down_feat (Tensor): Upsampled feature map from the top-down path.
                                    Shape: [B, C, H, W]
        Returns:
            Tensor: Fused feature map. Shape: [B, C, H, W]
        """
        # Ensure spatial dimensions match before fusion
        # FPN's interpolate should have handled this
        assert lateral_feat.shape[2:] == top_down_feat.shape[2:], \
            f"Spatial dimensions must match for fusion: {lateral_feat.shape[2:]} vs {top_down_feat.shape[2:]}"

        # 1. Calculate channel-wise attention weights for each input branch
        # Apply Global Average Pooling: [B, C, H, W] -> [B, C, 1, 1]
        w_current = self.se_current(self.gap(lateral_feat))   # Output: [B, C, 1, 1]
        w_top_down = self.se_top_down(self.gap(top_down_feat)) # Output: [B, C, 1, 1]

        # 2. Concatenate weights and apply Sigmoid + Softmax for competition
        # Squeeze last two dims to get [B, C] for concatenation
        #weights_combined = torch.cat([w_current.squeeze(-1).squeeze(-1),
        #                              w_top_down.squeeze(-1).squeeze(-1)], dim=2) # Output: [B, C, 2]
        weights_combined = torch.cat([w_current, w_top_down], dim=2)
        # Apply Sigmoid first, then Softmax across the "branch" dimension (dim=2)
        weights_combined = self.softmax(self.sigmoid(weights_combined)).squeeze(-1) # Output: [B, C, 2]

        # 3. Split the combined weights and unsqueeze back for broadcast multiplication
        # [B, C, 2] -> split into [B, C] for each branch
        weight_current_scaled = weights_combined[:, :, 0].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        weight_top_down_scaled = weights_combined[:, :, 1].unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]

        # 4. Perform weighted sum of the feature maps
        fused_feat = weight_current_scaled * lateral_feat + \
                     weight_top_down_scaled * top_down_feat

        return fused_feat

class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations, bias=False, use_bn=True, use_relu=True):
        super().__init__()
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations, bias=bias)]
        if use_bn:
            modules.append(nn.BatchNorm2d(out_channels))
        if use_relu:
            modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================================================================
# CSFblock 定义
# ======================================================================
class CSFblock(nn.Module):
    def __init__(self, in_channels: int, channels_1: int, strides: int, activation: str='relu'):
        super().__init__()
        #self.Up = nn.Sequential(
        #    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=strides, padding=0),
        #    nn.BatchNorm2d(in_channels),
        #    nn.ReLU(),            
        #)

        self.Up = EUCB_MultiKernel(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_sizes=[3,5],
            stride=1,
            activation='gelu',
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0
        )

        self.Fgp = nn.AdaptiveAvgPool2d(1)
        # 根据你的 oneConv 定义，这里需要传入 kernel_sizes, paddings, dilations
        # 假设它们都是 1, 0, 1
        self.layer1 = nn.Sequential(
            oneConv(in_channels, channels_1, 1, 0, 1),
            oneConv(channels_1, in_channels, 1, 0, 1), # 通常这里会用 bias=False if followed by BN
        )
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.softmax = nn.Softmax(dim=2)
            
        self.evs = EVS(dim=in_channels, ffn_expansion_factor=1)
        #self.hpf = HighFrequencyEnhancementFFT(in_channels=in_channels)
        #self.evs_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x_h: Tensor, x_l: Tensor) -> Tensor:
        # x_h: high-resolution feature (lateral_feat)
        # x_l: low-resolution feature (laterals[i])

        x1 = x_h
        x2 = self.Up(x_l) # x_l is upsampled here
        x_f = x1 + x2# Preliminary sum
        
        Fgp = self.Fgp(x_f)
        x_se = self.layer1(Fgp)
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        x_se = torch.cat([x_se1, x_se2], 2) # Concatenate on dim=2 (branch dim)
        x_se = self.softmax(x_se) # Softmax on dim=2

        # Extract weights and unsqueeze for broadcast multiplication
        # x_se is [B, C, 2, 1] after cat and softmax
        att_3 = torch.unsqueeze(x_se[:,:,0],2) # [B, C, 1, 1]
        att_5 = torch.unsqueeze(x_se[:,:,1],2) # [B, C, 1, 1]

        x1 = att_3 * x1
        x2 = att_5 * x2
        
        x_all = x1 + x2 #+ torch.sigmoid(self.evs_weight) * x_evs# Final weighted sum
        return x_all

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
        self.mlp_for_weights = GSAU(n_feats=channels)
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        # 1. 全局平均池化和全局最大池化
        avg_out = self.avg_pool(x) # [B, C, 1, 1]
        max_out = self.max_pool(x) # [B, C, 1, 1]

        # 2. 通过共享 MLP 学习权重
        # 移除了 1x1 的空间维度，使得 MLP 接受 [B, C]
        # 但 ConvModule 期望 4D 输入，所以不 squeeze，让 ConvModule 自己处理 1x1
        #avg_out_processed = self.mlp(avg_out) # [B, C, 1, 1]
        #max_out_processed = self.mlp(max_out) # [B, C, 1, 1]
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
        #filtered_feature_map_final = self.mlp_for_weights(filtered_feature_map)

        return filtered_feature_map

class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            start_level: int = 0,
            end_level: int = -1,
            tf_encoder: nn.Module = None,
            add_extra_convs: Union[bool, str] = False,
            relu_before_extra_convs: bool = False,
            no_norm_on_lateral: bool = False,
            conv_cfg: OptConfigType = None,
            norm_cfg: OptConfigType = None,
            act_cfg: OptConfigType = None,
            upsample_cfg: ConfigType = dict(mode='nearest'),
            init_cfg: MultiConfig = dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        # --- 新增：实例化 ChannelAttention 模块列表 ---
        self.channel_attention_blocks = nn.ModuleList()
        # for i in range(self.start_level, self.backbone_end_level):
        #     self.channel_attention_blocks.append(
        #         ChannelAttention(
        #             channels=in_channels[i], # CA 模块的输入通道数是 Backbone 原始通道数
        #             reduction_ratio=16,      # 可以根据需要调整
        #             conv_cfg=conv_cfg,
        #             act_cfg=act_cfg
        #         )
        #     )
        # for i in range(self.start_level, self.backbone_end_level):
        #     self.channel_attention_blocks.append(
        #         Adaptive_GSAU_ChannelAttention(
        #             channels=in_channels[i], # CA 模块的输入通道数是 Backbone 原始通道数
        #             target_gsau_size = 8,
        #         )
        #     )
        for i in range(self.start_level, self.backbone_end_level):
            self.channel_attention_blocks.append(
                high_freq_perception_module(
                    in_channels=in_channels[i],
                    ratio_spatial=0.3, ratio_channel=0.3 # CA 模块的输入通道数是 Backbone 原始通道数
                )
            )
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        #self.parallel_edge_fusions = nn.ModuleList()
        #self.fusion_conv = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1, 
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)

            if i < self.num_outs:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)
                # self.evs.append(
                #     EVS(
                #         dim = out_channels,
                #         ffn_expansion_factor = 2
                #     )
                # )

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.tf_encoder = tf_encoder
        self.pos = PositionEmbeddingSine(num_pos_feats=out_channels // 2, normalize=True) if tf_encoder else None
        self.fusion_blocks = nn.ModuleList()

        num_fusion_points_init = (self.backbone_end_level - self.start_level) - 1
        if num_fusion_points_init < 0: # 应对 num_outs=1 或只有一个输入层级的情况
            num_fusion_points_init = 0

        for _ in range(num_fusion_points_init):
            # CSFblock 的 strides 参数需要根据 FPN 的下采样倍数确定
            # FPN 的 laterals[i] 到 laterals[i-1] 是 2x 上采样关系
            # 所以 CSFblock 的 stride 应该就是 2
            self.fusion_blocks.append(
                CSFblock(
                    in_channels=out_channels, # FPN 内部统一的通道数
                    channels_1=out_channels // 4, # 根据 CSFblock 内部 layer1 的设计
                    strides=2 # FPN 的上采样倍数
                )
            )

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # --- 新增：在 lateral_convs 之前应用 ChannelAttention ---
        # inputs 列表是来自 Backbone 的原始特征图
        processed_inputs = []
        for i, x_input in enumerate(inputs):
            # 确保索引匹配 channel_attention_blocks
            # 这里的 i 与 self.channel_attention_blocks 中的索引是对应的
            ca_block = self.channel_attention_blocks[i + self.start_level] # 确保索引正确
            processed_inputs.append(ca_block(x_input))


        # build laterals
        laterals = [
            lateral_conv(processed_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if self.tf_encoder:
            src = laterals[-1]
            bs, c, h, w = src.shape

            mask = torch.zeros_like(src, dtype=torch.bool)
            mask = mask[:, 0]

            pos_embed = self.pos(src, mask)

            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)

            laterals[-1] = self.tf_encoder(src, src_key_padding_mask=mask, pos=pos_embed).permute(1, 2, 0).view(bs, c,
                                                                                                                h, w)

        # build top-down path
        used_backbone_levels = len(laterals)
        #laterals[-1] = self.top_level_enhancer(laterals[-1])

        # 确保有融合点才执行循环 (used_backbone_levels > 1)
        if used_backbone_levels > 1:
            for i in range(used_backbone_levels - 1, 0, -1):
                # 1. 确定要传入 CSFblock 的两个特征
                # x_h (高分辨率特征) 对应 laterals[i-1]
                lateral_high_res_feat = laterals[i - 1]
                # x_l (低分辨率特征，未上采样) 对应 laterals[i]
                top_down_low_res_feat = laterals[i]

                # 2. 选择当前的 CSFblock 实例
                fusion_block_index = (used_backbone_levels - 1) - i
                
                # 确保索引不越界 (防御性编程)
                assert fusion_block_index >= 0 and fusion_block_index < len(self.fusion_blocks), \
                    f"Fusion block index {fusion_block_index} out of bounds for {len(self.fusion_blocks)} blocks. " \
                    f"i={i}, used_backbone_levels={used_backbone_levels}"
                
                current_fusion_block = self.fusion_blocks[fusion_block_index]
                
                # 3. 调用 CSFblock 进行融合
                # CSFblock 的 forward(x_h, x_l)
                # x_h 传入 lateral_high_res_feat
                # x_l 传入 top_down_low_res_feat
                laterals[i - 1] = current_fusion_block(lateral_high_res_feat, top_down_low_res_feat)

        # --- END MODIFICATION ---

        # build outputs
        # part 1: from original levels
        # 这里的 fpn_convs 仍然保留，它在融合后对特征进行平滑
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(min(used_backbone_levels, len(self.fpn_convs)))
        ]

       
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

'''
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if self.tf_encoder:
            src = laterals[-1]
            bs, c, h, w = src.shape

            mask = torch.zeros_like(src, dtype=torch.bool)
            mask = mask[:, 0]

            pos_embed = self.pos(src, mask)

            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)

            laterals[-1] = self.tf_encoder(src, src_key_padding_mask=mask, pos=pos_embed).permute(1, 2, 0).view(bs, c,
                                                                                                                h, w)

        # build top-down path
        used_backbone_levels = len(laterals)
        # 融合块的索引，从融合最高层级到次高层级开始（即 FPN 自上而下路径的第一步融合）
        # laterals 列表中，index 0 是最高分辨率的，index -1 是最低分辨率的。
        # 循环 `range(used_backbone_levels - 1, 0, -1)` 意味着 `i` 从 `used_backbone_levels - 1` 开始，到 `1` 结束。
        # 当 `i` 最大时（例如 `used_backbone_levels - 1`），`i-1` 是 `used_backbone_levels - 2`。
        # 这是 FPN 中从最低分辨率 Backbone 特征向上融合的第一步。
        # 因此，fusion_blocks 的索引应该与这个 i 的顺序对应。
        # 或者更简单： fusion_blocks 的索引是 `(used_backbone_levels - 1 - i)`
        
        for i in range(used_backbone_levels - 1, 0, -1):
            # lateral_feat 是 laterals[i-1]
            # top_down_feat 是 F.interpolate(laterals[i])

            # 1. 上采样高层特征 (top-down feature)
            #if 'scale_factor' in self.upsample_cfg:
            #    upsampled_top_feat = F.interpolate(laterals[i], **self.upsample_cfg)
            #else:
            #    prev_shape = laterals[i - 1].shape[2:]
            #    upsampled_top_feat = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            # 2. 使用 AdaptiveFPNFusionBlock 进行自适应融合
            # 这里的索引计算：
            # 当 i = used_backbone_levels - 1 时（最深的层级融合，如 P5->P4），idx = 0
            # 当 i = 1 时（最浅的层级融合，如 P3->P2），idx = used_backbone_levels - 2
            # 也就是 `num_fusion_points_init - (i - self.start_level)` 
            # 或者简单的: `len(self.fusion_blocks) - (i)` 如果 i 是从 num_levels开始
            # 鉴于 `fusion_blocks` 的添加顺序是 `(used_backbone_levels - 1 - i)`
            # 且 `i` 从 `used_backbone_levels - 1` 递减到 `1`
            # 那么 `fusion_block_idx = (used_backbone_levels - 1) - i` 是正确的索引方式。
            fusion_block_index = (used_backbone_levels - 1) - i
            
            # 确保索引不越界
            assert fusion_block_index >= 0 and fusion_block_index < len(self.fusion_blocks), \
                f"Fusion block index {fusion_block_index} out of bounds for {len(self.fusion_blocks)} blocks"
            
            current_fusion_block = self.fusion_blocks[fusion_block_index]
            
            # lateral_feat 是 laterals[i-1] (当前层级的 lateral)
            # top_down_feat 是 upsampled_top_feat (上采样的高层特征)
            laterals[i - 1] = current_fusion_block(laterals[i - 1], upsampled_top_feat)

        # build outputs
        # part 1: from original levels
        # 这里的 fpn_convs 仍然保留，它在融合后对特征进行平滑
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(min(used_backbone_levels, len(self.fpn_convs)))
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
       return tuple(outs)
''' 

'''
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if self.tf_encoder:
            src = laterals[-1]
            bs, c, h, w = src.shape

            mask = torch.zeros_like(src, dtype=torch.bool)
            mask = mask[:, 0]

            pos_embed = self.pos(src, mask)

            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)

            laterals[-1] = self.tf_encoder(src, src_key_padding_mask=mask, pos=pos_embed).permute(1, 2, 0).view(bs, c,
                                                                                                                h, w)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(min(used_backbone_levels, len(self.fpn_convs)))
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
'''

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):

        import math

        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
