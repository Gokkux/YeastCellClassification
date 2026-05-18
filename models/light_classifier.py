import torch
import torch.nn as nn
import torch.nn.functional as F

# 从您的项目导入 FPN
from .fpn import FPN 
from transformers import AutoModel
from .EUCB import GSAU
import timm 
import sys
import os
from sklearn.cluster import KMeans

class DistilledHead(nn.Module):
    """
    An intelligent head that uses a pretrained linear layer as a fixed
    feature basis, and then learns a small linear probe on top of it.
    """
    def __init__(self, pretrained_weights, pretrained_bias, num_new_classes=7):
        super().__init__()
        
        # Get dimensions from the pretrained weights
        old_num_classes, in_features = pretrained_weights.shape
        
        # 1. Create the fixed, pretrained "feature basis" layer
        self.feature_basis = nn.Linear(in_features, old_num_classes)
        
        # Load the pretrained weights and biases
        with torch.no_grad():
            self.feature_basis.weight.copy_(pretrained_weights)
            self.feature_basis.bias.copy_(pretrained_bias)
        
        # 【Key Step】Freeze this layer. It will only act as a feature extractor.
        for param in self.feature_basis.parameters():
            param.requires_grad = False # False
            
        # 2. Create the new, trainable "linear probe" layer
        self.probe = nn.Linear(old_num_classes, num_new_classes)
        
        # Initialize the probe layer for better training
        nn.init.kaiming_normal_(self.probe.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.probe.bias)

    def forward(self, x):
        # x has shape [batch_size, in_features], e.g., [B, 768]
        
        # Pass through the fixed feature basis to get 1000-dim features
        x = self.feature_basis(x) # -> [batch_size, 1000]
        
        # Optional: Add a non-linearity and dropout for more capacity/regularization
        x = nn.functional.relu(x, inplace=True)
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        
        # Pass through the trainable probe to get final 7-class logits
        x = self.probe(x) # -> [batch_size, 7]
        
        return x

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        # 确保卷积的 bias 参数在有 norm 层时为 False，没有时为 True
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=(norm_cfg is None))
        
        # 真正有 BatchNorm 的地方
        if norm_cfg:
            self.norm = nn.BatchNorm2d(out_channels) # 或者根据 norm_cfg 类型实例化其他 norm 层
        else:
            self.norm = nn.Identity() # 不使用任何 norm 层

        self.act = nn.ReLU() if act_cfg else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# ChannelAttentionModule (您最新修正的版本)
class ChannelAttentionModule(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, conv_cfg: dict = None, act_cfg: dict = None):
        super(ChannelAttentionModule, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            ConvModule(
                channels, channels // reduction_ratio, 1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                channels // reduction_ratio, channels, 1,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=None
            )
        )
        self.sigmoid = nn.Sigmoid()
        self.GSAU = GSAU(n_feats=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        combined_features = avg_out + max_out 
        attention_weights_raw = self.GSAU(combined_features)
        attention_weights = self.sigmoid(attention_weights_raw)
        
        return x * attention_weights

# HighFrequencyEnhancementFFT (您最新修正的版本，包含门控融合和残差连接)
class HighFrequencyEnhancementFFT(nn.Module):
    def __init__(self, in_channels, threshold=0.3):
        super().__init__()
        self.threshold = threshold
        
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1)
        )
        
        self.CA = ChannelAttentionModule(in_channels, reduction_ratio=16)
        
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.Sigmoid() 
        )
        
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x): # x 是原始输入特征图 [B, C, H, W]
        x_original = x 
        B, C, H, W = x.shape
        
        # --- 分支 1: FFT 高频增强 + CA ---
        fft_map = torch.fft.fft2(x, norm='ortho')
        fft_map_shifted = torch.fft.fftshift(fft_map, dim=(-2, -1))
        mask = torch.zeros_like(fft_map_shifted)
        cx, cy = H // 2, W // 2
        th_h, th_w = int(cy * self.threshold), int(cx * self.threshold)
        mask[:, :, cy - th_h : cy + th_h, cx - th_w : cx + th_w] = 1 
        mask = 1 - mask 

        high_freq_fft = fft_map_shifted * mask
        high_freq_fft = torch.fft.ifftshift(high_freq_fft, dim=(-2, -1))
        hf_feat = torch.fft.ifft2(high_freq_fft, norm='ortho').real 

        gap_hf = F.adaptive_avg_pool2d(hf_feat, 1)
        gmp_hf = F.adaptive_max_pool2d(hf_feat, 1)
        ch_attn_logits = self.ca_mlp(gap_hf) + self.ca_mlp(gmp_hf)
        ch_attn_hf = torch.sigmoid(ch_attn_logits) 
        modulated_x_hf = x * ch_attn_hf 

        # --- 分支 2: 普通通道注意力 ---
        modulated_x_normal = self.CA(x) 

        # --- 门控融合 ---
        combined_for_gate = torch.cat([modulated_x_hf, modulated_x_normal], dim=1)
        gate = self.gate_predictor(combined_for_gate)
        fused_features = gate * modulated_x_hf + (1 - gate) * modulated_x_normal

        # 最终处理和残差连接
        processed_fused_features = self.final_conv(fused_features)
        # final_output = processed_fused_features + x_original
        
        return processed_fused_features


# --- 保持 MambaVision 的加载方式，但调整其输出 ---
class MambaVisionBackbone(nn.Module): # 重命名以避免与旧的 Backbone 混淆
    def __init__(self, cfg):
        super().__init__()
        # 加载 MambaVision 模型
        # cfg.prompter.backbone 包含 model_type, trust_remote_code 等
        self.mamba_model = AutoModel.from_pretrained(
            cfg.prompter.backbone.model_type,
            trust_remote_code=getattr(cfg.prompter.backbone, 'trust_remote_code', False)
        )
        print(type(self.mamba_model))
        # self.mamba_model = timm.create_model(
        #    **cfg.prompter.backbone
        # )
        # FPN 颈部
        # cfg.prompter.neck 包含 in_channels, out_channels 等
        self.neck = FPN(**cfg.prompter.neck)
        
        # 确定 FPN 输出的通道数
        # FPN 的 out_channels 通常是一个列表或单一值，如果FPN只有一个输出，那就是这个值
        # 这里假设 cfg.prompter.neck.out_channels 是一个整数，或者 FPN 内部会处理
        # 假设我们最终会从 FPN 输出的某个层获取特征，其通道数是 cfg.prompter.neck.out_channels[-1]
        # 或者 FPN 的 num_outs 后的 out_channels
        self.out_channels = cfg.prompter.neck.out_channels[0] if isinstance(cfg.prompter.neck.out_channels, list) else cfg.prompter.neck.out_channels # 假设FPN输出out_channels的第一个值或本身
        # 如果FPN的out_channels是列表，比如[256, 512, 1024]，那么通常是所有输出都有256通道，这里需要确认
        # 一般FPN会为每个输出层指定通道数，如果都是256，那out_channels就是256

        # FPN 可能会有多个输出，用于分类，我们通常选择最语义化的层 (最高层或 FPN 的统一输出)
        # 原始模型中 neck1 用于 num_outs=1 的情况，我们在此直接利用 FPN 的某个输出

    def forward(self, images):
        # MambaVision 的 forward 方法返回 (logits, features_list)
        # features_list 包含不同阶段的特征，例如 (stage1_feat, stage2_feat, stage3_feat, stage4_feat)
        # (通常对应 C2, C3, C4, C5 特征)
        _, features_list = self.mamba_model(images)

        # FPN 的输入通常是 backbone 不同阶段的特征列表
        # 确保 features_list 的顺序和数量与 FPN 期望的 in_channels 匹配
        fpn_input_features = [
            features_list[0], # 通常对应 MambaVision 的某个stage的输出
            features_list[1],
            features_list[2],
            features_list[3]  # 通常是最高语义层
        ]

        # self.neck(fpn_input_features) 返回一个列表，包含 P2, P3, P4, P5 等多尺度特征
        fpn_outputs = self.neck(fpn_input_features)
        
        # 对于图像分类，我们通常取 FPN 输出的最高语义层特征作为全局特征来源
        # 假设 fpn_outputs[0] 是 P2 (最高分辨率), fpn_outputs[-1] 是 P5 (最低分辨率，最高语义)
        # 或者 FPN 自身会进行全局池化并输出
        
        # 返回 FPN 的最高层特征 (例如 P5)，它具有最丰富的语义信息
        # 形状通常是 [B, C, H', W']，其中 H', W' 是较小的分辨率
        return fpn_outputs[-1] # [B, self.out_channels, H_p5, W_p5]

class CellClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = MambaVisionBackbone(cfg) # 使用 MambaVisionBackbone
        # 确定骨干网络输出的通道数
        self.backbone_out_channels = self.backbone.out_channels 

        # 特征增强模块
        enhancer_threshold = getattr(cfg.model, 'feature_enhancer_threshold', 0.3)
        self.feature_enhancer = HighFrequencyEnhancementFFT(
            in_channels=self.backbone_out_channels, 
            threshold=enhancer_threshold
        )

        # 分类头 
        num_classes = cfg.model.num_classes 
        
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out_channels, 512), # 分类头输入通道是 FPN 输出的通道数
            nn.ReLU(inplace=True),
            nn.Dropout(getattr(cfg.model, 'classifier_dropout', 0.5)),
            nn.Linear(512, num_classes)
        )

    def forward(self, images):
        # 1. 骨干网络提取特征
        # features 形状: [B, C, H_p5, W_p5] (来自 MambaVisionBackbone 的 FPN 输出的最高层)
        features_4d = self.backbone(images) 
        
        # 2. 特征增强模块 (期望 4D 输入，enhancer 输出也是 4D)
        enhanced_feature_map = self.feature_enhancer(features_4d)
        
        # 3. 全局池化，将 4D 特征图转换为 2D 特征向量
        # enhanced_feature_map 形状 [B, C, H_p5, W_p5]
        # 全局平均池化后形状 [B, C, 1, 1]
        pooled_feature = F.adaptive_avg_pool2d(features_4d, 1).flatten(1) # [B, C]

        # 4. 通过分类器预测 logits
        logits = self.classifier(pooled_feature)

        return {'logits': logits}

from transformers import AutoConfig, AutoModelForImageClassification

# 将你本地的、修改过的类注册到 AutoModel 体系中
from .custom_mamba_vision import MambaVisionConfig, MambaVisionModelForImageClassification, MambaVision, build_deep_fusion_yeastnet
AutoModelForImageClassification.register(MambaVisionConfig, MambaVisionModelForImageClassification)

class CellClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 这个是我们在 Hub 上的基础模型名称，只用来下载权重和原始配置
        hub_model_name = cfg.prompter.backbone.model_type # e.g., "shi-labs/prompt-mamba-h-plus"
        
        # --- 1. 从 Hub 加载配置 ---
        # 这一步是为了获取所有基础参数，如 hidden_size, num_heads 等
        print(f"\n--- STAGE 1: Loading base configuration from '{hub_model_name}' ---")
        config = AutoConfig.from_pretrained(
            hub_model_name,
            num_labels=cfg.model.num_classes,
            trust_remote_code=True
        )
        
        # --- 2. (可选) 注入你自己的参数 ---
        # 确保你的自定义模型能接收到这些参数
        config.use_hfe_branch = getattr(cfg.model, 'use_hfe_branch', True)
        
        print("\n--- STAGE 2: Explicitly creating model instance from YOUR local code ---")
        # --- 3. 【核心】直接、明确地使用你的本地类来创建模型实例 ---
        # 此时，self.mamba_model 100% 是你修改后的那个版本！
        # 它的权重是随机初始化的。
        self.mamba_model = MambaVisionModelForImageClassification(config)
        
        initial_params = sum(p.numel() for p in self.mamba_model.parameters())
        print(f"Successfully created an instance of YOUR <{self.mamba_model.__class__.__name__}>.")
        print(f"Parameters in your local model structure: {initial_params:,}")
        
        print(f"\n--- STAGE 3: Loading pretrained weights from '{hub_model_name}' into YOUR model ---")
        # --- 4. 【核心】从 Hub 加载预训练权重到我们自己的模型中 ---
        try:
            # a. 先从 Hub 加载一个标准的、未经修改的模型，把它当作权重的“临时容器”
            print("Downloading weights using a temporary standard model...")
            temp_model = AutoModelForImageClassification.from_pretrained(
                hub_model_name, # <-- 从 Hub 下载
                trust_remote_code=True
            )
            
            # b. 获取这个临时容器的权重字典
            pretrained_state_dict = temp_model.state_dict()
            
            # c. 将权重加载到我们自己创建的那个模型中
            #    strict=False 是关键！它允许加载那些名称匹配的权重，
            #    并优雅地忽略那些你新增的、在预训练权重中不存在的模块（比如你的 hfe_branch）。
            #    如果不加，当它发现你的模型里有预训练权重里没有的键时，会直接报错。
            self.mamba_model.load_state_dict(pretrained_state_dict, strict=False)
            print("Successfully loaded matching weights into your custom model.")
            
            # 释放临时模型的内存
            del temp_model
            
        except Exception as e:
            print(f"Could not load pretrained weights. Error: {e}. The model will be trained from scratch.")

        print("\n--- FINAL CHECK: Printing final model structure ---")
        print(self.mamba_model)
        print("---------------------------------------------------\n")

        # --- 5. 冻结逻辑 (现在可以正常工作了) ---
        # 替换为这段正确的代码
        if getattr(cfg.model, 'freeze_backbone', False):
            print("Attempting to freeze backbone parameters...")
            
            for param in self.mamba_model.parameters():
                param.requires_grad = False
            
            # 【核心修正】我们现在检查 self.mamba_model.model.head 这个正确的、嵌套的路径
            if hasattr(self.mamba_model, 'model') and hasattr(self.mamba_model.model, 'head'):
                print("Unfreezing the classification head ('model.head')...")
                # 同样，我们从正确的路径获取参数
                for param in self.mamba_model.model.head.parameters():
                    param.requires_grad = True
            else:
                # 如果还是失败，给出更精确的错误信息
                raise AttributeError("Could not find the '.model.head' attribute in your custom model for unfreezing.")
                
            # 验证冻结是否成功
            total_params = sum(p.numel() for p in self.mamba_model.parameters())
            trainable_params = sum(p.numel() for p in self.mamba_model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters after freezing: {trainable_params:,}")

        else:
            print("Training the entire model (backbone is not frozen).")
            
        # ================= 侦查代码 START =================
        print("\n\n>>>>>>>>>> 深入侦查开始 <<<<<<<<<<")

        # 1. 打印最终模型的完整配置
        print("\n--- 1. 最终模型的 CONFIG 对象 ---")
        print(self.mamba_model.config)

        # 2. 明确检查分类头的配置和参数
        if hasattr(self.mamba_model, 'model') and hasattr(self.mamba_model.model, 'head'):
            print("\n--- 2. 分类头 (model.head) 详情 ---")
            head = self.mamba_model.model.head
            print(f"  - 分类头类型: {type(head)}")
            
            # 检查分类头的权重形状
            if hasattr(head, 'weight'):
                print(f"  - 权重 (head.weight) 形状: {head.weight.shape}")
                # 形状通常是 [num_classes, input_features]
                num_classes_from_weight = head.weight.shape[0]
                input_features = head.weight.shape[1]
                print(f"  - 从权重推断的类别数: {num_classes_from_weight}")
                print(f"  - 输入到分类头的特征维度: {input_features}")
            else:
                print("  - 分类头中未找到 'weight' 参数。")
                
            # 检查config中的num_labels
            num_labels_in_config = self.mamba_model.config.num_labels
            print(f"  - CONFIG中的num_labels: {num_labels_in_config}")

            if num_labels_in_config != num_classes_from_weight:
                print("\n  >>> !!! 严重警告: Config中的num_labels与实际分类头权重形状不匹配! <<<")

        else:
            # 备用方案，如果结构不同，直接打印整个模型
            print("\n--- 2. 无法定位 'model.head'，打印完整模型结构 ---")
            print(self.mamba_model)

        # 3. 重新计算并打印参数，确保一致
        total_params_check = sum(p.numel() for p in self.mamba_model.parameters())
        trainable_params_check = sum(p.numel() for p in self.mamba_model.parameters() if p.requires_grad)
        print("\n--- 3. 参数量最终确认 ---")
        print(f"  - 总参数: {total_params_check:,}")
        print(f"  - 可训练参数: {trainable_params_check:,}")

        print(">>>>>>>>>> 侦查结束 <<<<<<<<<<\n\n")
        # ================= 侦查代码 END =================

class CellClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        hub_model_name = cfg.prompter.backbone.model_type
        
        # --- STAGE 1: Load base configuration to get model parameters ---
        # 我们仍然需要config来获取模型的超参数，如 dim, depths, mlp_ratio等
        # 但我们不再需要它的预训练权重信息。
        print(f"\n--- STAGE 1: Loading base configuration from '{hub_model_name}' for architecture definition ---")
        config = AutoConfig.from_pretrained(
            hub_model_name,
            num_labels=7, # <-- 关键：直接设置为你的目标类别数，例如7
            trust_remote_code=True
        )
        
        # --- Inject your custom parameters into the config ---
        config.use_hfe_branch = getattr(cfg.model, 'use_hfe_branch', False)
        
        # --- STAGE 2: Build YOUR custom model from scratch ---
        print("\n--- STAGE 2: Building custom model architecture from scratch ---")
        
        # 创建一个标准的 Hugging Face 封装模型
        self.model = AutoModelForImageClassification.from_config(config, trust_remote_code=True)
        
        # 【关键】用你本地的新架构替换掉它内部的骨干网络
        # 这确保了所有新模块（GaborStem, sym_block等）都被正确创建
        self.model.model = MambaVision(**config.to_dict())

        print("Successfully built custom model from scratch.")
        print("All weights are randomly initialized according to your _init_weights method.")

        # --- STAGE 5: Freezing logic (现在变得不那么必要，但保留以备将来使用) ---
        # 对于从头训练，通常我们会训练所有参数。
        print("\n--- STAGE 3: Applying freezing logic (if any) ---")
        if getattr(cfg.model, 'freeze_backbone', False):
            # 虽然从头训练时很少冻结，但我们还是保留这个逻辑以防特殊实验
            print("WARNING: 'freeze_backbone' is True, but model is trained from scratch.")
            print("Freezing backbone and only training specific new modules.")
            
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                
                trainable_modules = ['model.head', 'model.sym_block', 'model.patch_embed', 'model.hfe_branch']
                if any(name.startswith(mod) for mod in trainable_modules):
                    param.requires_grad = True
                    print(f"  - Unfreezing: {name}")
        else:
            # 这是最常见的情况
            print("Training the entire model from scratch (no freezing).")

        # 最终参数检查
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n--- Final Parameter Check ---")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("-----------------------------\n")

    def forward(self, x):
        return self.model(x)

def build_model(cfg):
    """
    根据配置构建 CellClassifier 模型。
    """
    model = build_deep_fusion_yeastnet(num_classes=cfg.model.num_classes)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"YeastNet total parameters: {total_params:,}")
    return model