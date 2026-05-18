# 这是一个新的配置文件，请将其保存为 config/cell_config.py
# 并在 main.py 中将 config 的默认值改为这个文件：
# parser.add_argument('--config', default='cell_config.py', type=str)

# --- 模型相关配置 ---
# 这部分配置将传递给 CellClassifier 及其内部组件
model = dict(
    # 分类任务的类别数量 (会被 prepare_cell_data 动态覆盖为 7)
    num_classes=7, 
    # 模型输入图像尺寸，与 CellLightScatteringDataset 中的 transform 匹配
    image_size=224, 
    
    # Backbone 配置 (MambaVisionBackbone 使用)
    backbone_type='mambavision', # 标记骨干网络类型为 mambavision
    # MambaVision 的具体配置直接放到 prompter.backbone 去了
    # 如果未来切换到 torchvision 的 ResNet，这里可以写 'resnet50' 等
    
    # HighFrequencyEnhancementFFT 模块的配置
    feature_enhancer_threshold=0.3, # 可以在这里调整 FFT 阈值

    # 分类头 dropout 率
    classifier_dropout=0.5,
    freeze_backbone = False, 
    # EMA 配置 (可以根据需要在 main.py args 中设置，或在此定义)
    # model_ema_decay=0.99,
    # model_ema_steps=1,
)

# --- Prompter 相关配置 (主要用于 MambaVisionBackbone) ---
prompter = dict(
    backbone=dict(
        # MambaVision 模型类型
        model_type='nvidia/MambaVision-S-1K', 
        trust_remote_code=True,
        # model_name='convnext_small',
        # pretrained=True,
        # features_only=True,
        # num_classes=0,
        # out_indices=(0, 1, 2, 3), 
        # global_pool=''
    ),
    neck=dict(
        # FPN 的输入通道数，需要匹配 MambaVision 各阶段的输出通道
        # 如果 MambaVision-S-1K 的 stages 输出是 96, 192, 384, 768，则这个配置是正确的
        in_channels=[96, 192, 384, 768], 
        # FPN 的输出通道数。CellClassifier 会使用 FPN 最后一层输出的这个通道数
        out_channels=256, 
        # FPN 的输出层数量。原始是 3，但我们现在只取 fpn_outputs[-1] 这一层
        # num_outs 仍然会影响 FPN 的构建，保留原始值可能更安全，或者根据FPN定义调整
        num_outs=3, 
        add_extra_convs='on_input',
        #projection_channels= 256
    ),
    # 以下为原检测任务特有，现在可以移除或注释
    # dropout=0.3,
    # space=32,
    # spaces=[16,32,64],
    # hidden_dim=256, # CellClassifier 的 hidden_dim 在其内部定义了 512
    # learnable_heatmap_prompter=dict(...),
    # GridHeatPrompter=dict(...),
    # wavelet_prompter=dict(...),
)

# --- 数据相关配置 ---
data = dict(
    # 数据集名称（仅用于记录或日志）
    name='cell_light_scattering', 
    
    # 类别数量（会被 main.py 中的 prepare_cell_data 动态计算并覆盖为 7）
    # 但在这里先给一个默认值，或者您知道实际类别数可以写死
    num_classes=7, 
    
    # 图像文件所在的根目录，对应 prepare_cell_data 中的 zip_files_parent_dir
    #root_dir='/root/autodl-tmp/PromptNucSeg-main/prompter/datasets/light_scattering',
    root_dir='/root/autodl-tmp/PromptNucSeg-main/prompter/datasets/light_scattering_resized',
    path_scatter = '/root/autodl-tmp/PromptNucSeg-main/prompter/datasets/light_scattering',
    path_microscope = '/root/autodl-tmp/PromptNucSeg-main/prompter/datasets/light_scattering_resized',
    # 图像文件所在的子文件夹名称，对应 prepare_cell_data 中的 IMAGE_SUBFOLDER
    image_subfolder='cropped', 

    batch_size_per_gpu=16, # 每块GPU的批次大小
    num_workers=8,        # 数据加载的工作进程数

    # --- 图像分类的 Transforms 配置 ---
    # 这部分配置将被 CellLightScatteringDataset 读取并构建 Albumentations Transforms
    train_transforms=[ # 重命名为 train_transforms
        dict(type='RandomResizedCrop', height=224, width=224, p=1.0),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='VerticalFlip', p=0.5), # 垂直翻转是否合理取决于细胞对称性
        dict(type='RandomRotate90', p=0.5), # 随机旋转90度
        # dict(type='Rotate', limit=30, p=0.5, interpolation=0), # 任意角度旋转，interpolation=0 for INTER_NEAREST
        # dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3), # 谨慎使用
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet 均值和标准差
        # ToTensorV2 已经在 CellLightScatteringDataset 中默认添加了，这里无需再定义
    ],
    val_transforms=[ # 重命名为 val_transforms
        dict(type='Resize', height=256, width=256), # 先缩放短边到256
        dict(type='CenterCrop', height=224, width=224), # 然后中心裁剪为224x224
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    test_transforms=[ # 重命名为 test_transforms，与 val_transforms 相同
        dict(type='Resize', height=256, width=256),
        dict(type='CenterCrop', height=224, width=224),
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    
    # 以下为原检测任务特有，现在可以移除或注释
    # num_mask_per_img=20,
    # num_neg_prompt=1,
    # train=dict(transform=...), # 移除旧的 train transforms
    # val=dict(transform=...), # 移除旧的 val transforms
    # test=dict(transform=...), # 移除旧的 test transforms
)

# --- 优化器配置 ---
optimizer = dict(
    type='AdamW', # 建议使用 AdamW，而不是 Adam
    lr=1e-4,  #1e-4
    weight_decay=1e-3
)


# --- 学习率调度器配置 ---
scheduler = dict(
    type='MultiStepLR',
    milestones=[250], # 训练 200 epochs，80 可能太早，可以调整为 [100, 150] 或使用 CosineAnnealing
    gamma=0.1
)

# --- 损失函数配置 ---
criterion = dict(
    type='CrossEntropyLoss', # 明确使用交叉熵损失
    # 如果想用 Focal Loss，可以改为 'FocalLoss'，并添加 alpha 和 gamma
    # type='FocalLoss', 
    # focal_alpha=0.25,
    # focal_gamma=2.0,
    # class_weights=None, # 如果需要类别加权，在这里定义一个列表，长度与 num_classes 相同
    
    # 以下为原检测任务特有，现在可以移除或注释
    # matcher=dict(type='HungarianMatcher', dis_type='l2', set_cost_point=0.1, set_cost_class=1),
    # eos_coef=0.4, 
    # reg_loss_coef=5e-3,
    # cls_loss_coef=1.0, # 这个是旧的 cls loss coef，新的是由 CrossEntropyLoss 统一管理
    # mask_loss_coef=1.0
)

# --- 测试配置 (移除检测相关) ---
test = dict(
    # metric_to_save='accuracy', # 可以在这里定义主评估指标，但 main.py 已经有 args.metrics_to_track
    # 以下为原检测任务特有，现在可以移除或注释
    # nms_thr=12, 
    # match_dis=13, 
    # filtering=False
)