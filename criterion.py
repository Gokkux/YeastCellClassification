
import torch
import torch.nn.functional as F
from torch import nn

from .utils import is_dist_avail_and_initialized, get_world_size


def build_criterion(cfg, device):
    """
    为图像分类任务构建损失函数。
    支持 nn.CrossEntropyLoss 和多分类 FocalLoss。
    """
    num_classes = cfg.model.num_classes
    loss_type = cfg.criterion.type 

    class_weights = None
    if hasattr(cfg.criterion, 'class_weights') and cfg.criterion.class_weights is not None:
        if len(cfg.criterion.class_weights) != num_classes:
            raise ValueError(f"Length of class_weights ({len(cfg.criterion.class_weights)}) "
                             f"must match num_classes ({num_classes}).")
        class_weights = torch.tensor(cfg.criterion.class_weights, dtype=torch.float).to(device)

    if loss_type == 'CrossEntropyLoss':
        print(f"Building CrossEntropyLoss for {num_classes} classes.")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05) 
        
    elif loss_type == 'FocalLoss':
        print(f"Building Multi-class FocalLoss for {num_classes} classes.")
        # --- 多分类 Focal Loss 的推荐实现 (二选一或自行实现) ---
        # 推荐使用 Kornia 的 FocalLoss，因为它明确支持多分类
        try:
            from kornia.losses import FocalLoss as KorniaFocalLoss

            focal_alpha = getattr(cfg.criterion, 'focal_alpha', 0.25)
            focal_gamma = getattr(cfg.criterion, 'focal_gamma', 2.0)

            criterion = KorniaFocalLoss(
                alpha=focal_alpha, 
                gamma=focal_gamma, 
                reduction='mean',
                weight=class_weights # 权重用于平衡类别
            )
        except ImportError:
            print("WARNING: Kornia not found for FocalLoss. Attempting with segmentation_models_pytorch.")
            try:
                from segmentation_models_pytorch.losses import FocalLoss as SMPFocalLoss
                # SMP 的 FocalLoss，mode='multiclass'
                focal_alpha = getattr(cfg.criterion, 'focal_alpha', 0.25)
                focal_gamma = getattr(cfg.criterion, 'focal_gamma', 2.0)
                criterion = SMPFocalLoss(
                    mode='multiclass',
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    reduction='mean',
                    weight=class_weights # 权重用于平衡类别
                )
            except ImportError:
                raise ImportError(
                    "FocalLoss requested but neither Kornia nor segmentation_models_pytorch were found. "
                    "Please install one (e.g., 'pip install kornia' or 'pip install segmentation_models_pytorch') "
                    "or change loss_type to 'CrossEntropyLoss'."
                )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. Choose 'CrossEntropyLoss' or 'FocalLoss'.")

    return criterion

