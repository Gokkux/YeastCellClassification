
# import wandb
import argparse
import os 
import zipfile 
import pandas as pd 
from sklearn.model_selection import train_test_split 

from .utils import * 
from mmengine.config import Config

from .engine import train_one_epoch, evaluate #

# --- 新增自定义数据集和 DataLoader 导入 ---
from prompter.cell_dataset import build_dataloaders
from torch.utils.tensorboard import SummaryWriter 
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .models.light_classifier import build_model
from .criterion import build_criterion
import sys
import os 

def parse_args():
    parser = argparse.ArgumentParser('Cell Classifier')
    parser.add_argument('--config', default='cell_config.py', type=str)
    parser.add_argument('--run-name', default=None, type=str)
    parser.add_argument('--group-name', default=None, type=str)

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Run Mode
    parser.add_argument('--eval', action='store_true')

    # * Train
    parser.add_argument('--seed', default=42, type=int) 
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--print-freq", default=5, type=int, help="print frequency")
    parser.add_argument("--use-wandb", action='store_true', help='use wandb for logging')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs.')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs.')
    parser.add_argument('--clip-grad', type=float, default=0.1,
                        help='Clip gradient norm (default: 0.1)')
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )

    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=1,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99)",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # * Distributed training
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--start_eval', default=10, type=int, help='Start evaluation after this epoch')

    parser.add_argument('--fast-check', action='store_true', help='use small subset and few epochs for fast testing')

    parser.add_argument("--metrics_to_track", default="accuracy", type=str, help="Metric to track for best model saving (e.g., accuracy, macro_f1)") # 这个参数已经有了，无需重复添加

    opt = parser.parse_args()
    return opt

import os, zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
EXCLUDE_DIRS = {'__MACOSX', '.DS_Store'}

def _has_images_recursively(path):
    for r, _, files in os.walk(path):
        if any(f.lower().endswith(IMAGE_EXTS) for f in files):
            return True
    return False

def _collect_images_recursively(root):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(r, f))
    return sorted(out)

def prepare_cell_data(cfg):
    """
    解压(如有) -> 识别类别目录 -> 递归收集图片 -> 建立映射并划分数据集
    兼容两种结构：
      1) root_dir/类名/IMAGE_SUBFOLDER/图片...
      2) root_dir/类名/(直接或子目录)图片...
    """
    zip_files_parent_dir = cfg.data.root_dir
    IMAGE_SUBFOLDER = getattr(cfg.data, "image_subfolder", None)
    if not IMAGE_SUBFOLDER:  # 允许 "", None
        IMAGE_SUBFOLDER = None

    print(f"Checking for ZIP files in: {zip_files_parent_dir}")
    if not os.path.exists(zip_files_parent_dir):
        raise FileNotFoundError(f"Parent directory '{zip_files_parent_dir}' not found.")

    # 1) 解压 .zip（如有）
    zip_file_paths = [os.path.join(zip_files_parent_dir, f)
                      for f in os.listdir(zip_files_parent_dir)
                      if f.lower().endswith('.zip')]
    if zip_file_paths:
        print(f"Found {len(zip_file_paths)} ZIP files. Starting/checking extraction...")
        for zip_file_path in zip_file_paths:
            folder_name_to_extract = os.path.splitext(os.path.basename(zip_file_path))[0]
            extract_to_path = os.path.join(zip_files_parent_dir, folder_name_to_extract)
            if os.path.isdir(extract_to_path) and any(os.scandir(extract_to_path)):
                print(f"  '{extract_to_path}' exists & not empty. Skip.")
            else:
                print(f"  Extracting '{os.path.basename(zip_file_path)}' -> '{extract_to_path}'")
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    zf.extractall(extract_to_path)
        print("Extraction step done.")
    else:
        print("No ZIP files found. Assuming folders are already extracted.")

    print("\n--- Starting image file collection and label mapping ---")

    # 2) 寻找类别目录：优先把 root_dir 下所有“递归内含图片”的一级子目录作为类别
    category_base_folders = []
    for item_name in sorted(os.listdir(zip_files_parent_dir)):
        if item_name in EXCLUDE_DIRS:
            continue
        p = os.path.join(zip_files_parent_dir, item_name)
        if os.path.isdir(p) and _has_images_recursively(p):
            category_base_folders.append(p)

    # 若没有找到子目录类别，但 root_dir 自己就有图片，则视作单类数据
    if not category_base_folders and _has_images_recursively(zip_files_parent_dir):
        print("No subdir classes found, but root has images. Treat root as ONE class.")
        category_base_folders = [zip_files_parent_dir]

    if not category_base_folders:
        raise ValueError(f"ERROR: No category folders found under '{zip_files_parent_dir}'. "
                         f"Expected like 'C.albicans', 'C.auris', ... each containing images.")

    print(f"Found {len(category_base_folders)} category folders: "
          f"{[os.path.basename(f) for f in category_base_folders]}")

    # 3) 收集图片与标签
    all_image_paths, all_image_labels_str = [], []
    print("\nCollecting images (recursive):")
    for folder_path in category_base_folders:
        class_name = os.path.basename(folder_path)
        # 若设置了 IMAGE_SUBFOLDER 且存在，则仅在该子目录下搜图；否则在类别目录下递归搜图
        scan_root = (os.path.join(folder_path, IMAGE_SUBFOLDER)
                     if IMAGE_SUBFOLDER and os.path.isdir(os.path.join(folder_path, IMAGE_SUBFOLDER))
                     else folder_path)
        imgs = _collect_images_recursively(scan_root)
        print(f"  {class_name}: {len(imgs)} images from '{scan_root}'")
        all_image_paths.extend(imgs)
        all_image_labels_str.extend([class_name] * len(imgs))

    print(f"\nTotal images found: {len(all_image_paths)}")
    if len(all_image_paths) == 0:
        raise ValueError("No images found. Please check paths or extensions.")

    # 4) 类别映射（按类名字典序保证可复现）
    unique_classes = sorted(set(all_image_labels_str))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    print("\nClass to ID mapping:")
    for c, i in class_to_idx.items():
        print(f"  '{c}': {i}")

    data_df = pd.DataFrame({
        'path': all_image_paths,
        'label_str': all_image_labels_str,
    })
    data_df['label_int'] = data_df['label_str'].map(class_to_idx)

    # 5) 保存类别数到 cfg（供构建模型用）
    cfg.model.num_classes = len(unique_classes)
    print(f"\nDetected {cfg.model.num_classes} classes.")

    # 6) 划分数据集（尽量分层，失败则无分层回退）
    def _safe_split(df, test_size, strat_col):
        try:
            return train_test_split(
                df, test_size=test_size, stratify=df[strat_col], random_state=42
            )
        except ValueError as e:
            print(f"  [warn] stratified split failed: {e}. Falling back to non-stratified.")
            return train_test_split(
                df, test_size=test_size, stratify=None, random_state=42
            )

    train_val_df, test_df = _safe_split(data_df, test_size=0.15, strat_col='label_int')
    # 验证集占比=0.15/(1-0.15)
    val_ratio = 0.15 / (1 - 0.15)
    train_df, val_df = _safe_split(train_val_df, test_size=val_ratio, strat_col='label_int')

    return train_df, val_df, test_df, class_to_idx, idx_to_class


def prepare_multimodal_data(cfg):
    """
    对齐光散射和显微镜两种模态的数据，并进行划分。
    """
    path_scatter = cfg.data.path_scatter
    path_microscope = cfg.data.path_microscope
    
    # --- 1. 定义类别名称映射 ---
    class_mapping = {
        'Calb_new': 'C.albicans',
        'Caur_new': 'C.auris',
        'Cgla_new': 'C.glabrata', 
        'Chae_new': 'C.haemulonii',     
        'Ckru_new':'krusei',
        'Cparap_new':'parapsilosis',
        "Ctro_new":'tropicalis'
    }
    print("Using class mapping:")
    print(class_mapping)

    # --- 2. 分别收集两种模态的图像路径 ---
    def collect_paths(base_path, subfolder_name=None):
        data = {}
        for class_name in os.listdir(base_path):
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                image_dir = os.path.join(class_path, subfolder_name) if subfolder_name else class_path
                if os.path.isdir(image_dir):
                    paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
                    data[class_name] = paths
        return data

    scatter_data = collect_paths(path_scatter, subfolder_name='cropped')
    microscope_data = collect_paths(path_microscope)

    # --- 3. 数据对齐与配对 ---
    paired_paths = []
    labels_str = []
    
    print("\n--- Aligning and pairing data ---")
    for scatter_class, microscope_class in class_mapping.items():
        if scatter_class in scatter_data and microscope_class in microscope_data:
            scatter_list = scatter_data[scatter_class]
            microscope_list = microscope_data[microscope_class]
            
            # 取两个列表数量的最小值
            num_pairs = min(len(scatter_list), len(microscope_list))
            
            print(f"  Class '{scatter_class}' <-> '{microscope_class}': "
                  f"Found {len(scatter_list)} scatter vs {len(microscope_list)} microscope. "
                  f"Creating {num_pairs} pairs.")
            
            # 随机打乱列表，然后取前 num_pairs 个进行配对
            random.shuffle(scatter_list)
            random.shuffle(microscope_list)
            
            for i in range(num_pairs):
                paired_paths.append({
                    'path_scatter': scatter_list[i],
                    'path_microscope': microscope_list[i]
                })
                labels_str.append(scatter_class) # 我们使用光散射的类别名作为标准
        else:
            print(f"  Warning: Class pair '{scatter_class}'/'{microscope_class}' not found in both datasets. Skipping.")
            
    # --- 4. 创建 DataFrame 和标签映射 ---
    data_df = pd.DataFrame(paired_paths)
    data_df['label_str'] = labels_str
    
    unique_classes = sorted(list(set(labels_str)))
    class_to_idx = {name: i for i, name in enumerate(unique_classes)}
    idx_to_class = {i: name for i, name in enumerate(unique_classes)}
    
    data_df['label_int'] = data_df['label_str'].map(class_to_idx)
    
    cfg.model.num_classes = len(unique_classes)
    print(f"\nTotal paired images: {len(data_df)}")
    print(f"Detected {cfg.model.num_classes} classes.")

    # --- 5. 划分数据集 ---
    train_val_df, test_df = train_test_split(
        data_df, test_size=0.15, stratify=data_df['label_int'], random_state=42
    )

    # 再从剩余的 85% 里划出验证集，使其占总体 15% → 0.15 / 0.85
    val_ratio_within_trainval = 0.15 / (1 - 0.15)  # = 0.1764705882

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio_within_trainval,
        stratify=train_val_df['label_int'], random_state=42
    )
    
    print("\nDataset split:")
    print(f"  Training set size: {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")
    print(f"  Test set size: {len(test_df)}")
    
    return train_df, val_df, test_df, class_to_idx, idx_to_class

import copy

def main():
    args = parse_args()
    init_distributed_mode(args)
    set_seed(args)
    #set_seed(42)
    # 获取 main.py 文件所在的目录的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 config 文件的完整路径
    config_path = os.path.join(current_script_dir, 'config', args.config)

    cfg = Config.fromfile(config_path)
    cfg.model.num_classes = 7 

    # --- TensorBoard 初始化 ---
    writer = None
    if is_main_process() and args.output_dir:
        # 日志将保存在 checkpoint/your_output_dir/runs/ 目录下
        tensorboard_log_dir = os.path.join(current_script_dir, 'checkpoint', args.output_dir, 'runs')
        os.makedirs(tensorboard_log_dir, exist_ok=True) # 确保目录存在
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
    # ---------------------------

    if args.output_dir:
        # 如果 output_dir 也在 current_script_dir 下，需要调整路径
        # 原来是 f'checkpoint/{args.output_dir}'，现在改成相对 current_script_dir
        output_checkpoint_dir = os.path.join(current_script_dir, 'checkpoint', args.output_dir)
        mkdir(output_checkpoint_dir)
        cfg.dump(os.path.join(output_checkpoint_dir, 'config.py'))
        print(f"Checkpoints will be saved to: {output_checkpoint_dir}")


    device = torch.device(args.device)
 
    #########################################################

    class_mapping = {
        'Calb_new': 'C.albicans',
        'Caur_new': 'C.auris',
        'Cgla_new': 'C.glabrata', 
        'Chae_new': 'C.haemulonii',     
        'Ckru_new':'krusei',
        'Cparap_new':'parapsilosis',
        "Ctro_new":'tropicalis'
    }
    print("--- Target Class Mapping ---")
    print(class_mapping)
    
    scatter_target_classes = list(class_mapping.keys())
    microscope_target_classes = list(class_mapping.values())

    # --- 2. 为光散射数据准备并【筛选】 ---
    print("\n--- Preparing and Filtering Scatter Data ---")
    cfg_scatter = copy.deepcopy(cfg)
    cfg_scatter.data.root_dir = cfg.data.path_scatter
    cfg_scatter.data.image_subfolder = 'cropped'
    train_s_raw, val_s_raw, test_s_raw, _, _ = prepare_cell_data(cfg_scatter) # 加载所有7类
    
    # 【关键】只保留我们需要的4个类的样本
    train_scatter_df = train_s_raw[train_s_raw['label_str'].isin(scatter_target_classes)].copy()
    val_scatter_df = val_s_raw[val_s_raw['label_str'].isin(scatter_target_classes)].copy()
    test_scatter_df = test_s_raw[test_s_raw['label_str'].isin(scatter_target_classes)].copy()

    # --- 3. 为显微镜数据准备并【筛选】 ---
    print("\n--- Preparing and Filtering Microscope Data ---")
    cfg_microscope = copy.deepcopy(cfg)
    cfg_microscope.data.root_dir = cfg.data.path_microscope
    cfg_microscope.data.image_subfolder = None
    train_m_raw, val_m_raw, test_m_raw, _, _ = prepare_cell_data(cfg_microscope)
    
    # 【关键】只保留我们需要的4个类的样本
    train_microscope_df_raw = train_m_raw[train_m_raw['label_str'].isin(microscope_target_classes)].copy()
    val_microscope_df_raw = val_m_raw[val_m_raw['label_str'].isin(microscope_target_classes)].copy()
    test_microscope_df_raw = test_m_raw[test_m_raw['label_str'].isin(microscope_target_classes)].copy()

    # --- 4. 重新建立统一的、只包含4个类的标签体系 ---
    print("\n--- Creating Final 4-Class Labeling System ---")
    unique_classes = sorted(scatter_target_classes)
    class_to_idx = {name: i for i, name in enumerate(unique_classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    
    # 更新 cfg，这是后续所有模块的num_classes来源
    cfg.model.num_classes = len(unique_classes)
    print(f"Final number of classes set to: {cfg.model.num_classes}")
    for name, idx in class_to_idx.items():
        print(f"  '{name}': {idx}")

    # --- 5. 对齐并重命名 ---
    # a. 对齐光散射数据的标签
    train_scatter_df['label_int'] = train_scatter_df['label_str'].map(class_to_idx)
    val_scatter_df['label_int'] = val_scatter_df['label_str'].map(class_to_idx)
    test_scatter_df['label_int'] = test_scatter_df['label_str'].map(class_to_idx)
    
    # b. 对齐显微镜数据的标签
    inverse_class_mapping = {v: k for k, v in class_mapping.items()}
    train_microscope_df_raw['label_str'] = train_microscope_df_raw['label_str'].map(inverse_class_mapping)
    val_microscope_df_raw['label_str'] = val_microscope_df_raw['label_str'].map(inverse_class_mapping)
    test_microscope_df_raw['label_str'] = test_microscope_df_raw['label_str'].map(inverse_class_mapping)

    train_microscope_df = train_microscope_df_raw
    val_microscope_df = val_microscope_df_raw
    test_microscope_df = test_microscope_df_raw
    
    train_microscope_df['label_int'] = train_microscope_df['label_str'].map(class_to_idx)
    val_microscope_df['label_int'] = val_microscope_df['label_str'].map(class_to_idx)
    test_microscope_df['label_int'] = test_microscope_df['label_str'].map(class_to_idx)
    
    # c. 重命名 path 列
    train_scatter_df.rename(columns={'path': 'path_scatter'}, inplace=True)
    val_scatter_df.rename(columns={'path': 'path_scatter'}, inplace=True)
    test_scatter_df.rename(columns={'path': 'path_scatter'}, inplace=True)
    
    train_microscope_df.rename(columns={'path': 'path_microscope'}, inplace=True)
    val_microscope_df.rename(columns={'path': 'path_microscope'}, inplace=True)
    test_microscope_df.rename(columns={'path': 'path_microscope'}, inplace=True)
    #########################################################

    cfg.model.num_classes = len(class_to_idx)

    model = build_model(cfg).to(device)

    model_without_ddp = model

    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
        train_dfs=(train_scatter_df, train_microscope_df),
        val_dfs=(val_scatter_df, val_microscope_df),
        test_dfs=(test_scatter_df, test_microscope_df),
        batch_size=cfg.data.batch_size_per_gpu,
        num_workers=cfg.data.num_workers,
        cfg=cfg,
        distributed=args.distributed
    )
    
    criterion = build_criterion(cfg, device)

    if args.eval:
        if args.resume and args.resume != '':
            try:
                ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt.get('model_ema', ckpt['model']))
                print(f"Loaded model from {args.resume} for evaluation.")
            except FileNotFoundError:
                print(f"ERROR: Checkpoint '{args.resume}' not found for evaluation.")
                return # 找不到检查点则退出评估模式
            except Exception as e:
                print(f"ERROR: Could not load checkpoint '{args.resume}' for evaluation. Error: {e}")
                return # 加载失败则退出
        else:
            print("ERROR: --resume path is required for evaluation mode.")
            return # 评估模式必须提供检查点路径

        # --- 评估：调用修改后的 evaluate 函数，移除 calc_map ---
        #model_ema.copy_to(model.parameters()) #新增ema
        model.eval()

        evaluate(
            cfg,
            model,
            test_dataloader, # 在评估模式下使用 test_dataloader
            device,
            epoch=0, # 评估通常不与 epoch 关联
            idx_to_class=idx_to_class, # 传递类别映射
            criterion = criterion,
        )
        return

    model_ema = None
    if args.model_ema:
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=args.model_ema_decay)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    #criterion = build_criterion(cfg, device)
    
    actual_lr = cfg.optimizer.lr * (cfg.data.batch_size_per_gpu * get_world_size()) / 8 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_without_ddp.parameters()),
        lr=actual_lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    scheduler = MultiStepLR(
        optimizer,
        milestones=cfg.scheduler.milestones,
        gamma=cfg.scheduler.gamma
    )
    
    if args.use_wandb and is_main_process():
        wandb.init(
            project='CellClassifier', 
            name=args.run_name,
            group=args.group_name,
            config=vars(args),
        )

    # load checkpoint for resume training
    max_track_metric = 0 
    if args.resume and args.resume != '': # 确保 args.resume 既非 None 也非空字符串
        try:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            args.start_epoch = checkpoint["epoch"] + 1
            max_track_metric = checkpoint.get("tracked_metric", 0) 
            if model_ema:
                model_ema.module.load_state_dict(checkpoint["model_ema"], strict=False)
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])
            print(f"Resumed training from epoch {args.start_epoch} with tracked metric: {max_track_metric:.4f}")
        except FileNotFoundError:
            print(f"WARNING: Checkpoint '{args.resume}' not found. Starting training from scratch.")
        except Exception as e:
            print(f"WARNING: Could not load checkpoint '{args.resume}'. Error: {e}. Starting training from scratch.")
    else:
        print("No resume checkpoint provided or path is empty. Starting training from scratch.")

    if args.fast_check:
        print("[Fast Debug] Using a small subset of data and few epochs.")
        args.epochs = 5
        cfg.data.batch_size_per_gpu = 1
        cfg.data.num_workers = 0

        # 从现有 DataFrames 中取 Subset
        from torch.utils.data import Subset
        train_df = train_df.sample(n=100, random_state=42) 
        val_df = val_df.sample(n=20, random_state=42)     
        test_df = test_df.sample(n=20, random_state=42)    

        # 重新构建 DataLoader
        train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
            train_df, val_df, test_df,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.num_workers,
            cfg=cfg,
            distributed=args.distributed
        )

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        if hasattr(train_dataloader.dataset, 'shuffle_partners'):
                    train_dataloader.dataset.shuffle_partners()

        log_info = train_one_epoch(
            args, 
            model,
            train_dataloader,
            criterion,
            optimizer,
            epoch,
            device,
            model_ema,
            scaler
        )

        scheduler.step() 

        if args.output_dir:
            output_checkpoint_path = os.path.join(current_script_dir, 'checkpoint', args.output_dir)
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "tracked_metric": max_track_metric, 
                "epoch": epoch,
                "args": args
            }

            if model_ema:
                checkpoint["model_ema"] = model_ema.module.state_dict()

            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()

            save_on_master(
                checkpoint,
                os.path.join(output_checkpoint_path, 'latest.pth'), 
            )

        try:
            if epoch >= args.start_eval:
                metrics = evaluate(
                    cfg,
                    model_ema or model,
                    val_dataloader, 
                    device,
                    epoch, 
                    idx_to_class=idx_to_class,
                    criterion = criterion
                )
                
                current_metric = metrics.get(args.metrics_to_track, 0) 

                log_info.update(metrics) 

                if max_track_metric < current_metric:
                    max_track_metric = current_metric

                    checkpoint = {
                        "model": model_without_ddp.state_dict() if not model_ema else model_ema.module.state_dict(),
                        "tracked_metric": max_track_metric, 
                        "epoch": epoch,
                    }
                    if args.output_dir:
                        save_on_master(
                            checkpoint,
                            os.path.join(output_checkpoint_path, 'best.pth'), 
                        )

        except NameError: 
            pass 

        if is_main_process() and args.use_wandb:
            wandb.log(
                log_info,
                step=epoch
            )
        
        # --- TensorBoard 日志记录 ---
        if writer and is_main_process():
            for key, value in log_info.items():
                if isinstance(value, (int, float)): # 确保只记录数值类型
                    writer.add_scalar(key, value, global_step=epoch)
            # 添加学习率到 TensorBoard
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], global_step=epoch)
        # -----------------------------

    # --- 训练结束时关闭 TensorBoard writer ---
    if writer:
        writer.close()
    # ------------------------------------------

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()