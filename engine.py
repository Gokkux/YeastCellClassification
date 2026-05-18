
import sys
import math
import prettytable as pt 
from .utils import * 
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np 


def _unwrap(m):
    # 处理 DDP/EMA 包裹，拿到真实模型
    if hasattr(m, 'module'):   # DDP
        return m.module
    return m

def train_one_epoch(
        args,
        model,
        train_loader,
        criterion, # 现在是 nn.CrossEntropyLoss
        optimizer,
        epoch,
        device,
        model_ema=None,
        scaler=None
):
    model.train()
    total_loss = 0.0            # ← 新增：按样本累计
    total_samples = 0           # ← 新增：样本计数
    criterion.train() # 虽然 nn.CrossEntropyLoss 没有 train() 方法，但保持结构一致性无害

    log_info = dict()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=args.print_freq, fmt="{value:.4f}")) # 新增loss meter
    header = f"Epoch: [{epoch}]"

    for data_iter_step, (data, labels) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        images_a, images_b = data
        images_a = images_a.to(device, non_blocking=True)
        images_b = images_b.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)
        model_input = (images_a, images_b)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(model_input)['logits']
            loss = criterion(logits, labels)   

        if is_dist_avail_and_initialized():
            # 把本 step 的平均损失在所有进程上取 mean，便于日志显示
            avg_loss = loss.detach().clone()
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= torch.distributed.get_world_size()
            loss_for_log = avg_loss.item()
        else:
            loss_for_log = loss.item()

        # ——【样本加权累计】用于 epoch 级别的 train_loss 统计
        bs = labels.size(0)
        total_loss    += loss.item() * bs      # ← 关键：按样本数累计
        total_samples += bs

        # 反向与优化
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        if model_ema and data_iter_step % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.warmup_epochs:
                model_ema.n_averaged.fill_(0)
        metric_logger.update(loss=loss_for_log, lr=optimizer.param_groups[0]["lr"])

    # ====== epoch 末尾：做一次全局归并并计算“样本级平均训练损失” ======
    if is_dist_avail_and_initialized():
        tl = torch.tensor([total_loss],    dtype=torch.float64, device=device)
        ts = torch.tensor([total_samples], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(tl, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.SUM)
        total_loss, total_samples = tl.item(), ts.item()

    train_loss_epoch = total_loss / max(1, total_samples)   # ← 这是 0.x～1.x 量级
    log_info = {"loss": float(train_loss_epoch)}            # ← 返回“样本级平均”的 epoch loss
    return log_info

# --- 辅助函数：用于分布式环境中聚合标量损失 ---
def reduce_scalar(scalar_tensor):
    if not is_dist_avail_and_initialized():
        return scalar_tensor
    
    # 假设 dist 已经导入并初始化
    # import torch.distributed as dist
    world_size = dist.get_world_size()
    if world_size < 2:
        return scalar_tensor
    
    # 克隆一份，因为 all_reduce 会修改传入的 tensor
    reduced_scalar = scalar_tensor.clone()
    dist.all_reduce(reduced_scalar, op=dist.ReduceOp.SUM)
    return reduced_scalar / world_size # 返回平均值

from prettytable import PrettyTable as PT
@torch.inference_mode()

def evaluate(
        cfg,
        model,
        test_loader,
        device,
        epoch=0, # 保持 epoch 参数
        # calc_map=False, # 移除此参数
        idx_to_class: dict = None,
        criterion = None,
):

    model.eval()

    # --- 获取类别名称和数量 ---
    # 假设 cfg.model.num_classes 已经在 main.py 中被 prepare_cell_data 设置
    num_classes = cfg.model.num_classes 
    
    # 获取有序的类别名称列表
    class_names = [idx_to_class[i] for i in range(num_classes)] if idx_to_class else [f'Class {i}' for i in range(num_classes)]
    
    all_preds = []
    all_labels = []
    all_probs = [] 

    total_val_loss = 0.0

    epoch_iterator = tqdm(test_loader, file=sys.stdout, desc="Eval (X / X Steps)",
                          dynamic_ncols=True, disable=not is_main_process())

    for data_iter_step, (data, labels) in enumerate(epoch_iterator):
        
        # --- 【核心修改 2】: 将元组解包并移动到设备 ---
        images_a, images_b = data
        images_a = images_a.to(device)
        images_b = images_b.to(device)
        
        labels = labels.to(device)

        model_input = (images_a, images_b)
        outputs = model(model_input)
        logits = outputs['logits'] # 新增

        routing_weights = outputs.get('routing_weights', None)
        if routing_weights is not None:
            for k in [0,1,2,3,4,5,6]:   # 遍历每个类别
                mask = (labels == k)
                if mask.any():
                    mean_gate = routing_weights[mask].mean(0).detach().cpu().numpy()
                    print(f"[Eval][class {k}] mean routing = {mean_gate}")

        # --- 新增：计算验证损失 ---
        if criterion is not None:
            loss = criterion(logits, labels)
            reduced_loss = reduce_scalar(loss) if is_dist_avail_and_initialized() else loss
            total_val_loss += reduced_loss.item()
        # ---------------------------
        _, predicted = torch.max(logits, 1) # 新增
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        epoch_iterator.set_description(f"Epoch={epoch}: Eval ({data_iter_step+1} / {len(test_loader)} Steps)")

    # --- 分布式聚合所有预测和标签 ---
    if get_world_size() > 1:
        all_preds_gathered = list(itertools.chain.from_iterable(all_gather(torch.tensor(all_preds, device=device))))
        all_labels_gathered = list(itertools.chain.from_iterable(all_gather(torch.tensor(all_labels, device=device))))
        
        all_preds = np.array([x.item() for x in all_preds_gathered]) if isinstance(all_preds_gathered[0], torch.Tensor) else np.array(all_preds_gathered)
        all_labels = np.array([x.item() for x in all_labels_gathered]) if isinstance(all_labels_gathered[0], torch.Tensor) else np.array(all_labels_gathered)

    else: # 单进程模式下，all_preds 和 all_labels 已经是 numpy 数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

    # --- 计算图像分类指标 ---
    # 1. 总体准确率
    avg_val_loss = total_val_loss / len(test_loader) if criterion is not None and len(test_loader) > 0 else 0.0
    accuracy = accuracy_score(all_labels, all_preds) * 100 

    # 2. 精确率、召回率、F1-Score (每类和宏平均)
    precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes), average=None, zero_division=0)
    
    macro_precision = precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes), average='macro', zero_division=0)[0] * 100
    macro_recall = precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes), average='macro', zero_division=0)[1] * 100
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes), average='macro', zero_division=0)[2] * 100

    # 3. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # --- 打印评估结果 ---
    metrics_table = pt.PrettyTable()
    metrics_table.field_names = ["Class", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    
    # 打印每一类的性能
    for i in range(num_classes):
        metrics_table.add_row([
            class_names[i], # 使用真实的类别名称
            f"{precision_per_class[i]*100:.2f}",
            f"{recall_per_class[i]*100:.2f}",
            f"{f1_per_class[i]*100:.2f}"
        ])
    
    metrics_table.add_row(["---"] * len(metrics_table.field_names)) # 分隔线
    
    # 打印宏平均性能
    metrics_table.add_row([
        "Macro Avg",
        f"{macro_precision:.2f}",
        f"{macro_recall:.2f}",
        f"{macro_f1:.2f}"
    ])
    
    print("\n" + metrics_table.get_string()) # 打印美化的表格

    print("\nOverall Accuracy:", f"{accuracy:.2f}%")
    print("Confusion Matrix:\n", cm)


    # --- 返回指标字典 ---
    metrics_dict = {
        'val_loss': avg_val_loss,
        'accuracy': accuracy,
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
        'f1_macro': macro_f1,
    }
    
    # 添加每一类的指标到返回字典 (可选，如果 main.py 需要细粒度日志)
    for i in range(num_classes):
        metrics_dict[f'precision_{class_names[i]}'] = precision_per_class[i] * 100
        metrics_dict[f'recall_{class_names[i]}'] = recall_per_class[i] * 100
        metrics_dict[f'f1_{class_names[i]}'] = f1_per_class[i] * 100

    return metrics_dict
    #return metrics_dict, all_probs, np.asarray(all_labels) 
