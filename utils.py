import random
import torch.distributed as dist

import numpy as np
import scipy.spatial as S
import torchvision.transforms as T

import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def get_tp(
        pred_points,
        pred_scores,
        gd_points,
        thr=12,
        return_index=False
):
    sorted_pred_indices = np.argsort(-pred_scores)
    sorted_pred_points = pred_points[sorted_pred_indices]

    unmatched = np.ones(len(gd_points), dtype=bool)
    dis = S.distance_matrix(sorted_pred_points, gd_points)

    for i in range(len(pred_points)):
        min_index = dis[i, unmatched].argmin()
        if dis[i, unmatched][min_index] <= thr:
            unmatched[np.where(unmatched)[0][min_index]] = False

        if not np.any(unmatched):
            break

    if return_index:
        return sum(~unmatched), np.where(unmatched)[0]
    else:
        return sum(~unmatched)


def point_nms(points, scores, classes, nms_thr=-1):
    _reserved = np.ones(len(points), dtype=bool)
    dis_matrix = S.distance_matrix(points, points)
    np.fill_diagonal(dis_matrix, np.inf)

    for idx in np.argsort(-scores):
        if _reserved[idx]:
            _reserved[dis_matrix[idx] <= nms_thr] = False

    points = points[_reserved]
    scores = scores[_reserved]
    classes = classes[_reserved]

    return points, scores, classes


def set_seed(args):
    seed = args.seed
    #seed = 42
    # seed = args.seed + get_rank()
    #torch.use_deterministic_algorithms(True, warn_only=True)
    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pre_processing(img):
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return trans(img).unsqueeze(0)


@torch.no_grad()
def predict(
        model,
        image,
        nms_thr=-1,
        ori_shape=None,
        filtering=False
):
    ori_h, ori_w = ori_shape
    outputs = model(image)

    points = outputs['pred_coords'][0].cpu().numpy()
    scores = outputs['pred_logits'][0].softmax(-1).cpu().numpy()
    classes = np.argmax(scores, axis=-1)

    np.clip(points[:, 0], a_min=0, a_max=ori_w - 1, out=points[:, 0])
    np.clip(points[:, 1], a_min=0, a_max=ori_h - 1, out=points[:, 1])
    valid_flag = classes < (scores.shape[-1] - 1)

    points = points[valid_flag]
    scores = scores[valid_flag].max(1)
    classes = classes[valid_flag]

    mask = outputs['pred_masks'][0, 0].cpu().numpy() > 0

    if filtering:
        valid_flag = mask[points.astype(int)[:, 1], points.astype(int)[:, 0]]
        points = points[valid_flag]
        scores = scores[valid_flag]
        classes = classes[valid_flag]

    if len(points) and nms_thr > 0:
        points, scores, classes = point_nms(points, scores, classes, nms_thr)

    return points, scores, classes, mask


def collate_fn(batch):
    images, points, labels, masks = [[] for _ in range(4)]
    for x in batch:
        images.append(x[0])
        points.append(x[1])
        labels.append(x[2])
        masks.append(x[3])
    return torch.stack(images), torch.stack(masks), points, labels


# utils/feature_probe.py
import os, torch
import torch.fft as fft
from torchvision.utils import make_grid, save_image
import os
import torch
import torch.fft as fft
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import cv2
class FeatureProbe:
    def __init__(self, save_dir="vis_feats"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.hooks = []
        self.features = {}

    # ✅ 实例方法（非 static），并在调用处用 self._norm01(...)
    def _norm01(self, v: torch.Tensor) -> torch.Tensor:
        v = v - v.min()
        vmax = v.max()
        if vmax > 0:
            v = v / vmax
        return v

    def save_colored_heatmap(
        self, 
        feat: torch.Tensor, 
        name: str, 
        colormap: str = 'jet',
        original_image: torch.Tensor = None,
        alpha: float = 0.5
    ):
        """
        将特征图转换为彩色热力图并保存。可以选择性地与原始图像叠加。

        Args:
            feat (torch.Tensor): 要可视化的特征图，形状为 [B, C, H, W]。
            name (str): 保存的文件名 (例如 'heatmap.png')。
            colormap (str, optional): Matplotlib的颜色映射名称。
                                    常用: 'jet', 'viridis', 'inferno', 'magma'. 
                                    Defaults to 'jet'.
            original_image (torch.Tensor, optional): 原始输入图像，用于叠加。
                                                    形状为 [B, C, H, W]，值范围 [0, 1]。
                                                    Defaults to None.
            alpha (float, optional): 叠加时热力图的透明度。Defaults to 0.5.
        """
        path = os.path.join(self.save_dir, name)
        
        # 1. 准备特征图
        # 取第一个样本，计算通道均值，并归一化到 [0, 1]
        heatmap = feat[2].detach().mean(dim=0).cpu() # [H, W]
        heatmap = self._norm01(heatmap)
        heatmap_numpy = heatmap.numpy() # 转换为 numpy 数组

        # 2. 应用颜色映射
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(heatmap_numpy)[:, :, :3] # [H, W, 3], RGBA -> RGB
        colored_heatmap = torch.from_numpy(colored_heatmap).permute(2, 0, 1) # [3, H, W]
        
        # 如果没有提供原始图像，直接保存彩色热力图
        if original_image is None:
            save_image(colored_heatmap, path)
            return

        # 3. 如果提供了原始图像，进行叠加
        # 准备原始图像
        img_to_overlay = original_image[2].detach().cpu() # [C, H, W]
        
        # 如果原始图像是单通道的灰度图，先转换为3通道
        if img_to_overlay.shape[0] == 1:
            img_to_overlay = img_to_overlay.repeat(3, 1, 1)

        # 确保热力图和原始图像尺寸一致
        # 使用OpenCV进行插值，因为它比PyTorch的interpolate更方便
        if colored_heatmap.shape[1:] != img_to_overlay.shape[1:]:
            h, w = img_to_overlay.shape[1:]
            # 将PyTorch张量转换为OpenCV格式 (H, W, C)
            heatmap_to_resize = colored_heatmap.permute(1, 2, 0).numpy()
            # 使用双线性插值进行缩放
            resized_heatmap_cv = cv2.resize(heatmap_to_resize, (w, h), interpolation=cv2.INTER_LINEAR)
            # 转回PyTorch张量 (C, H, W)
            colored_heatmap = torch.from_numpy(resized_heatmap_cv).permute(2, 0, 1)

        # 4. 混合图像
        # overlay = alpha * heatmap + (1 - alpha) * image
        overlayed_image = (alpha * colored_heatmap + (1 - alpha) * img_to_overlay)
        
        # 保存叠加后的图像
        save_image(overlayed_image, path)


    def _save_channel_grid(self, feat: torch.Tensor, name: str, nrow=8, max_ch=32):
        """
        将 [B,C,H,W] 的特征，取第一个样本的前 max_ch 个通道，
        每通道单独归一化后，按网格拼成“多张灰度图”并保存。
        """
        path = os.path.join(self.save_dir, name)
        x = feat[2:3].detach().float().cpu().clone()   # [1,C,H,W]
        x = x[0]                                      # [C,H,W]
        C = min(x.shape[0], max_ch)
        x = x[:C]

        # 每通道独立归一化 -> [C,1,H,W]
        x_norm = []
        for i in range(C):
            ch = self._norm01(x[i])
            x_norm.append(ch.unsqueeze(0))            # [1,H,W]
        x_norm = torch.stack(x_norm, dim=0)           # [C,1,H,W]

        grid = make_grid(x_norm, nrow=nrow, padding=2)  # [3,Hg,Wg]
        save_image(grid, path)

    def _save_mean_heat(self, feat: torch.Tensor, name: str):
        """
        通道均值热图：先对通道求均值 -> [B,1,H,W]，再保存 0-1 灰度图。
        """
        path = os.path.join(self.save_dir, name)
        x = feat[2:3].detach().float().cpu()           # [1,C,H,W]
        m = x.mean(dim=1, keepdim=True)               # [1,1,H,W]
        m = self._norm01(m[0, 0]).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        save_image(m, path)

    def _save_logspec(self, feat: torch.Tensor, name: str):
        """
        频谱可视化：对通道求均值 -> FFT -> log(1+|.|) -> 0-1 归一 -> 灰度保存。
        """
        path = os.path.join(self.save_dir, name)
        x = feat[2:3].detach().float().cpu().mean(dim=1, keepdim=True)  # [1,1,H,W]
        spec = fft.fftshift(fft.fft2(x, norm='ortho'), dim=(-2, -1))
        mag = torch.log1p(spec.abs())                                  # [1,1,H,W]
        v = self._norm01(mag[0, 0]).unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
        save_image(v, path)

    def hook(self, module, key: str):
        """
        在 module 上注册 forward hook，输出保存在 self.features[key]
        """
        def _cb(m, inp, out):
            self.features[key] = out.detach()
        self.hooks.append(module.register_forward_hook(_cb))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.features.clear()

    def compare(
            self, 
            key_in: str, 
            key_out: str, 
            prefix="hfe",
            original_image: torch.Tensor = None, # <-- 新增参数
            alpha: float = 0.5 # <-- 新增参数
        ):
            """
            对比同一位置的输入/输出特征。
            - 新增: 生成彩色的热力图和叠加图。
            """
            assert key_in in self.features and key_out in self.features, \
                f"Missing features: have {list(self.features.keys())}, need {key_in}/{key_out}"
            if original_image is not None:
                # 取出批次中的第一张图
                img_to_save = original_image[2:3] 
                
                # 检查图像是否需要从 [-1, 1] 范围恢复到 [0, 1]
                # (如果你的 Normalize 变换是 ToTensor() 后的标准变换)
                # 这是一个好习惯，但如果你的输入本身就在 [0, 1]，则可以省略
                if img_to_save.min() < 0:
                    img_to_save = img_to_save * 0.5 + 0.5 # [-1, 1] -> [0, 1]

                # 检查是否是单通道，如果是，save_image 会保存为灰度图
                # 如果想让参照图总是彩色的（即使是灰度），可以复制通道
                # if img_to_save.shape[1] == 1:
                #     img_to_save = img_to_save.repeat(1, 3, 1, 1)

                save_image(img_to_save, os.path.join(self.save_dir, f"{prefix}_original_input.png"))

            x_in  = self.features[key_in]
            x_out = self.features[key_out]

            diff  = (x_out - x_in).abs()
            gain  = (x_out / (x_in + 1e-8)).clamp(0, 3.0)

            # --- 1. 保存网格 (保持不变) ---
            self._save_channel_grid(x_in,  f"{prefix}_in_grid.png")
            self._save_channel_grid(x_out, f"{prefix}_out_grid.png")

            # --- 2. 保存灰度热图 (保持不变) ---
            self._save_mean_heat(x_in,     f"{prefix}_in_mean_gray.png") # 加个后缀以区分
            self._save_mean_heat(x_out,    f"{prefix}_out_mean_gray.png")
            self._save_mean_heat(diff,     f"{prefix}_absdiff_mean_gray.png")
            self._save_mean_heat(gain,     f"{prefix}_gain_mean_gray.png")

            # --- 3. 【新增】保存彩色热力图 ---
            # 只保存彩色热力图 (不叠加)
            self.save_colored_heatmap(x_in,     f"{prefix}_in_mean_color.png")
            self.save_colored_heatmap(x_out,    f"{prefix}_out_mean_color.png")
            self.save_colored_heatmap(diff,     f"{prefix}_absdiff_mean_color.png")
            self.save_colored_heatmap(gain,     f"{prefix}_gain_mean_color.png", colormap='magma') # 增益图换个颜色

            # 【新增】如果提供了原始图像，则保存叠加图
            if original_image is not None:
                self.save_colored_heatmap(x_in,  f"{prefix}_in_overlay.png",  original_image=original_image, alpha=alpha)
                self.save_colored_heatmap(x_out, f"{prefix}_out_overlay.png", original_image=original_image, alpha=alpha)
                self.save_colored_heatmap(diff,  f"{prefix}_diff_overlay.png", original_image=original_image, alpha=alpha)
                # 增益图通常不适合叠加，因为它表示的是比例关系，所以这里省略

            # --- 4. 保存频谱 (保持不变) ---
            self._save_logspec(x_in,       f"{prefix}_in_logspec.png")
            self._save_logspec(x_out,      f"{prefix}_out_logspec.png")
            self._save_logspec(x_out-x_in, f"{prefix}_delta_logspec.png")

    # def compare(self, key_in: str, key_out: str, prefix="hfe"):
    #     """
    #     对比同一位置的输入/输出特征（例如 HFE 模块的输入/输出）：
    #     - 通道网格图（前 max_ch 个）
    #     - 通道均值热图（in/out/abs diff/gain）
    #     - 频谱图（in/out/delta）
    #     """
    #     assert key_in in self.features and key_out in self.features, \
    #         f"Missing features: have {list(self.features.keys())}, need {key_in}/{key_out}"

    #     x_in  = self.features[key_in]
    #     x_out = self.features[key_out]

    #     # 差异/增益
    #     diff  = (x_out - x_in).abs()
    #     gain  = (x_out / (x_in + 1e-6)).clamp(0, 3.0)

    #     # 保存网格
    #     self._save_channel_grid(x_in,  f"{prefix}_in_grid.png")
    #     self._save_channel_grid(x_out, f"{prefix}_out_grid.png")

    #     # 保存热图
    #     self._save_mean_heat(x_in,     f"{prefix}_in_mean.png")
    #     self._save_mean_heat(x_out,    f"{prefix}_out_mean.png")
    #     self._save_mean_heat(diff,     f"{prefix}_absdiff_mean.png")
    #     self._save_mean_heat(gain,     f"{prefix}_gain_mean.png")

    #     # 保存频谱
    #     self._save_logspec(x_in,       f"{prefix}_in_logspec.png")
    #     self._save_logspec(x_out,      f"{prefix}_out_logspec.png")
    #     self._save_logspec(x_out-x_in, f"{prefix}_delta_logspec.png")


# ===== gradcam_vis.py (或直接粘到 engine.py 顶部 imports 下面) =====
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def _denorm(img: torch.Tensor, mean=_IMAGENET_MEAN, std=_IMAGENET_STD):
    """img: [B,3,H,W], 反归一化到 [0,1] 区间"""
    mean = torch.tensor(mean, device=img.device).view(1,3,1,1)
    std  = torch.tensor(std,  device=img.device).view(1,3,1,1)
    x = img * std + mean
    return x.clamp(0, 1)

def _normalize_01(t: torch.Tensor):
    t = t - t.min()
    tmax = t.max()
    if tmax > 0: t = t / tmax
    return t
class SimpleGradCAM:
    def __init__(self, model, target_layer, device=None):
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device
        self._acts = None
        self._grads = None
        self._fh = None
        self._bh = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(m, inp, out):
            self._acts = out  # 不要 detach，保持图
        def bwd_hook(m, grad_in, grad_out):
            self._grads = grad_out[0]
        self._fh = self.target_layer.register_forward_hook(fwd_hook)
        self._bh = self.target_layer.register_full_backward_hook(bwd_hook)

    @torch.no_grad()
    def _resize_cam(self, cam: torch.Tensor, size_hw):
        return F.interpolate(cam, size=size_hw, mode='bilinear', align_corners=False)

    def generate(self, inputs, target_class=None, return_cam=False,
                 save_path=None, overlay_on=None):
        self.model.eval()

        imgs_s, imgs_m = inputs
        imgs_s = imgs_s.to(self.device)
        imgs_m = imgs_m.to(self.device)

        # 🚫 千万别在 no_grad 里
        with torch.enable_grad():
            # 可选：关闭 autocast，避免半精度带来的反传问题
            # from torch.cuda.amp import autocast
            # with autocast(enabled=False):
            logits = self.model((imgs_s, imgs_m))['logits']  # [B,C]
            if target_class is None:
                target_class = logits.argmax(dim=1)
            elif isinstance(target_class, int):
                target_class = torch.full((logits.size(0),), target_class,
                                          dtype=torch.long, device=logits.device)

            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, target_class.view(-1,1), 1.0)

            self.model.zero_grad(set_to_none=True)
            (logits * one_hot).sum().backward(retain_graph=False)

            acts  = self._acts          # [B,C,h,w]
            grads = self._grads         # [B,C,h,w]

            weights = grads.mean(dim=(2,3), keepdim=True)     # [B,C,1,1]
            cam = (weights * acts).sum(dim=1, keepdim=True)   # [B,1,h,w]
            cam = F.relu(cam)

        # 叠加/保存
        if overlay_on is not None:
            H, W = overlay_on.shape[-2:]
        else:
            H, W = imgs_s.shape[-2:]
        cam = self._resize_cam(cam, (H, W))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

        if overlay_on is not None and save_path is not None:
            base = _denorm(overlay_on.to(self.device))
            heat = cam[0].expand_as(base)
            overlay = (0.5 * base + 0.5 * heat).clamp(0, 1)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(overlay, save_path)

        return cam[0,0].detach().cpu().numpy() if return_cam else None
