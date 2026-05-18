
import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

from transformers import PreTrainedModel
from timm import create_model
from .configuration_mambavision import MambaVisionConfig
from .EUCB import high_freq_perception_module_fft, EUCB_MultiKernel, HighFrequencyEnhancementFFT
from .fpn import CSFblock
import kornia
#from .shift_module import shift_module

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'mamba_vision_T': _cfg(url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_T2': _cfg(url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
                            crop_pct=0.98,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_S': _cfg(url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
                           crop_pct=0.93,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L2': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 224, 224),
                            crop_mode='center')                                
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=1, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.transformer_block = True
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x, x
        return self.downsample(x), x

class GatedSymmetryBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        flipped = torch.flip(x, dims=[-1])
        gate = self.gate(x)
        return x * gate + flipped * (1 - gate)

class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3, # 3
                 num_classes=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 use_hfe_branch=True,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def preprocess_to_powerspectrum(self, x):
        """
        将一批空间域图像转换为中心化的对数能量谱。
        x 的输入形状: (B, C, H, W)
        """
        # 确保输入是单通道灰度图
        if x.shape[1] > 1:
            # 如果输入是RGB图，先转为灰度图
            # 这里的转换方式可以根据需要调整，例如使用 torchvision.transforms.Grayscale
            x = torch.mean(x, dim=1, keepdim=True)

        # 1. 对实数输入进行傅里叶变换
        # 使用 ortho 范数可以保持信号能量不变
        x_ft = torch.fft.fft2(x, norm='ortho')

        # 2. 如果开启了中心化，则进行fftshift
        x_ft = torch.fft.fftshift(x_ft, dim=(-2, -1))

        # 3. 计算能量谱（幅度的平方）
        power_spectrum = x_ft.abs()**2
        
        # 4. 对数变换，以压缩动态范围，稳定训练
        log_power_spectrum = torch.log1p(power_spectrum)
        
        return log_power_spectrum
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        #log_power_spectrum = self.preprocess_to_powerspectrum(x)

        # 2. 对能量谱进行归一化 (Normalization)
        # 使用我们为能量谱数据集计算的专属均值和标准差
        # (B, 1, H, W) -> (B, 1, H, W)
        #x = (log_power_spectrum - self.spectrum_mean) / self.spectrum_std
        x = self.patch_embed(x)
        # --- 核心修改：在 STEM 之后应用 HFE ---
        #if self.use_hfe_branch:
            # 将 STEM 的输出送入 HFE 增强
        #x_hfe = self.hfe_branch(x)
            # 将原始特征与增强特征相加 (残差连接)
        #x = x_hfe + x 
        # ------------------------------------
        outs = []
        for level in self.levels:
            x, xo = level(x)
            outs.append(xo)
        #x = self.sym_block(x) #新增
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.pooling(x)
        return x, outs

    def forward(self, x):
        x, outs = self.forward_features(x)
        x = self.head(x)
        return x

    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)


class MambaVisionModel(PreTrainedModel):
    config_class = MambaVisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MambaVision(
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            dim=config.dim,
            in_dim=config.in_dim,
            mlp_ratio=config.mlp_ratio,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)


class MambaVisionModelForImageClassification(PreTrainedModel):
    config_class = MambaVisionConfig

    def __init__(self, config):
        super().__init__(config)
        
        # --- 核心修改：实例化我们修改过的 MambaVision ---
        # 我们可以从 config 中读取 HFE 的参数
        use_hfe = getattr(config, 'use_hfe_branch', True)
        self.model = MambaVision(
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            dim=config.dim,
            in_dim=config.in_dim,
            mlp_ratio=config.mlp_ratio,
            num_classes=config.num_labels, # <-- 确保 num_classes 也传入
            use_hfe_branch = use_hfe
        )

    def forward(self, tensor, labels=None):
        # 直接调用 MambaVision 的 forward，它现在返回 logits
        logits = self.model(tensor)
        
        if labels is not None:
            # Hugging Face 的标准做法是在模型内部计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}
            
        # 返回一个字典，以匹配你 engine.py 的期望
        return {"logits": logits}

# ----------------------------------------------------
# 1. 定义一个可复用的基础卷积块 (类似ResNet的残差块)
# ----------------------------------------------------
class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接的 shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果维度或通道数发生变化，需要用1x1卷积来匹配
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # 加上残差
        out = self.relu(out)
        return out

# ----------------------------------------------------
# 2. "YeastNet"
# ----------------------------------------------------
class YeastNet(nn.Module):
    def __init__(self, num_blocks, num_classes=7):
        super().__init__()
        self.in_channels = 64

        # Stem: 初始特征提取层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.hfe_branch = high_freq_perception_module_fft(
                 in_channels=64
            )
        self.enhenceFFT = HighFrequencyEnhancementFFT(64, 0.3)
        #self.hfe_branch = MultiBandPerceptionModule(in_channels=64)
        # Body: 堆叠四个阶段的卷积块
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        #self.shift = shift_module(1)
        # Head: 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5) # 加入Dropout防止过拟合
        self.fc = nn.Linear(512, num_classes)

        # 初始化权重
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ConvBlock2(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.hfe_branch(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.hfe_branch(x)
        #x = self.shift(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return {'logits': x}

def build_yeastnet_small(num_classes=7):
    # 类似 ResNet-18 的配置，参数量约 11M
    return YeastNet([2, 2, 2, 2], num_classes=num_classes)

def build_yeastnet_tiny(num_classes=7):
    # 一个更小的版本，参数量更少
    return YeastNet([1, 1, 2, 1], num_classes=num_classes)

class LayerNorm(nn.Module):
    """一个自定义的LayerNorm，支持(N, C, H, W)格式的输入"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class YeastNetBackbone(nn.Module):
    def __init__(self, num_blocks, use_hfe_branch=False):
        super().__init__()
        # 创建一个完整的YeastNet实例
        full_yeastnet = YeastNet(num_blocks=num_blocks, num_classes=7) # num_classes随便填
        
        # 将它的特征提取部分作为我们的属性
        self.stem = full_yeastnet.stem
        self.hfe_branch = full_yeastnet.hfe_branch # 假设你还在用它
        self.layer1 = full_yeastnet.layer1
        self.layer2 = full_yeastnet.layer2
        self.layer3 = full_yeastnet.layer3
        self.layer4 = full_yeastnet.layer4
        self.avgpool = full_yeastnet.avgpool
        
        self.output_dim = 512 # 我们知道YeastNet最终输出512维特征
        self.use_hfe_branch = use_hfe_branch
        if self.use_hfe_branch:
            # 确保 in_channels 与 stem 的输出匹配
            self.hfe_branch = high_freq_perception_module_fft(in_channels=64) 
        else:
            #self.hfe_branch = None 
            self.hfe_branch = nn.Identity()

    def _make_layer(self, out_channels, num_blocks, stride):
        """这个方法现在是 YeastNetBackbone 的一部分了。"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            # 确保 ConvBlock2 这个类在这里是可见的
            layers.append(ConvBlock2(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        """这个方法现在是 YeastNetBackbone 的一部分了。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        
        # --- 【核心修改】根据参数，决定是否使用 hfe_branch ---
        if self.use_hfe_branch and self.hfe_branch is not None:
            #x = self.pre_fft_denoise(x)
            x = self.hfe_branch(x)
            
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features

# ---- 基础工具：NCHW <-> NLC ----
def nchw_to_nlc(x):
    B, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
    return x, (H, W)

def nlc_to_nchw(x, hw):
    B, N, C = x.shape
    H, W = hw
    return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

def build_sincos2d(H, W, C, device):
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    pos = torch.stack([gy, gx], dim=-1).float()  # H,W,2
    dim = C // 2
    omega = torch.arange(dim, device=device) / dim
    omega = 1. / (10000 ** omega)  # dim
    pe_y = torch.einsum('hwc,c->hwc', pos[..., :1], omega)  # H,W,dim
    pe_x = torch.einsum('hwc,c->hwc', pos[..., 1:], omega)  # H,W,dim
    pe = torch.cat([torch.sin(pe_y), torch.cos(pe_y), torch.sin(pe_x), torch.cos(pe_x)], dim=-1)
    return pe[..., :C]  # H,W,C

# ---- Cross-Attention 2D：Q 来自 x_q，K/V 来自 x_kv ----
class CrossAttention2D(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0.0, proj_drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_kv = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
        self.gamma = nn.Parameter(torch.zeros(1))  # residual gate

    def forward(self, x_q, x_kv, pe_q=None, pe_kv=None):
        # x_*: [B,C,H,W]; pe_*: [H,W,C]
        B, C, Hq, Wq = x_q.shape
        B2, C2, Hk, Wk = x_kv.shape
        assert B == B2 and C == C2

        q, hw_q = nchw_to_nlc(x_q)
        k, hw_k = nchw_to_nlc(x_kv)
        v = k

        if pe_q is not None:  # 减少注释：位置编码加到 token 上
            q = q + pe_q.view(1, -1, C)
        if pe_kv is not None:
            k = k + pe_kv.view(1, -1, C)
            v = v + pe_kv.view(1, -1, C)

        q = self.norm1_q(q)
        k = self.norm1_kv(k); v = self.norm1_kv(v)

        # 多头
        def split_heads(t):
            B, N, C = t.shape
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B,h,N,d
        qh = split_heads(self.q(q))
        kh = split_heads(self.k(k))
        vh = split_heads(self.v(v))

        attn = (qh @ kh.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ vh  # B,h,Nq,d
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        out = self.proj_drop(self.proj(out))
        out = out + q  # 残差

        out = out + self.gamma * self.mlp(out)  # gated FFN
        return nlc_to_nchw(out, hw_q)


# ---- 双向 Co-Attention 层：micro<-scatter 与 scatter<-micro 交替/并行 ----
class CoAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, window=None):
        super().__init__()
        self.ca_ms = CrossAttention2D(dim, num_heads)  # micro guided by scatter
        self.ca_sm = CrossAttention2D(dim, num_heads)  # scatter guided by micro
        self.window = window  # 可选：窗口注意力减复杂度

    def forward(self, x_micro, x_scatter):
        B, C, H, W = x_micro.shape
        pe = build_sincos2d(H, W, C, x_micro.device)
        # 可选窗口：略；直接全局
        micro_upd = self.ca_ms(x_micro, x_scatter, pe, pe)
        scatter_upd = self.ca_sm(x_scatter, x_micro, pe, pe)
        return micro_upd, scatter_upd


class FreqGuidedCoAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, num_freq_bands=3):
        super().__init__()
        # --- 交叉注意力模块 ---
        self.norm_m1 = nn.LayerNorm(dim)
        self.norm_s1 = nn.LayerNorm(dim)
        self.cross_attn_m_to_s = CrossAttention2D(dim, num_heads) # Micro guided by Scatter
        
        self.norm_m2 = nn.LayerNorm(dim)
        self.norm_s2 = nn.LayerNorm(dim)
        self.cross_attn_s_to_m = CrossAttention2D(dim, num_heads) # Scatter guided by Micro

        # --- 【新】频域引导模块 ---
        self.freq_gate = FrequencyGate(in_channels=dim, num_bands=num_freq_bands)

    def forward(self, x_micro, x_scatter, pos_embed):
        # x_micro, x_scatter: [B, C, H, W]
        # pos_embed: [1, H*W, C]
        
        B, C, H, W = x_micro.shape
        x_micro_seq = x_micro.flatten(2).transpose(1, 2)
        x_scatter_seq = x_scatter.flatten(2).transpose(1, 2)
        
        # 1. 为两个模态的token注入位置信息
        x_micro_pos = x_micro_seq + pos_embed
        x_scatter_pos = x_scatter_seq + pos_embed
        
        # 2. 【核心创新】从光散射特征图(x_scatter)中提取频域引导信号
        freq_modulation = self.freq_gate(x_scatter) # -> [B, C]
        
        # 将调制信号变成一个门控，并扩展维度以便广播
        gate = torch.sigmoid(freq_modulation).unsqueeze(1) # -> [B, 1, C]

        # --- 交互 ---
        # a. 显微镜 '查询' 光散射 (被频域引导)
        q1 = self.norm_m1(x_micro_pos)
        k1 = self.norm_s1(x_scatter_pos)
        
        # 【核心创新】用门控来调制 Key
        k1_modulated = k1 * gate 
        
        micro_enhanced = self.cross_attn_m_to_s(q1, k1_modulated)
        x_micro_out = x_micro_seq + micro_enhanced
        
        # b. 光散射 '查询' 显微镜 (标准交叉注意力)
        q2 = self.norm_s2(x_scatter_pos)
        k2 = self.norm_m2(x_micro_pos)
        scatter_enhanced = self.cross_attn_s_to_m(q2, k2)
        x_scatter_out = x_scatter_seq + scatter_enhanced
        
        # 还原为特征图
        x_micro_out = x_micro_out.transpose(1, 2).reshape(B, C, H, W)
        x_scatter_out = x_scatter_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_micro_out, x_scatter_out

# --- ArcFace 头 ---
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):  # s=缩放，m=角度间隔
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # cosθ
        x = F.normalize(features, dim=1)
        W = F.normalize(self.weight, dim=1)
        cos = torch.mm(x, W.t())             # [B, C]

        if labels is None:                   # 推理
            return self.s * cos

        # 加 margin（ArcFace）
        theta = torch.acos(cos.clamp(-1+1e-7, 1-1e-7))
        target_logits = torch.cos(theta + self.m)

        logits = cos.clone()
        logits.scatter_(1, labels.view(-1, 1), target_logits.gather(1, labels.view(-1,1)))
        return self.s * logits               # 返回缩放后的 logits


class LayerNorm2d(nn.Module):
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

class MobileViTXS_L2(nn.Module):
    """
    取 MobileViT-XS 在 H/8 处的特征图 (对应 out_indices=2).
    输出张量形状: [B, C_mvit, H/8, W/8]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # features_only=True 可直接拿中间层特征
        # out_indices=(2,) 对应到 H/8 这一层（timm 内部已定义各 stage 的下采样比例）
        self.net = create_model(
            'mobilevit_xs',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2,)
        )
        # 该层的通道数（通常为 64，但以模型的实际声明为准）
        self.out_channels = self.net.feature_info.channels()[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)   # 是一个 list/tuple，长度为 1（因为 out_indices=(2,)）
        return feats[0]       # [B, C_mvit, H/8, W/8]

class YeastMMT3(nn.Module):
    def __init__(self, num_blocks, use_bicam=True, use_fgca=True, num_classes=7, fusion_dim=256, depth=2, num_heads=8):
        super().__init__()
        self.use_bicam = use_bicam
        self.use_fgca = use_fgca
        # --- a. 创建两个前端 ---
        # A: 光散射 (保留你原来的 YeastNetBackbone 到 layer2)
        # encoder_a_base = YeastNetBackbone(num_blocks, use_hfe_branch=True)
        # self.backbone_scatter = nn.Sequential(
        #     encoder_a_base.stem,
        #     encoder_a_base.hfe_branch,
        #     encoder_a_base.layer1,
        #     encoder_a_base.layer2   # -> [B, 128, H/8, W/8]
        # )
        # #cnn_out_dim_scatter = 128  # 原注释里写的 layer2 输出通道
        # encoder_b_base = YeastNetBackbone(num_blocks, use_hfe_branch=False)
        # self.backbone_microscope = nn.Sequential(
        #     encoder_b_base.stem,
        #     # 注意：这里需要 temp_yeastnet_b.hfe_branch (它是一个 nn.Identity)
        #     # 以保持模块索引的一致性，虽然直接用 layer1 也可以
        #     encoder_b_base.hfe_branch, 
        #     encoder_b_base.layer1,
        #     encoder_b_base.layer2
        # )
        # B: 显微镜 -> 换成 MobileViT-XS 的 H/8 特征
        self.backbone_microscope = MobileViTXS_L2(pretrained=True)
        self.backbone_scatter = MobileViTXS_L2(pretrained=True)      
        cnn_out_dim_micro = self.backbone_microscope.out_channels  
        cnn_out_dim_scatter = self.backbone_scatter.out_channels
        # --- b. 投影到统一的 fusion 维度 ---
        self.proj_scatter    = nn.Conv2d(cnn_out_dim_scatter, fusion_dim, kernel_size=1)
        self.proj_microscope = nn.Conv2d(cnn_out_dim_scatter,   fusion_dim, kernel_size=1)
        # cnn_out_dim = 128 # layer2 的输出通道数
        # self.proj_scatter = nn.Conv2d(cnn_out_dim, fusion_dim, kernel_size=1)
        # self.proj_microscope = nn.Conv2d(cnn_out_dim, fusion_dim, kernel_size=1)
        # --- c. 堆叠交互层（保持不变） ---
        self.co_attention_layers = nn.ModuleList(
            [CoAttentionLayer(fusion_dim, num_heads) for _ in range(depth)]
        )

        # 可选：你已有的频域引导 FGCA 块
        self.fgca_block = FGCA_Block(feature_dim=256, num_heads=8)

        # --- d. L3/L4 保持与你的 YeastNet 后续构建逻辑一致，并添加hfebranch支持 ---
        # 光散射分支的L3/L4层，使用hfebranch
        builder_a = YeastNetBackbone(num_blocks, use_hfe_branch=True)
        builder_a.in_channels = fusion_dim
        self.post3_scatter = builder_a._make_layer(256, num_blocks[2], stride=2)  # [B,256,H/16,W/16]
        self.post4_scatter = builder_a._make_layer(512, num_blocks[3], stride=2)  # [B,512,H/32,W/32]

        # 显微镜分支的L3/L4层，也使用hfebranch
        builder_b = YeastNetBackbone(num_blocks, use_hfe_branch=False)  
        builder_b.in_channels = fusion_dim
        self.post3_microscope = builder_b._make_layer(256, num_blocks[2], stride=2)  # [B,256,H/16,W/16]
        self.post4_microscope = builder_b._make_layer(512, num_blocks[3], stride=2)  # [B,512,H/32,W/32]

        # --- e. 你原有的 L3 处双向交互（保持不变） ---
        self.fgca_l3_micro_to_scatter = FGCA_Block(feature_dim=256, num_heads=8)
        self.cross_attn_l3_scatter_to_micro = nn.MultiheadAttention(256, 8, batch_first=True)
        self.norm_l3_scatter = nn.LayerNorm(256)
        self.norm_l3_micro   = nn.LayerNorm(256)

        # --- f. 分类头（保持不变） ---
        final_feature_dim = 512 + 512
        self.head = nn.Sequential(
            nn.BatchNorm1d(final_feature_dim),
            nn.Linear(final_feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.arc = ArcMarginProduct(512, num_classes, s=24.0, m=0.5) 
        self.feat = nn.Sequential(
            nn.BatchNorm1d(final_feature_dim),
            nn.Linear(final_feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
        )  
        self.gating_network = nn.Linear(final_feature_dim, 2)
        
        self.expert_difficult = ArcMarginProduct(
            in_features=512,        # embedding a-z
            out_features=3,         # 负责的类别数量
            s=24.0, m=0.5           # ArcFace的超参数 24 0.3
        )
        self.expert_easy = ArcMarginProduct(
            in_features=512,
            out_features=4,         # 负责的类别数量
            s=30.0, m=0.5
        )
        self.register_buffer('map_difficult', torch.tensor([2, 5, 6]))
        self.register_buffer('map_easy', torch.tensor([0, 1, 3, 4]))

    def forward(self, x, labels=None):
        img_scatter, img_microscope = x

        # 1) L2 前端
        fs = self.backbone_scatter(img_scatter)          # [B,128,H/8,W/8]
        fm = self.backbone_microscope(img_microscope)    # [B,C_mvit,H/8,W/8] 例如 C_mvit=64

        # 2) 投影到 fusion_dim 并在 L2 做若干层 co-attn
        fs_proj = self.proj_scatter(fs)                  # [B,fusion_dim,H/8,W/8]
        fm_proj = self.proj_microscope(fm)               # [B,fusion_dim,H/8,W/8]
        for layer in self.co_attention_layers:
            fm_proj, fs_proj = layer(fm_proj, fs_proj)

        # 3) 进入各自 L3
        fs_l3 = self.post3_scatter(fs_proj)              # [B,256,H/16,W/16]
        fm_l3 = self.post3_microscope(fm_proj)           # [B,256,H/16,W/16]
        # 4) 你原有的 L3 频域引导 + 反向交互（保持原逻辑）
        fm_vec = torch.flatten(self.avgpool(fm_l3), 1)                 # [B,256]
        fm_l3_enh = self.fgca_l3_micro_to_scatter(fm_vec, fs_l3)       # [B,256]

        fs_vec = torch.flatten(self.avgpool(fs_l3), 1)                 # [B,256]
        q  = fs_vec.unsqueeze(1)                                       # [B,1,256]
        kv = fm_vec.unsqueeze(1)                                       # [B,1,256]
        fs_attn, _ = self.cross_attn_l3_scatter_to_micro(q, kv, kv)    # [B,1,256]
        fs_l3_enh = self.norm_l3_scatter(fs_vec + fs_attn.squeeze(1))
        fm_gain = fm_l3_enh.unsqueeze(-1).unsqueeze(-1) / (fm_vec.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        fs_gain = fs_l3_enh.unsqueeze(-1).unsqueeze(-1) / (fs_vec.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        fm_l3 = fm_l3 * fm_gain.clamp(0.5, 2.0)
        fs_l3 = fs_l3 * fs_gain.clamp(0.5, 2.0)
        # 5) 进入各自 L4
        fs_final = self.post4_scatter(fs_l3)             # [B,512,H/32,W/32]
        fm_final = self.post4_microscope(fm_l3)          # [B,512,H/32,W/32]

        # 6) 池化、拼接、分类
        vec_s = torch.flatten(self.avgpool(fs_final), 1) # [B,512]
        vec_m = torch.flatten(self.avgpool(fm_final), 1) # [B,512]
        fused_vec = torch.cat([vec_s, vec_m], dim=1)     # [B,1024]

        routing_weights = F.softmax(self.gating_network(fused_vec), dim=-1) # [B, 2]
        log_pi_d = torch.log(routing_weights[:, 0:1] + 1e-8)  # [B,1]
        log_pi_e = torch.log(routing_weights[:, 1:2] + 1e-8)  # [B,1]

        emb = self.feat(fused_vec)
        
        z_d = self.expert_difficult(emb, labels=None) # [B,3]
        z_e = self.expert_easy(emb, labels=None)  # [B,4]
        # ---------- 6) 将专家 logits 对齐到全局 7 类并做 logsumexp 融合（CE 友好） ----------
        B = emb.size(0)
        # 用一个足够小的常数当作 -inf（兼容 AMP）
        NEG_INF = z_d.new_tensor(-1e9)

        big_d = z_d.new_full((B, 7), NEG_INF)                           # [B,7]
        big_e = z_e.new_full((B, 7), NEG_INF)
        big_d.scatter_(1, self.map_difficult.repeat(B, 1), z_d + log_pi_d)  # 对应类加上 log π_d
        big_e.scatter_(1, self.map_easy.repeat(B, 1),      z_e + log_pi_e)

        fused_logits = torch.logsumexp(torch.stack([big_d, big_e], dim=0), dim=0)  # [B,7]

        return {
            'logits': fused_logits,                # ← 直接喂 CrossEntropyLoss
            'emb': emb,
        }

    def extract_features(self, x):
        """
        返回：
            feat_ls:  LS 分支 embedding [B, 512]
            feat_mic: Mic 分支 embedding [B, 512]
            feat_fused: 融合 embedding (emb) [B, D]
        """
        img_scatter, img_microscope = x

        # 1) L2
        fs = self.backbone_scatter(img_scatter)
        fm = self.backbone_microscope(img_microscope)

        fs_proj = self.proj_scatter(fs)
        fm_proj = self.proj_microscope(fm)

        for layer in self.co_attention_layers:
            fm_proj, fs_proj = layer(fm_proj, fs_proj)

        # 3) L3
        fs_l3 = self.post3_scatter(fs_proj)
        fm_l3 = self.post3_microscope(fm_proj)

        # 4) FGCA + 反向 cross-attn
        fm_vec = torch.flatten(self.avgpool(fm_l3), 1)
        fm_l3_enh = self.fgca_l3_micro_to_scatter(fm_vec, fs_l3)

        fs_vec = torch.flatten(self.avgpool(fs_l3), 1)
        q  = fs_vec.unsqueeze(1)
        kv = fm_vec.unsqueeze(1)
        fs_attn, _ = self.cross_attn_l3_scatter_to_micro(q, kv, kv)
        fs_l3_enh = self.norm_l3_scatter(fs_vec + fs_attn.squeeze(1))

        fm_gain = fm_l3_enh.unsqueeze(-1).unsqueeze(-1) / (fm_vec.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        fs_gain = fs_l3_enh.unsqueeze(-1).unsqueeze(-1) / (fs_vec.unsqueeze(-1).unsqueeze(-1) + 1e-6)

        fm_l3 = fm_l3 * fm_gain.clamp(0.5, 2.0)
        fs_l3 = fs_l3 * fs_gain.clamp(0.5, 2.0)

        # 5) L4
        fs_final = self.post4_scatter(fs_l3)
        fm_final = self.post4_microscope(fm_l3)

        # 6) 全局特征
        feat_ls  = torch.flatten(self.avgpool(fs_final), 1)  # [B,512]
        feat_mic = torch.flatten(self.avgpool(fm_final), 1)  # [B,512]

        fused_vec = torch.cat([feat_ls, feat_mic], dim=1)    # [B,1024]
        feat_fused = self.feat(fused_vec)                    # [B, D]

        return feat_ls, feat_mic, feat_fused

# ==============================================================================
# 3. 最终的构建函数
# ==============================================================================
def build_deep_fusion_yeastnet(
        num_classes=7,
        use_bicam=True,
        use_fgca=True):

    return YeastMMT3(
        num_blocks=[2, 2, 2, 2],   # backbone 层配置
        num_classes=num_classes,
        fusion_dim=256,
        depth=2,
        num_heads=2,
        use_bicam=use_bicam,
        use_fgca=use_fgca,
    )

class FrequencyGate(nn.Module):
    def __init__(self, in_channels, num_bands=3):
        """
        从特征图的频域信息中学习一个门控/调制信号。
        
        Args:
            in_channels (int): 输入特征图的通道数.
            num_bands (int): 要分析的频带数量 (e.g., 3 for low, mid, high).
        """
        super().__init__()
        self.num_bands = num_bands
        
        # 定义频带的边界 (这些可以作为超参数调整)
        # 这里我们将频谱半径分为 num_bands 个等距的环带
        self.band_ratios = torch.linspace(0.0, 1.0, num_bands + 1)
        
        # 一个小型的MLP，将频带能量转换为一个调制向量
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels * num_bands, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. FFT
        fft_map_shifted = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        
        # 计算幅度谱
        magnitude_map = torch.abs(fft_map_shifted)
        
        # 2. 计算每个频带的能量
        band_energies = []
        center_h, center_w = H // 2, W // 2
        
        for i in range(self.num_bands):
            # 创建一个环形掩码
            inner_radius_h = int(center_h * self.band_ratios[i])
            inner_radius_w = int(center_w * self.band_ratios[i])
            outer_radius_h = int(center_h * self.band_ratios[i+1])
            outer_radius_w = int(center_w * self.band_ratios[i+1])
            
            mask = torch.zeros_like(magnitude_map)
            mask[:, :, center_h - outer_radius_h : center_h + outer_radius_h, 
                      center_w - outer_radius_w : center_w + outer_radius_w] = 1.0
            if i > 0:
                mask[:, :, center_h - inner_radius_h : center_h + inner_radius_h, 
                          center_w - inner_radius_w : center_w + inner_radius_w] = 0.0

            # 计算掩码区域内的平均能量
            # (masked_sum / mask_count) -> [B, C]
            masked_energy = magnitude_map * mask
            band_energy_per_channel = masked_energy.sum(dim=(-2, -1)) / (mask.sum(dim=(-2,-1)) + 1e-6)
            band_energies.append(band_energy_per_channel)
        
        # 3. 将所有频带的能量拼接起来
        all_band_energies = torch.cat(band_energies, dim=1) # -> [B, C * num_bands]
        
        # 4. 通过MLP生成最终的调制向量
        modulation_vector = self.gate_mlp(all_band_energies) # -> [B, C]
        
        return modulation_vector

def add_pe(x_4d):  # x_4d: [B, C, H, W]
        B, C, H, W = x_4d.shape
        pe = build_sincos2d(H, W, C, x_4d.device)             # [H,W,C]
        return x_4d + pe.permute(2,0,1).unsqueeze(0)           # [B,C,H,W]

class FGCA_Block(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        
        # --- 交叉注意力模块 ---
        # 显微镜特征 '查询' 光散射特征
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.freq_gate = FrequencyGate(in_channels=feature_dim, num_bands=3)
        
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_kv = nn.LayerNorm(feature_dim) # 对展平后的kv进行归一化
        self.norm_out = nn.LayerNorm(feature_dim)

    def forward(self, query_feat_2d, kv_feat_4d):
        """
        Args:
            query_feat_2d (Tensor): 查询模态的特征 (e.g., 显微镜), shape [B, D]
            kv_feat_4d (Tensor): 键/值模态的特征图 (e.g., 光散射), shape [B, D, H, W]
        """
        B, D, H, W = kv_feat_4d.shape
        
        # 1. 从光散射特征图(kv_feat_4d)中提取频域引导信号
        freq_modulation = self.freq_gate(kv_feat_4d) # -> [B, D]
        
        # 2. 准备Q, K, V
        query = self.norm_q(query_feat_2d).unsqueeze(1) # -> [B, 1, D]
        
        # 将4D特征图展平为序列
        kv_flat = kv_feat_4d.flatten(2).transpose(1, 2) # -> [B, H*W, D]
        kv_norm = self.norm_kv(kv_flat)
        
        #    我们用 sigmoid 将调制向量转换为 0-1 的门控值
        #    然后用它来加权 Key
        gate = torch.sigmoid(freq_modulation).unsqueeze(1) # -> [B, 1, D]
        
        key_modulated = kv_norm * gate # 广播: [B, H*W, D] * [B, 1, D]
        value = kv_norm # Value 通常不被调制
        
        # 4. 执行交叉注意力
        attn_output, _ = self.cross_attention(query, key_modulated, value) # Q, K, V
        
        # 5. 残差连接
        output = self.norm_out(query_feat_2d + attn_output.squeeze(1))
        
        return output

