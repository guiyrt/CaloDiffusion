import math
import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from einops import rearrange
from functools import partial
import torch.nn.functional as F


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cosine_beta_schedule(nsteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    x = torch.linspace(0, nsteps, nsteps+1)
    alphas_cumprod = torch.cos(((x / nsteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class CylindricalConvTranspose(nn.Module):
    # Format of channels, zbin, phi_bin, rbin
    def __init__(self, dim_in, dim_out, kernel_size=(3,4,4), stride=(1,2,2), padding=1, output_padding=0):
        super().__init__()
        self.circ_pad = (0, 0, padding, padding, 0, 0)
        conv_transpose_pad = (padding, kernel_size[1]-1 ,padding)
        
        self.conv_transpose = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=conv_transpose_pad, output_padding=output_padding)

    def forward(self, x):
        # Out size is : O = (i-1)*S + K - 2P
        # To achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        # Pad last dim with nothing, 2nd to last dim is circular one
        return self.conv_transpose(F.pad(x, pad=self.circ_pad, mode='circular'))


class CylindricalConv(nn.Module):
    # Format of channels, zbin, phi_bin, rbin
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, groups=1, padding=0, bias=True):
        super().__init__()
        self.circ_pad = (0, 0, padding, padding, 0, 0)
        conv_pad = (padding, 0, padding)

        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, groups=groups, padding=conv_pad, bias=bias)

    def forward(self, x):
        # To achieve 'same' use padding P = ((S-1)*W-S+F)/2, with F = filter size, S = stride, W = input size
        # Pad last dim with nothing, 2nd to last dim is circular one
        return self.conv(F.pad(x, pad=self.circ_pad, mode='circular'))


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        halfdim = dim // 2

        self.dim_in = halfdim // 4
        self.hidden = nn.Sequential(nn.Linear(halfdim//2, halfdim), nn.GELU())
        self.out =  nn.Linear(halfdim, halfdim)

    def forward(self, time: torch.Tensor):
        freq = torch.exp(torch.arange(self.dim_in, device=time.device) * -(np.log(10_000)/(self.dim_in - 1)))
        pos = torch.einsum("t, e -> te", time, freq)

        return self.out(self.hidden(torch.cat((pos.sin(), pos.cos()), dim=-1)))

class MlpEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        halfdim = dim // 2

        self._in = nn.Sequential(nn.Linear(1, halfdim // 2), nn.GELU())
        self.hidden = nn.Sequential(nn.Linear(halfdim // 2, halfdim), nn.GELU())
        self.out = nn.Linear(halfdim, halfdim)

    def forward(self, time: torch.Tensor):
        return self.out(self.hidden(self._in(time)))


class ConvBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups:int = 8, cylindrical: bool = False):
        super().__init__()
        conv_type = CylindricalConv if cylindrical else nn.Conv3d

        self.conv = conv_type(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.norm(self.conv(x)))

class ResNetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim_in: int, dim_out: int, cond_dim: Optional[int] = None, groups: int = 8, cylindrical: bool = False):
        super().__init__()
        conv_type = CylindricalConv if cylindrical else nn.Conv3d

        self.block1 = ConvBlock(dim_in, dim_out, groups, cylindrical)
        self.block2 = ConvBlock(dim_out, dim_out, groups, cylindrical)
        self.cond_emb = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim_out)) if exists(cond_dim) else None
        self.res_conv = conv_type(dim_in, dim_out, kernel_size=1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None):
        h = self.block1(x)

        if exists(self.cond_emb) and exists(time):
            h += rearrange(self.cond_emb(time), "b c -> b c 1 1 1")

        return self.block2(h) + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, n_heads: int = 1, dim_head: int = 32, cylindrical: bool = False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.n_heads = n_heads

        hidden_dim = dim_head * n_heads
        conv_type = CylindricalConv if cylindrical else nn.Conv3d

        self.to_qkv = conv_type(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(conv_type(hidden_dim, dim, kernel_size=1), nn.GroupNorm(1, dim))


    def forward(self, x):
        _, _, l, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.n_heads), qkv)

        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        c = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", c, q)

        return self.to_out(rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.n_heads, x=l, y=h, z=w))

class PreNorm(nn.Module):
    def __init__(self, dim: int, layer: nn.Module):
        super().__init__()
        self.layer = layer
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.layer(self.norm(x))
    
class Residual(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x) + x


#up and down sample in 2 dims but keep z dimm

def Upsample(dim, extra_upsample = [0,0,0], cylindrical = False, compress_Z: bool = False):
    Z_stride = 2 if compress_Z else 1
    Z_kernel = 4 if extra_upsample[0] > 0 else 3

    extra_upsample[0] = 0
    if(cylindrical): return CylindricalConvTranspose(dim, dim, kernel_size = (Z_kernel,4,4), stride = (Z_stride,2,2), padding = 1, output_padding = extra_upsample)
    else: return nn.ConvTranspose3d(dim, dim, kernel_size = (Z_kernel,4,4), stride = (Z_stride,2,2), padding = 1, output_padding = extra_upsample)

def Downsample(dim, cylindrical = False, compress_Z = False):
    Z_stride = 2 if compress_Z else 1
    if(cylindrical): return CylindricalConv(dim, dim, kernel_size = (3,4,4), stride = (Z_stride,2,2), padding = 1)
    else: return nn.Conv3d(dim, dim, kernel_size = (3,4,4), stride = (Z_stride,2,2), padding = 1)


class FCN(nn.Module):
    #Fully connected network
    def __init__(self,
            dim_in = 356,
            num_layers = 4, 
            cond_dim = 64,
            time_embed = True,
            cond_embed = True,
            ):

        super().__init__()



        # time and energy embeddings
        half_cond_dim = cond_dim // 2
        time_layers = []
        if(time_embed): time_layers = [SinusoidalPositionEmbeddings(half_cond_dim//2)]
        else: time_layers = [nn.Linear(1, half_cond_dim//2),nn.GELU()]
        time_layers += [ nn.Linear(half_cond_dim//2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]


        cond_layers = []
        if(cond_embed): cond_layers = [SinusoidalPositionEmbeddings(half_cond_dim//2)]
        else: cond_layers = [nn.Linear(1, half_cond_dim//2),nn.GELU()]
        cond_layers += [ nn.Linear(half_cond_dim//2, half_cond_dim), nn.GELU(), nn.Linear(half_cond_dim, half_cond_dim)]


        self.time_mlp = nn.Sequential(*time_layers)
        self.cond_mlp = nn.Sequential(*cond_layers)


        out_layers = [nn.Linear(dim_in + cond_dim, dim_in)]
        for i in range(num_layers-1):
            out_layers.append(nn.GELU())
            out_layers.append(nn.Linear(dim_in, dim_in))

        self.main_mlp = nn.Sequential(*out_layers)



    def forward(self, x, cond, time):

        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        x = torch.cat([x, t,c], axis = -1)

        x = self.main_mlp(x)
        return x



class CondUnet(nn.Module):
#Unet with conditional layers
    def __init__(
        self,
        out_dim=1,
        layer_sizes = None,
        channels=1,
        cond_dim = 64,
        resnet_block_groups=8,
        mid_attn = False,
        block_attn = False,
        compress_Z = False,
        cylindrical = False,
        data_shape = (-1,1,45, 16,9),
        time_embed = True,
        cond_embed = True,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.block_attn = block_attn
        self.mid_attn = mid_attn



        #dims = [channels, *map(lambda m: dim * m, dim_mults)]
        #layer_sizes.insert(0, channels)
        in_out = list(zip(layer_sizes[:-1], layer_sizes[1:])) 
        
        if(not cylindrical): self.init_conv = nn.Conv3d(channels, layer_sizes[0], kernel_size = 3, padding = 1)
        else: self.init_conv = CylindricalConv(channels, layer_sizes[0], kernel_size = 3, padding = 1)

        block_klass = partial(ResNetBlock, groups=resnet_block_groups, cylindrical = cylindrical)


        self.time_mlp = SinusoidalPositionEmbeddings(cond_dim) if time_embed else MlpEmbeddings(cond_dim)
        self.cond_mlp = SinusoidalPositionEmbeddings(cond_dim) if cond_embed else MlpEmbeddings(cond_dim)


        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.downs_attn = nn.ModuleList([])
        self.ups_attn = nn.ModuleList([])
        self.extra_upsamples = []
        self.Z_even = []
        num_resolutions = len(in_out)

        cur_data_shape = data_shape[-3:]

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= (num_resolutions - 1))
            if(not is_last):
                extra_upsample_dim = [(cur_data_shape[0] + 1)%2, cur_data_shape[1]%2, cur_data_shape[2]%2]
                Z_dim = cur_data_shape[0] if not compress_Z else math.ceil(cur_data_shape[0]/2.0)
                cur_data_shape = (Z_dim, cur_data_shape[1] // 2, cur_data_shape[2] //2)
                self.extra_upsamples.append(extra_upsample_dim)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, cond_dim=cond_dim),
                        block_klass(dim_out, dim_out, cond_dim=cond_dim),
                        Downsample(dim_out, cylindrical, compress_Z = compress_Z) if not is_last else nn.Identity(),
                    ]
                )
            )
            if(self.block_attn) : self.downs_attn.append(Residual(PreNorm(dim_out, LinearAttention(dim_out, cylindrical = cylindrical))))

        mid_dim = layer_sizes[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, cond_dim=cond_dim)
        if(self.mid_attn): self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, cylindrical = cylindrical)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, cond_dim=cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (ind >= (num_resolutions - 1))

            if(not is_last): 
                extra_upsample = self.extra_upsamples.pop()

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, cond_dim=cond_dim),
                        block_klass(dim_in, dim_in, cond_dim=cond_dim),
                        Upsample(dim_in, extra_upsample, cylindrical, compress_Z = compress_Z) if not is_last else nn.Identity(),
                    ]
                )
            )
            if(self.block_attn): self.ups_attn.append( Residual(PreNorm(dim_in, LinearAttention(dim_in, cylindrical = cylindrical))) )

        if(not cylindrical): final_lay = nn.Conv3d(layer_sizes[0], out_dim, 1)
        else:  final_lay = CylindricalConv(layer_sizes[0], out_dim, 1)
        self.final_conv = nn.Sequential( block_klass(layer_sizes[1], layer_sizes[0]),  final_lay )

    def forward(self, x, cond, time):

        x = self.init_conv(x)

        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        conditions = torch.cat([t,c], axis = -1)


        h = []

        # downsample
        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.block_attn): x = self.downs_attn[i](x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, conditions)
        if(self.mid_attn): x = self.mid_attn(x)
        x = self.mid_block2(x, conditions)


        # upsample
        for i, (block1, block2, upsample) in enumerate(self.ups):
            s = h.pop()
            x = torch.cat((x, s), dim=1)
            x = block1(x, conditions)
            x = block2(x, conditions)
            if(self.block_attn): x = self.ups_attn[i](x)
            x = upsample(x)

        return self.final_conv(x)