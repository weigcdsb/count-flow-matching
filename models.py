import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#### helper class
def _xy_coords_like(x):
    # x: [B,C,H,W] -> coords: [B,2,H,W] with channels [X,Y] in [-1,1]
    B, _, H, W = x.shape
    device, dtype = x.device, x.dtype
    ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij') 
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, H, W)
    return coords


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

    

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
        
class GaussianFourierProjection(torch.nn.Module):
  """
  Gaussian random features for encoding time steps.
  This is similar to implementation from Karras.
  """  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(torch.nn.Module):
  """
  A fully connected layer that reshapes outputs to feature maps.
  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = torch.nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

def _prefer_gn_groups(C, max_groups):
    for g in [32, 16, 8, 4, 2, 1]:
        if g <= max_groups and C % g == 0:
            return g
    return 1



####--------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# class MLP_rate(nn.Module):
#     def __init__(self, base_mlp):
#         super().__init__(); self.base = base_mlp
#     def forward(self, x):
#         return F.softplus(self.base(x))

class MLP_rate(nn.Module):
    def __init__(self, base_mlp, log_cap=12.0, eps=1e-8):
        super().__init__(); self.base = base_mlp; self.log_cap = log_cap; self.eps = eps
    def forward(self, x):
        log_rate = torch.clamp(self.base(x), max=self.log_cap)  # cap huge logits
        return torch.exp(log_rate) + self.eps   


#### -------------------------------------------------------------------
#### transformer used in DFM for scRNA

def _timestep_embedding(t, dim, max_period=10000):
    if t.ndim == 2:
        t = t[:, 0]
    half = dim // 2
    device = t.device
    dtype = t.dtype
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=device, dtype=dtype) / max(half, 1)
    )
    args = t[:, None] * freqs[None, :] * 2.0 * math.pi
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _modulate(x, shift, scale):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        d_ff = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

    def forward(self, x, cond):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), s1, sc1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None, :] * attn_out

        h = _modulate(self.norm2(x), s2, sc2)
        mlp_out = self.mlp(h)
        x = x + g2[:, None, :] * mlp_out
        return x


class ChunkedAdaLNTransformer(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        time_varying=True,
        d_model=256,
        depth=8,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        chunk_size=128,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.time_varying = time_varying
        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.max_period = max_period

        if self.out_dim % self.dim != 0:
            raise ValueError(
                f"ChunkedAdaLNTransformer currently expects out_dim to be a multiple of dim. "
                f"Got dim={self.dim}, out_dim={self.out_dim}."
            )
        self.out_mult = self.out_dim // self.dim

        self.n_tokens = (self.dim + self.chunk_size - 1) // self.chunk_size
        self.padded_dim = self.n_tokens * self.chunk_size

        self.token_in = nn.Linear(self.chunk_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))

        if self.time_varying:
            self.time_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.time_mlp = None

        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )

        self.token_out = nn.Linear(d_model, self.chunk_size * self.out_mult)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _get_cond(self, t):
        if not self.time_varying:
            return torch.zeros((t.shape[0], self.d_model), device=t.device, dtype=t.dtype)
        temb = _timestep_embedding(t, self.d_model, max_period=self.max_period)
        return self.time_mlp(temb)

    def forward(self, z):
        if self.time_varying:
            x = z[:, :self.dim]
            t = z[:, self.dim:]
        else:
            x = z
            t = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

        B = x.shape[0]

        if self.padded_dim > self.dim:
            x = F.pad(x, (0, self.padded_dim - self.dim))
        x = x.view(B, self.n_tokens, self.chunk_size)

        h = self.token_in(x) + self.pos_embed
        cond = self._get_cond(t)
        
        for blk in self.blocks:
            h = blk(h, cond)

        s, sc = self.final_adaLN(cond).chunk(2, dim=-1)
        h = _modulate(self.final_norm(h), s, sc)
        y = self.token_out(h)
        
        y = y.view(B, self.n_tokens, self.out_mult, self.chunk_size)
        y = y.permute(0, 2, 1, 3).contiguous()

        y = y.view(B, self.out_mult, self.padded_dim)
        y = y[:, :, :self.dim]
        y = y.reshape(B, self.out_dim)
        return y


class ChunkedAdaLNTransformer_rate(nn.Module):
    def __init__(self, base_model, log_cap=12.0, eps=1e-8):
        super().__init__()
        self.base = base_model
        self.log_cap = log_cap
        self.eps = eps

    def forward(self, x):
        log_rate = torch.clamp(self.base(x), max=self.log_cap)
        return torch.exp(log_rate) + self.eps


####--------------------------------------------------------------------
#### Adapted from previous double flow UNet
class Adapted_FlatConvUNet(nn.Module):
    """
    Drop-in styled like your MLP:

        net_base = Adapted_FlatConvUNet(dim=d, out_dim=2*d, w=64, time_varying=True)
        y = net_base(z)

    If time_varying=True:
        - expect input z: [B, d+1] = concat([x, t])
        - use x for content, t for FiLM conditioning.
    If time_varying=False:
        - expect input x: [B, d], no time, FiLM is disabled (identity).
    """

    def __init__(
        self,
        dim,
        out_dim=None,
        time_varying=False,
        channels=(32, 64, 128),
        embed_dim=256,
        img_size=8,
        img_ch=1,
        cov_dim=0,
        use_coord_input=True,
        reflect_pad_first=True,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.dim = dim                    # feature dimension (no t)
        self.out_dim = out_dim
        self.time_varying = time_varying
        self.img_size = img_size
        self.img_ch = img_ch
        self.latent_size = img_ch * (img_size ** 2)
        self.cov_dim = cov_dim
        self.use_coord_input = use_coord_input
        self.reflect_pad_first = reflect_pad_first

        # time embedding (only used if time_varying=True)
        if self.time_varying:
            self.embed = nn.Sequential(
                GaussianFourierProjection(embed_dim=embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.embed = None
        self.cov_embed = nn.Linear(cov_dim, embed_dim) if (cov_dim > 0 and self.time_varying) else None

        # flat <-> pseudo-image
        self.up_fc = Linear(dim, self.latent_size, bias=True)
        self.down_fc = Linear(self.latent_size, out_dim, bias=True)

        # encoder
        in_ch1 = img_ch + (2 if self.use_coord_input else 0)
        self.conv1 = nn.Conv2d(
            in_ch1,
            channels[0],
            3,
            stride=1,
            bias=False,
        )
        self.dense1 = Dense(embed_dim, 2 * channels[0]) if self.time_varying else None
        self.gnorm1 = nn.GroupNorm(_prefer_gn_groups(channels[0], 4), channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
        self.dense2 = Dense(embed_dim, 2 * channels[1]) if self.time_varying else None
        self.gnorm2 = nn.GroupNorm(_prefer_gn_groups(channels[1], 32), channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=False)
        self.dense3 = Dense(embed_dim, 2 * channels[2]) if self.time_varying else None
        self.gnorm3 = nn.GroupNorm(_prefer_gn_groups(channels[2], 32), channels[2])

        # decoder
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=False)
        self.dense4 = Dense(embed_dim, 2 * channels[1]) if self.time_varying else None
        self.tgnorm3 = nn.GroupNorm(_prefer_gn_groups(channels[1], 32), channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=1, bias=False
        )
        self.dense5 = Dense(embed_dim, 2 * channels[0]) if self.time_varying else None
        self.tgnorm2 = nn.GroupNorm(_prefer_gn_groups(channels[0], 32), channels[0])

        self.tconv1 = nn.ConvTranspose2d(
            channels[0] + channels[0], img_ch, 3, stride=1
        )

    @staticmethod
    def _film(h, gb):
        # h: [B,C,H,W], gb: [B,2C,1,1]
        g, b = gb.chunk(2, dim=1)
        return h * (1.0 + g) + b

    def _get_cond(self, t, c=None):
        if (not self.time_varying) or (self.embed is None):
            return None
        t = t.view(-1)
        cond = F.silu(self.embed(t))     # now cond: [B, embed_dim]
        if (self.cov_embed is not None) and (c is not None):
            cond = cond + self.cov_embed(c)
        return cond

    def forward(self, z, c=None):
        """
        If time_varying:
            z: [B, dim+1] = concat([x, t])
        Else:
            z: [B, dim]   = x

        Returns: [B, out_dim]
        """
        if self.time_varying:
            x = z[:, : self.dim]
            t = z[:, self.dim:]
            cond = self._get_cond(t, c)
        else:
            x = z
            cond = None

        B = x.size(0)

        # flat -> pseudo image
        h = self.up_fc(x)                           # [B, latent_size]
        h = h.view(B, self.img_ch, self.img_size, self.img_size)

        if self.use_coord_input:
            h = torch.cat([h, _xy_coords_like(h)], dim=1)

        # encoder
        h1 = self.conv1(h)
        h1 = self.gnorm1(h1)
        if cond is not None:
            h1 = self._film(h1, self.dense1(cond))
        h1 = F.silu(h1)

        h2 = self.conv2(h1)
        h2 = self.gnorm2(h2)
        if cond is not None:
            h2 = self._film(h2, self.dense2(cond))
        h2 = F.silu(h2)

        h3 = self.conv3(h2)
        h3 = self.gnorm3(h3)
        if cond is not None:
            h3 = self._film(h3, self.dense3(cond))
        h3 = F.silu(h3)

        # decoder
        h4 = self.tconv3(h3)
        h4 = self.tgnorm3(h4)
        if cond is not None:
            h4 = self._film(h4, self.dense4(cond))
        h4 = F.silu(h4)

        h5 = self.tconv2(torch.cat([h4, h2], dim=1))
        h5 = self.tgnorm2(h5)
        if cond is not None:
            h5 = self._film(h5, self.dense5(cond))
        h5 = F.silu(h5)

        h6 = self.tconv1(torch.cat([h5, h1], dim=1))

        # back to flat
        h6 = h6.view(B, self.latent_size)
        out = self.down_fc(h6)                      # [B, out_dim]
        return out


# class Adapted_FlatConvUNet_rate(nn.Module):
#     def __init__(self, base_unet: Adapted_FlatConvUNet):
#         super().__init__()
#         self.base = base_unet

#     def forward(self, z, c=None):
#         return F.softplus(self.base(z, c))


class Adapted_FlatConvUNet_rate(nn.Module):
    def __init__(self, base_unet, log_cap=12.0, eps=1e-8):
        super().__init__(); self.base = base_unet; self.log_cap = log_cap; self.eps = eps
    def forward(self, x):
        log_rate = torch.clamp(self.base(x), max=self.log_cap)  # cap huge logits
        return torch.exp(log_rate) + self.eps   

        
