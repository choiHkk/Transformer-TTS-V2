import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, out_channels: int, n_heads: int, p_dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = d_model // n_heads

        self.q_proj = nn.Conv1d(d_model, d_model, 1)
        self.k_proj = nn.Conv1d(d_model, d_model, 1)
        self.v_proj = nn.Conv1d(d_model, d_model, 1)
        self.o_proj = nn.Conv1d(d_model, out_channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def init(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

    def get_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        b, d, t_kv, t_q = (*k.size(), q.size(2))
        q = q.view(b, self.n_heads, self.k_channels, t_q).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_kv).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_kv).transpose(2, 3)
        attn = torch.matmul(q / math.sqrt(self.k_channels), k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)
        attn = F.softmax(attn, dim=-1)  # [B, n_heads, t_q, t_kv]
        attn = self.dropout(attn)
        o = torch.matmul(attn, v)
        o = o.transpose(2, 3).contiguous().view(b, d, t_q)
        return o, attn

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ):
        q = self.q_proj(x)
        k = self.k_proj(c)
        v = self.v_proj(c)
        x, self.attn = self.get_attention(q, k, v, attn_mask)
        x = self.o_proj(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model: int, p_dropout: float = 0.0):
        super().__init__()
        self.preproj = nn.Sequential(
            nn.Conv1d(d_model, d_model * 4, 1), nn.GELU(), nn.Dropout(p_dropout)
        )
        self.postproj = nn.Conv1d(d_model * 4, d_model, 1)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        x = self.preproj(x * x_mask if x_mask is not None else x)
        x = self.postproj(x * x_mask if x_mask is not None else x)
        x = x * x_mask if x_mask is not None else x
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, p_dropout: float = 0.0
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.attn_norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norm_layes = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(d_model, d_model, n_heads, p_dropout)
            )
            self.attn_norm_layers.append(LayerNorm(d_model))
            self.ffn_layers.append(FFN(d_model, p_dropout))
            self.ffn_norm_layes.append(LayerNorm(d_model))

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None):
        if x_mask is not None:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(3)
            x = x * x_mask
        else:
            attn_mask = None
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.attn_norm_layers[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)
            x = self.ffn_norm_layes[i](x + y)
        if x_mask is not None:
            x = x * x_mask
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float = 0.0,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.use_cross_attention = use_cross_attention
        self.dropout = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.self_attn_norm_layers = nn.ModuleList()
        if use_cross_attention:
            self.cross_attn_layers = nn.ModuleList()
            self.cross_attn_norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norm_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(d_model, d_model, n_heads, p_dropout)
            )
            self.self_attn_norm_layers.append(LayerNorm(d_model))
            if use_cross_attention:
                self.cross_attn_layers.append(
                    MultiHeadAttention(d_model, d_model, n_heads, p_dropout)
                )
                self.cross_attn_norm_layers.append(LayerNorm(d_model))
            self.ffn_layers.append(FFN(d_model, p_dropout))
            self.ffn_norm_layers.append(LayerNorm(d_model))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        c_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        if self_attn_mask is None:
            self_attn_mask = utils.subsequent_mask(x_mask.size(2)).to(x)
            self_attn_mask = self_attn_mask.bool() + x_mask.unsqueeze(2).bool()
            self_attn_mask = self_attn_mask.to(x)
        if self.use_cross_attention:
            cross_attn_mask = c_mask.unsqueeze(2) * x_mask.unsqueeze(3)
        if x_mask is not None:
            x = x * x_mask
        if self.use_cross_attention:
            if c_mask is not None:
                c = c * c_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.dropout(y)
            x = self.self_attn_norm_layers[i](x + y)

            if self.use_cross_attention:
                y = self.cross_attn_layers[i](x, c, cross_attn_mask)
                y = self.dropout(y)
                x = self.cross_attn_norm_layers[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)
            x = self.ffn_norm_layers[i](x + y)
        if x_mask is not None:
            x = x * x_mask
        return x
