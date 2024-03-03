from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import attentions
import utils


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        filter_channel_list: List[int] = [32, 32, 64, 64, 128, 128],
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.in_channels = in_channels

        K = len(filter_channel_list)
        filters = [1] + filter_channel_list
        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=(
                    utils.get_padding(kernel_size[0]),
                    utils.get_padding(kernel_size[1]),
                ),
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=filter_channel_list[i]) for i in range(K)]
        )

        out_channels = self.calculate_channels(in_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=filter_channel_list[-1] * out_channels,
            hidden_size=hidden_channels,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, return_memory: bool = False):
        N = x.size(0)
        out = x.contiguous().view(N, 1, -1, self.in_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]
        if return_memory:
            return memory

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class StyleTokenLayer(nn.Module):
    def __init__(self, n_tokens: int, hidden_channels: int, n_heads: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.FloatTensor(hidden_channels, n_tokens))
        self.attention = attentions.MultiHeadAttention(
            hidden_channels, hidden_channels, n_heads
        )
        nn.init.normal_(self.embeddings, mean=0, std=0.5)

    def forward(self, x: torch.Tensor, use_global_embedding: bool = True):
        if use_global_embedding:
            x = x.unsqueeze(2)
        embeddings = F.tanh(self.embeddings)
        embeddings = embeddings.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = self.attention(x, embeddings)
        return x


class GlobalStyleTokenEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int = 16,
        in_channels: int = 128,
        hidden_channels: int = 128,
        filter_channel_list: List[int] = [32, 32, 64, 64, 128, 128],
        n_heads: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.reference_encoder = ReferenceEncoder(
            in_channels, hidden_channels, filter_channel_list, kernel_size, stride
        )
        self.style_encoder = StyleTokenLayer(n_tokens, hidden_channels, n_heads)

    def forward(self, x: torch.Tensor):
        x = self.reference_encoder(x)
        x = self.style_encoder(x)
        return x


class StyleTokenEncoder(nn.Module):
    def __init__(
        self,
        n_tokens: int = 32,
        in_channels: int = 80,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.preproj = nn.Conv1d(in_channels, d_model, 1)
        self.transformer_encoder = attentions.TransformerEncoder(
            d_model, n_layers, n_heads, p_dropout
        )
        self.style_encoder = StyleTokenLayer(n_tokens, d_model, n_heads)

    def forward(self, x: torch.Tensor):
        x = self.preproj(x)
        x = self.transformer_encoder(x)
        x = self.style_encoder(x, use_global_embedding=False)
        return x


if __name__ == "__main__":
    n_tokens = 16
    n_mel_channels = 80
    in_channels = 128
    hidden_channels = 128
    d_model = 512
    filter_channel_list = [32, 32, 64, 64, 128, 128]
    n_layers = 4
    n_heads = 8
    kernel_size = (3, 3)
    stride = (2, 2)
    p_dropout = 0.1

    global_style_token_encoder = GlobalStyleTokenEncoder(
        n_tokens=n_tokens,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        filter_channel_list=filter_channel_list,
        n_heads=n_heads,
        kernel_size=kernel_size,
        stride=stride,
    )
    style_encoder = StyleTokenEncoder(
        in_channels=n_mel_channels,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        p_dropout=p_dropout,
    )
    print("GST Parameters[M]: ", utils.count_parameters(global_style_token_encoder))
    print("SE Parameters[M]: ", utils.count_parameters(style_encoder))

    x = torch.randn(32, 128, 177)
    o = global_style_token_encoder(x)
    print(o.size())

    x = torch.randn(32, 80, 177)
    o = style_encoder(x)
    print(o.size())
