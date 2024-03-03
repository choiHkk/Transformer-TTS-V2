import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import attentions
import utils
from configs.base import TransformerTTSConfig
from configs.cross_entropy import TransformerTTSv2Config
from gst import GlobalStyleTokenEncoder, StyleTokenEncoder


class TokenEmbedding(nn.Module):
    def __init__(self, n_vocabs: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocabs,
            embedding_dim=d_model,
        )
        std = math.sqrt(2.0 / (n_vocabs + d_model))
        val = math.sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, sequence_max_length: int = 1024):
        super().__init__()
        position = torch.arange(sequence_max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(sequence_max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2).transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = x.transpose(0, 1).transpose(1, 2)
        return x


class EncoderPrenet(nn.Module):
    def __init__(
        self,
        n_vocabs: int,
        d_model: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
        n_quantizers: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers

        emb_channels = d_model
        if n_quantizers is not None:
            assert d_model % n_quantizers == 0
            emb_channels = emb_channels // n_quantizers

        self.token_embedding = TokenEmbedding(n_vocabs, emb_channels)

        self.preproj = nn.Conv1d(d_model, d_model, 1)

        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size,
                    padding=utils.get_padding(kernel_size),
                ),
                attentions.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(p_dropout),
            )
            for _ in range(n_layers)
        )

        self.postproj = nn.Conv1d(d_model, d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ):
        if emb is None:
            x = self.token_embedding(x)
        else:
            x = emb
        x_mask = utils.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.preproj(x * x_mask)
        for i in range(self.n_layers):
            x = self.layers[i](x * x_mask)
        x = self.postproj(x * x_mask)
        x = x * x_mask
        return x, x_mask


class DecoderPrenet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.5,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.preproj = nn.Sequential(
            nn.Conv1d(in_channels, d_model, 1), attentions.LayerNorm(d_model), nn.GELU()
        )
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size,
                    padding=utils.get_padding(kernel_size),
                ),
                attentions.LayerNorm(d_model),
                nn.GELU(),
            )
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        x_mask = utils.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = F.dropout(self.preproj(x * x_mask), p=self.p_dropout, training=True)
        for i in range(self.n_layers):
            x = F.dropout(self.layers[i](x * x_mask), p=self.p_dropout, training=True)
        x = x * x_mask
        return x, x_mask


class Encoder(nn.Module):
    def __init__(
        self,
        model_architecture_type: str,
        n_vocabs: int,
        d_model: int,
        kernel_size: int = 5,
        n_prenet_layers: int = 3,
        n_layers: int = 6,
        n_heads: int = 8,
        sequence_max_length: int = 1024,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.prenet = EncoderPrenet(
            n_vocabs, d_model, kernel_size, n_prenet_layers, p_dropout
        )
        self.positional_embedding = PositionalEmbedding(d_model, sequence_max_length)
        self.transformer_encoder = attentions.TransformerEncoder(
            d_model, n_layers, n_heads, p_dropout
        )

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, c: Optional[torch.tensor] = None
    ):
        x, x_mask = self.prenet(x, x_lengths)
        x = self.positional_embedding(x)
        x = self.transformer_encoder(x, x_mask)
        return x, x_mask


class Decoder(nn.Module):
    def __init__(
        self,
        model_architecture_type: str,
        in_channels: int,
        d_model: int,
        kernel_size: int = 5,
        n_prenet_layers: int = 1,
        n_layers: int = 6,
        n_heads: int = 8,
        sequence_max_length: int = 1024,
        p_dropout: float = 0.0,
        n_acoustic_vocabs: Optional[int] = None,
        n_quantizers: Optional[int] = None,
    ):
        super().__init__()
        self.model_architecture_type = model_architecture_type
        self.n_quantizers = n_quantizers

        self.positional_embedding = PositionalEmbedding(d_model, sequence_max_length)

        if model_architecture_type == "reconstruction":
            self.prenet = DecoderPrenet(
                in_channels, d_model, kernel_size, n_prenet_layers, p_dropout=0.5
            )
            self.transformer_decoder = attentions.TransformerDecoder(
                d_model, n_layers, n_heads, p_dropout, True
            )
        elif model_architecture_type == "cross_entropy":
            self.prenet = EncoderPrenet(
                n_acoustic_vocabs,
                d_model,
                kernel_size,
                n_prenet_layers,
                p_dropout=0.5,
                n_quantizers=n_quantizers,
            )
            self.transformer_decoder = attentions.TransformerDecoder(
                d_model, n_layers, n_heads, p_dropout, False
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        c: torch.Tensor,
        c_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        T_g: Optional[int] = None,
        T_l: Optional[int] = None,
        T_a: Optional[int] = None,
        # return_latents: bool = False,
        norm: Optional[nn.Module] = None,
    ):
        if self.model_architecture_type == "reconstruction":
            x, x_mask = self.prenet(x, x_lengths)
        elif self.model_architecture_type == "cross_entropy":
            T = x.size(2)
            x = self.prenet.token_embedding(x)
            x = x.contiguous().view(x.size(0), -1, T)
            x, x_mask = self.prenet(None, x_lengths, emb=x)
        else:
            raise NotImplementedError

        c = self.positional_embedding(c)
        x = self.positional_embedding(x)

        if self.model_architecture_type == "reconstruction":
            x = self.transformer_decoder(x, x_mask, c, c_mask)
            return x, x_mask

        elif self.model_architecture_type == "cross_entropy":
            x = torch.cat([g, c, x], dim=2)
            g_lengths = torch.LongTensor([g.size(2)] * g.size(0)).to(g.device)
            g_mask = utils.sequence_mask(g_lengths, g.size(2)).unsqueeze(1)
            padding_mask = torch.cat([g_mask, c_mask, x_mask], dim=2)
            subsequent_mask = utils.subsequent_mask(padding_mask.size(2))
            self_attn_mask = padding_mask.unsqueeze(2).bool() + subsequent_mask.bool()
            self_attn_mask = self_attn_mask.to(x)

            x = self.transformer_decoder(x, None, None, None, self_attn_mask)
            x = x[:, :, T_g:]
            x = norm(x)
            c, x = x[:, :, :T_l], x[:, :, -T_a:]
            # if not return_latents:
            #     c = self.proj_l(c * c_mask) * c_mask
            #     x = self.proj_a(x * x_mask) * x_mask
            return g, c, c_mask, x, x_mask

        else:
            raise NotImplementedError


class Postnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    d_model * 2,
                    kernel_size,
                    padding=utils.get_padding(kernel_size),
                ),
                attentions.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(p_dropout),
            )
        )
        for _ in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        d_model * 2,
                        d_model * 2,
                        kernel_size,
                        padding=utils.get_padding(kernel_size),
                    ),
                    attentions.LayerNorm(d_model * 2),
                    nn.GELU(),
                    nn.Dropout(p_dropout),
                )
            )
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(
                    d_model * 2,
                    out_channels,
                    kernel_size,
                    padding=utils.get_padding(kernel_size),
                ),
                attentions.LayerNorm(out_channels),
                nn.GELU(),
                nn.Dropout(p_dropout),
            )
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        for i in range(self.n_layers):
            x = self.layers[i](x * x_mask)
        x = self.layers[-1](x * x_mask)
        x = x * x_mask
        return x


class TransformerTTS(nn.Module):
    def __init__(self, config: TransformerTTSConfig):
        super().__init__()
        self.config = config
        assert config.model_architecture_type == "reconstruction"

        self.encoder = Encoder(
            model_architecture_type=config.model_architecture_type,
            n_vocabs=config.n_vocabs,
            d_model=config.d_model,
            kernel_size=config.encoder_kernel_size,
            n_prenet_layers=config.n_encoder_prenet_layers,
            n_layers=config.n_encoder_layers,
            n_heads=config.n_encoder_heads,
            sequence_max_length=config.sequence_max_length,
            p_dropout=config.p_dropout,
        )

        self.decoder = Decoder(
            model_architecture_type=config.model_architecture_type,
            in_channels=config.decoder_in_channels,
            d_model=config.d_model,
            kernel_size=config.decoder_kernel_size,
            n_prenet_layers=config.n_decoder_prenet_layers,
            n_layers=config.n_decoder_layers,
            n_heads=config.n_decoder_heads,
            sequence_max_length=config.sequence_max_length,
            p_dropout=config.p_dropout,
        )

        self.mel_proj = nn.Conv1d(config.d_model, config.decoder_in_channels, 1)
        self.gate_proj = nn.Conv1d(config.d_model, 1, 1)
        self.postnet = Postnet(
            in_channels=config.decoder_in_channels,
            d_model=config.d_model,
            out_channels=config.decoder_in_channels,
            kernel_size=config.postnet_kernel_size,
            n_layers=config.n_postnet_layers,
            p_dropout=config.p_dropout,
        )

        self.gst = GlobalStyleTokenEncoder(
            n_tokens=config.gst_n_tokens,
            in_channels=config.gst_in_channels,
            hidden_channels=config.gst_hidden_channels,
            filter_channel_list=config.gst_filter_channel_list,
            n_heads=config.gst_n_heads,
            kernel_size=config.gst_kernel_size,
            stride=config.gst_stride,
        )
        self.cond_g = nn.Conv1d(
            config.d_model + config.gst_hidden_channels, config.d_model, 1
        )

    def get_start_frame(self, x: torch.Tensor):
        start_y_frame = x.data.new(x.size(0), self.config.decoder_in_channels).zero_()
        start_gate_frame = x.data.new(x.size(0), 1).fill_(-1e4)
        return start_y_frame, start_gate_frame

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor,
    ):
        g = self.gst(g)
        x, x_mask = self.encoder(x, x_lengths)
        x = self.cond_g(torch.cat([x, g.repeat(1, 1, x.size(2))], dim=1))
        y, y_mask = self.decoder(y, y_lengths, x, x_mask)
        gate = F.sigmoid(self.gate_proj(y))
        y = self.mel_proj(y) * y_mask
        y_post = self.postnet(y, y_mask) + y
        return y_post, y, gate, x_mask, y_mask

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor,
        gate_threshold: float = 0.5,
        sequence_max_length: int = 1000,
    ):
        g = self.gst(g)
        x, x_mask = self.encoder(x, x_lengths)
        x = self.cond_g(torch.cat([x, g.repeat(1, 1, x.size(2))], dim=1))
        y_hat = torch.FloatTensor([]).to(x)
        gate_hat = torch.FloatTensor([]).to(x)
        start_y_frame, start_gate_frame = self.get_start_frame(x)
        y_hat = torch.cat([y_hat, start_y_frame], dim=1).unsqueeze(2)
        gate_hat = torch.cat([gate_hat, start_gate_frame], dim=1).unsqueeze(2)
        while True:
            y_lengths = torch.LongTensor([y_hat.size(2)]).to(x.device)
            y_frame, y_mask = self.decoder(y_hat, y_lengths, x, x_mask)
            gate_frame = self.gate_proj(y_frame) * y_mask
            y_frame = self.mel_proj(y_frame) * y_mask

            if F.sigmoid(gate_frame[..., -1].data).item() > gate_threshold:
                break
            elif y_hat.size(2) == sequence_max_length:
                print("Warning! Reached max decoder steps")
                break
            else:
                y_hat = torch.cat([y_hat, y_frame[:, :, -1:]], dim=2)
                gate_hat = torch.cat([gate_hat, gate_frame[:, :, -1:]], dim=2)

        gate_hat = F.sigmoid(gate_hat)
        y_hat = y_hat * y_mask
        y_post_hat = self.postnet(y_hat, y_mask) + y_hat
        return y_post_hat, y_hat, gate_hat, x_mask, y_mask


class TransformerTTSv2(nn.Module):
    def __init__(self, config: TransformerTTSv2Config):
        super().__init__()
        self.config = config
        assert config.model_architecture_type == "cross_entropy"

        self.encoder = Encoder(
            model_architecture_type=config.model_architecture_type,
            n_vocabs=config.n_linguistic_vocabs,
            d_model=config.d_model,
            kernel_size=config.encoder_kernel_size,
            n_prenet_layers=config.n_encoder_prenet_layers,
            n_layers=config.n_encoder_layers,
            n_heads=config.n_encoder_heads,
            sequence_max_length=config.sequence_max_length,
            p_dropout=config.p_dropout,
        )

        self.decoder = Decoder(
            model_architecture_type=config.model_architecture_type,
            in_channels=config.decoder_in_channels,
            d_model=config.d_model,
            kernel_size=config.decoder_kernel_size,
            n_prenet_layers=config.n_decoder_prenet_layers,
            n_layers=config.n_decoder_layers,
            n_heads=config.n_decoder_heads,
            sequence_max_length=config.sequence_max_length,
            p_dropout=config.p_dropout,
            n_acoustic_vocabs=config.n_acoustic_vocabs,
            n_quantizers=config.n_quantizers,
        )

        self.st = StyleTokenEncoder(
            n_tokens=config.gst_n_tokens,
            in_channels=config.n_mel_channels,
            d_model=config.d_model,
            n_layers=config.gst_n_layers,
            n_heads=config.gst_n_heads,
            p_dropout=config.gst_p_dropout,
        )

        self.final_norm = attentions.LayerNorm(config.d_model)
        self.proj_l = nn.Conv1d(config.d_model, config.n_linguistic_vocabs, 1)
        self.proj_a = nn.Conv1d(
            config.d_model, config.n_acoustic_vocabs * config.n_quantizers, 1
        )

    def get_start_frame(self, x: torch.Tensor):
        start_y_frame = x.data.new(x.size(0), self.config.decoder_in_channels).zero_()
        return start_y_frame

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor,
        return_logits: bool = True,
        is_encoded: bool = False,
        x_mask: Optional[torch.Tensor] = None,
    ):
        g = self.st(g)
        T_g, T_l, T_a = g.size(2), x.size(1), y.size(2)
        if not is_encoded:
            x, x_mask = self.encoder(x, x_lengths)
        g, x, x_mask, y, y_mask = self.decoder(
            x=y,
            x_lengths=y_lengths,
            c=x,
            c_mask=x_mask,
            g=g,
            T_g=T_g,
            T_l=T_l,
            T_a=T_a,
            norm=self.final_norm,
        )
        if return_logits:
            x = self.proj_l(x * x_mask)
            y = self.proj_a(y * y_mask)
        return x, x_mask, y, y_mask
