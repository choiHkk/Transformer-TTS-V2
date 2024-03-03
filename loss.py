from typing import Optional

import torch
import torch.nn.functional as F


def transformer_tts_loss(
    y_hat_post: torch.Tensor,
    y_hat: torch.Tensor,
    gate_hat: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    y_mask: torch.Tensor,
    c_reconstuction: float,
    c_gate: float,
):
    loss_post_reconstruction = (
        masked_reconstruction_loss(y_hat_post, y, y_mask) * c_reconstuction
    )
    loss_reconstruction = masked_reconstruction_loss(y_hat, y, y_mask) * c_reconstuction
    loss_gate = gate_loss(gate_hat, gate) * c_gate
    loss = loss_post_reconstruction + loss_reconstruction + loss_gate
    losses = (loss_post_reconstruction, loss_reconstruction, loss_gate)
    return loss, losses


def transformer_tts_v2_loss(
    x_hat: torch.Tensor,
    y_hat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: torch.Tensor,
    y_lengths: torch.Tensor,
    ref_ids: torch.Tensor,
    n_acoustic_vocabs: int,
    n_quantizers: int,
    c_linguistic: float,
    c_acoustic: float,
    ref_segment_size: int,
):
    # Set paddings to -1 to ignore them in loss
    for idx, l in enumerate(x_lengths):
        x[idx, l + 1 :] = -1

    for idx, l in enumerate(y_lengths):
        y[idx, :, l + 1 :] = -1
        if ref_ids is not None:
            ref_start = ref_ids[idx]
            ref_end = ref_start + ref_segment_size
            y[idx, :, ref_start:ref_end] = -1

    loss_x = cross_entropy_loss(x_hat, x) * c_linguistic
    y_hat = y_hat.view(y_hat.size(0) * n_quantizers, n_acoustic_vocabs, y_hat.size(2))
    y = y.view(y.size(0) * n_quantizers, y.size(2))
    loss_y = cross_entropy_loss(y_hat, y) * c_acoustic
    loss = loss_x + loss_y
    losses = (loss_x, loss_y)
    return loss, losses


def masked_reconstruction_loss(
    y_hat: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor
):
    y.requires_grad = False
    y_mask = y_mask.bool()
    y = y.masked_select(y_mask)
    y_hat = y_hat.masked_select(y_mask)
    return F.mse_loss(y_hat, y)


def cross_entropy_loss(
    y_hat: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0
):
    return F.cross_entropy(
        y_hat, labels.long(), ignore_index=-1, label_smoothing=label_smoothing
    ).mean()


def gate_loss(gate_hat: torch.Tensor, gate: torch.Tensor):
    gate.requires_grad = False
    gate_hat = gate_hat.view(-1, 1)
    gate = gate.view(-1, 1)
    return F.binary_cross_entropy(gate_hat, gate)
