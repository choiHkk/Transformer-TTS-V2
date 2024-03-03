import torch

import utils
from configs.base import (TransformerTTSConfig, TransformerTTSInferenceConfig,
                          TransformerTTSTrainingConfig)
from configs.cross_entropy import (TransformerTTSv2Config,
                                   TransformerTTSv2InferenceConfig,
                                   TransformerTTSv2TrainingConfig)
from encodec_wrapper import EncodecWrapper
from loss import transformer_tts_loss, transformer_tts_v2_loss
from mel_processing import mel_spectrogram_torch
from models import TransformerTTS, TransformerTTSv2


def test_transformer_tts():
    model_config = TransformerTTSConfig()
    training_config = TransformerTTSTrainingConfig()
    inference_config = TransformerTTSInferenceConfig()

    B = training_config.batch_size
    T_enc = 77
    T_dec = 177

    encodec = EncodecWrapper()
    transformer_tts = TransformerTTS(config=model_config)
    print("Encodec Parameters[M]: ", utils.count_parameters(encodec))
    print("Model Parameters[M]: ", utils.count_parameters(transformer_tts))

    x = torch.randint(0, model_config.n_vocabs, (B, T_enc))
    x_lengths = torch.randint(T_enc - 16, T_enc, (B,))
    x_lengths[0] = T_enc

    y = torch.randn(B, model_config.decoder_in_channels, T_dec)
    y_lengths = torch.randint(T_dec - 16, T_dec, (B,))
    y_lengths[0] = T_dec

    y_ref = 2 * torch.randn(B, model_config.sampling_rate) - 1
    with torch.no_grad():
        y_ref = encodec.encode(y_ref, return_emb=True)

    gate = torch.zeros(B, T_dec)
    for i in range(B):
        gate[i, y_lengths[i] - 1 :] = 1

    y_hat_post, y_hat, gate_hat, x_mask, y_mask = transformer_tts(
        x, x_lengths, y, y_lengths, y_ref
    )
    print(
        y_hat_post.size(),
        y_hat.size(),
        gate_hat.size(),
        x_mask.size(),
        y_mask.size(),
    )

    for i in range(model_config.n_encoder_layers):
        print(transformer_tts.encoder.transformer_encoder.attn_layers[i].attn.size())

    for i in range(model_config.n_decoder_layers):
        print(
            transformer_tts.decoder.transformer_decoder.self_attn_layers[i].attn.size()
        )
        print(
            transformer_tts.decoder.transformer_decoder.cross_attn_layers[i].attn.size()
        )

    loss, (loss_post_reconstruction, loss_reconstruction, loss_gate) = (
        transformer_tts_loss(
            y_hat_post=y_hat_post,
            y_hat=y_hat,
            gate_hat=gate_hat,
            y=y,
            gate=gate,
            y_mask=y_mask,
            c_reconstuction=training_config.c_reconstuction,
            c_gate=training_config.c_gate,
        )
    )
    print(loss, loss_post_reconstruction, loss_reconstruction, loss_gate)

    y_hat_post, y_hat, gate_hat, x_mask, y_mask = transformer_tts.inference(
        x[:1],
        x_lengths[:1],
        y_ref[:1],
        gate_threshold=inference_config.gate_threshold,
        max_length=inference_config.max_length,
    )
    print(
        y_hat_post.size(),
        y_hat.size(),
        gate_hat.size(),
        x_mask.size(),
        y_mask.size(),
    )


def test_transformer_tts_v2():
    model_config = TransformerTTSv2Config()
    training_config = TransformerTTSv2TrainingConfig()

    B = training_config.batch_size
    T_enc = 77
    T_dec = model_config.sampling_rate

    encodec = EncodecWrapper()
    transformer_tts = TransformerTTSv2(config=model_config)
    print("Encodec Parameters[M]: ", utils.count_parameters(encodec))
    print("Model Parameters[M]: ", utils.count_parameters(transformer_tts))

    x = torch.randint(0, model_config.n_linguistic_vocabs, (B, T_enc))
    x_lengths = torch.randint(T_enc - 16, T_enc, (B,))
    x_lengths[0] = T_enc

    y = 2 * torch.rand(B, T_dec) - 1
    y_lengths = torch.randint(T_dec // 2, T_dec, (B,))
    y_lengths[0] = y.size(-1)

    y_ref, ref_ids = utils.rand_slice_segments(
        y.unsqueeze(1), y_lengths, segment_size=training_config.ref_segment_size
    )
    y_ref = mel_spectrogram_torch(
        y=y_ref.squeeze(1),
        n_fft=model_config.n_fft,
        n_mel_channels=model_config.n_mel_channels,
        sampling_rate=model_config.sampling_rate,
        hop_length=model_config.hop_length,
        win_length=model_config.win_length,
        fmin=model_config.fmin,
        fmax=model_config.fmax,
    )
    ref_ids = ref_ids // model_config.hop_length
    print(y_ref.size(), ref_ids)

    with torch.no_grad():
        y = encodec.encode(y, return_emb=False)
        y = y[:, : model_config.n_quantizers, :]
    y_lengths = y_lengths // model_config.hop_length + 1
    y_lengths = torch.clamp_max(y_lengths, max=y.size(2))

    x_hat, x_mask, y_hat, y_mask = transformer_tts(x, x_lengths, y, y_lengths, y_ref)
    print(x_hat.size(), x_mask.size(), y_hat.size(), y_mask.size())

    loss, (loss_x, loss_y) = transformer_tts_v2_loss(
        x_hat=x_hat,
        y_hat=y_hat,
        x=x,
        y=y,
        x_lengths=x_lengths,
        y_lengths=y_lengths,
        ref_ids=ref_ids,
        n_acoustic_vocabs=model_config.n_acoustic_vocabs,
        n_quantizers=model_config.n_quantizers,
        c_linguistic=training_config.c_linguistic,
        c_acoustic=training_config.c_acoustic,
        ref_segment_size=training_config.ref_segment_size // model_config.hop_length,
    )
    print(loss, loss_x, loss_y)


if __name__ == "__main__":
    # test_transformer_tts()
    test_transformer_tts_v2()
