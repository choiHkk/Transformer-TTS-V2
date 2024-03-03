from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TransformerTTSConfig:
    model_architecture_type: str = "reconstruction"

    sampling_rate: int = 24000
    hop_length: int = 320

    n_vocabs: int = 200
    n_mel_channels: int = 80
    d_model: int = 512
    decoder_in_channels: int = 80
    encoder_kernel_size: int = 5
    decoder_kernel_size: int = 5
    postnet_kernel_size: int = 5
    n_encoder_prenet_layers: int = 3
    n_encoder_layers: int = 6
    n_encoder_heads: int = 8
    n_decoder_prenet_layers: int = 1
    n_decoder_layers: int = 6
    n_decoder_heads: int = 8
    n_postnet_layers: int = 5
    p_dropout: float = 0.1
    sequence_max_length: int = 1024

    gst_n_tokens: int = 16
    gst_in_channels: int = 128
    gst_hidden_channels: int = 128
    gst_filter_channel_list: List[int] = field(
        default_factory=lambda: [32, 32, 64, 64, 128, 128]
    )
    gst_n_layers: int = 4
    gst_n_heads: int = 8
    gst_kernel_size: Tuple[int, int] = (3, 3)
    gst_stride: Tuple[int, int] = (2, 2)
    gst_p_dropout: float = 0.1


@dataclass
class TransformerTTSTrainingConfig:
    batch_size: int = 32
    ref_segment_size: int = 10240  # 320 hopsize, 32 frame
    c_reconstuction: float = 1.0
    c_gate: float = 10.0
    c_linguistic: float = 0.01
    c_acoustic: float = 1.0


@dataclass
class TransformerTTSInferenceConfig:
    gate_threshold: int = 0.5
    max_length: int = 1000


if __name__ == "__main__":
    model_config = TransformerTTSConfig()
    training_config = TransformerTTSTrainingConfig()
    inference_config = TransformerTTSInferenceConfig()

    print(model_config)
    print(training_config)
    print(inference_config)
