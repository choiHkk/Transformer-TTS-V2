from dataclasses import dataclass


@dataclass
class TransformerTTSv2Config:
    model_architecture_type: str = "cross_entropy"

    sampling_rate: int = 24000
    n_fft: int = 1280
    hop_length: int = 320
    win_length: int = 1280
    fmin: int = 0
    fmax: int = 12000

    n_linguistic_vocabs: int = 200
    n_acoustic_vocabs: int = 1024
    n_mel_channels: int = 80
    d_model: int = 512
    decoder_in_channels: int = 80
    encoder_kernel_size: int = 5
    decoder_kernel_size: int = 5
    n_encoder_prenet_layers: int = 3
    n_encoder_layers: int = 4
    n_encoder_heads: int = 8
    n_decoder_prenet_layers: int = 1
    n_decoder_layers: int = 12
    n_decoder_heads: int = 16
    p_dropout: float = 0.1
    sequence_max_length: int = 1024

    gst_n_tokens: int = 16
    gst_n_layers: int = 4
    gst_n_heads: int = 8
    gst_p_dropout: float = 0.1

    n_quantizers: int = 1


@dataclass
class TransformerTTSv2TrainingConfig:
    batch_size: int = 32
    ref_segment_size: int = 10240  # 320 hopsize, 32 frame
    c_reconstuction: float = 1.0
    c_gate: float = 10.0
    c_linguistic: float = 0.01
    c_acoustic: float = 1.0


@dataclass
class TransformerTTSv2InferenceConfig:
    gate_threshold: int = 0.5
    max_length: int = 1000


if __name__ == "__main__":
    model_config = TransformerTTSv2Config()
    training_config = TransformerTTSv2TrainingConfig()
    inference_config = TransformerTTSv2InferenceConfig()

    print(model_config)
    print(training_config)
    print(inference_config)
