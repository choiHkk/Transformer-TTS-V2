from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerTTSv3Config:
    model_architecture_type: str = "cross_entropy_hf"

    sampling_rate: int = 24000
    n_fft: int = 1280
    hop_length: int = 320
    win_length: int = 1280
    fmin: int = 0
    fmax: int = 12000

    n_linguistic_vocabs: int = 50270
    n_acoustic_vocabs: int = 1024
    n_mel_channels: int = 80
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    p_dropout: float = 0.1
    max_linguistic_length: int = 1024
    max_acoustic_length: int = 1024
    max_prompt_length: int = 1024
    gradient_checkpointing: bool = False

    gst_n_tokens: int = 16
    gst_n_layers: int = 4
    gst_n_heads: int = 8
    gst_p_dropout: float = 0.1

    n_quantizers: int = 1

    linguistic_start_token: Optional[int] = None
    linguistic_stop_token: Optional[int] = None

    acoustic_start_token: Optional[int] = None
    acoustic_stop_token: Optional[int] = None

    c_vocab: int = 32


@dataclass
class TransformerTTSv3MediumConfig(TransformerTTSv3Config):
    d_model: int = 1024
    n_layers: int = 30
    n_heads: int = 16
    gradient_checkpointing: bool = True


@dataclass
class TransformerTTSv3LargeConfig(TransformerTTSv3Config):
    d_model: int = 1536
    n_layers: int = 32
    n_heads: int = 24
    gradient_checkpointing: bool = True


@dataclass
class TransformerTTSv3TrainingConfig:
    batch_size: int = 32
    ref_segment_size: int = 10240  # 320 hopsize, 32 frame
    c_linguistic: float = 0.01
    c_acoustic: float = 1.0


@dataclass
class TransformerTTSv3InferenceConfig:
    pass


if __name__ == "__main__":
    model_config = TransformerTTSv3Config()
    training_config = TransformerTTSv3TrainingConfig()
    inference_config = TransformerTTSv3InferenceConfig()

    print(model_config)
    print(training_config)
    print(inference_config)

    model_medium_config = TransformerTTSv3MediumConfig()
    model_large_config = TransformerTTSv3LargeConfig()
    print(model_medium_config)
    print(model_large_config)
