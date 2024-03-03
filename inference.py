import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Tokenizer, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from configs.cross_entropy import TransformerTTSv2Config
from models import TransformerTTSv2


class HFTransformerTTSv2Config(PretrainedConfig, TransformerTTSv2Config):
    model_type = "transformer_tts_v2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HFTransformerTTSv2(GPT2PreTrainedModel):
    config_class = HFTransformerTTSv2Config

    def __init__(self, config: HFTransformerTTSv2Config, model: TransformerTTSv2):
        super().__init__(config)
        self.transformer = model
        self.lm_head = nn.Sequential(
            self.transformer.final_norm, self.transformer.proj_a
        )

    def prepare_inputs_for_generation(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor,
        **kwargs
    ):
        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "g": g,
        }

    def get_start_code(self, x: torch.Tensor, start_code_ids: int = 0):
        start_code = torch.zeros(1, config.n_quantizers, 1).long().to(x.device)
        start_code = start_code.fill_(start_code_ids)
        start_code_lengths = torch.LongTensor([1]).to(x.device)
        return start_code, start_code_lengths

    @torch.no_grad()
    def pretrained_model_forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor,
    ):
        _, _, y, _ = self.transformer(
            x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, g=g, return_logits=False
        )
        return y

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor,
        **kwargs
    ):
        y = self.pretrained_model_forward(
            x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, g=g
        )
        lm_logits = self.lm_head(y)
        return CausalLMOutputWithCrossAttentions(loss=None, logits=lm_logits)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        g: torch.Tensor,
        **kwargs
    ):
        T_x = int(x.size(1) + g.size(2))
        y = super().generate(
            inputs=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            g=g,
            max_length=self.config.max_length + T_x,
            **kwargs
        )
        y = y.unsqueeze(1)
        y = y[:, :, T_x:]
        y_lengths = torch.LongTensor([y.size(2)] * y.size(0)).to(y.device)
        return y, y_lengths

    @torch.inference_mode()
    def inference(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: torch.Tensor, **kwargs
    ):
        y, y_lengths = self.get_start_code(x=x, start_code_ids=0)
        codes, code_lengths = self.generate(
            x=x, x_lengths=x_lengths, y=y, y_lengths=y_lengths, g=g, **kwargs
        )
        y = self.pretrained_model_forward(
            x=x, x_lengths=x_lengths, y=codes, y_lengths=code_lengths, g=g
        )
        return y


if __name__ == "__main__":
    from mel_processing import mel_spectrogram_torch

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    config = HFTransformerTTSv2Config()
    config.n_linguistic_vocabs = tokenizer.vocab_size
    model = TransformerTTSv2(config)
    model = HFTransformerTTSv2(config=config, model=model)

    text = "hi? good to see you."
    tokens = tokenizer(text)["input_ids"]

    B = 1
    T_enc = len(tokens)
    T_dec = config.sampling_rate

    x = torch.LongTensor(tokens).unsqueeze(0)
    x_lengths = torch.LongTensor([T_enc])

    y_ref = 2 * torch.rand(B, T_dec) - 1
    y_ref = mel_spectrogram_torch(
        y=y_ref.squeeze(1),
        n_fft=config.n_fft,
        n_mel_channels=config.n_mel_channels,
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        win_length=config.win_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )

    y = model.inference(
        x=x,
        x_lengths=x_lengths,
        g=y_ref,
        top_p=0.85,
        top_k=50,
        do_sample=True,
        num_beams=1,
    )
