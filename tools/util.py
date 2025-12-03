from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
from typing import Any, Dict, List, Tuple, Optional
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as L

from transformers import T5EncoderModel, T5Tokenizer
from typing import List, Optional, Tuple, Union
import os
import math
import torchvision



class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, sanity_checks=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sanity_checks = sanity_checks

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 计算总共扫过的batch数
        training_step = batch_idx + trainer.current_epoch * trainer.num_training_batches
        # 判断是否要保存
        # breakpoint()
        if self.sanity_checks and training_step == 0: # 安全性保存
            samples_path = os.path.join(pl_module.hparams.output_root, pl_module.hparams.experiment_name, 'sanity_checks')
            os.makedirs(samples_path, exist_ok=True)
            pl_module.print(f"Sanity_checks, Save to {samples_path}")
            #
            samples = batch['pixel_values2'].cpu()
            [
                torchvision.io.write_video(
                    os.path.join(samples_path, f"video_{idx}.mp4"),
                    ((sample.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0),
                    fps=8,
                )
                for idx, sample in enumerate(samples)
            ]
        # ✅ 保持 ModelCheckpoint 的原始逻辑
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # log save information
        pl_module.print(f"⏰ Save ckpt at: epoch={trainer.current_epoch}, step={trainer.global_step}.")



class CustomProgressBar(TQDMProgressBar):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.train_progress_bar.reset(self.total_train_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Training step")

    def on_train_epoch_start(self, trainer=None, pl_module=None):
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        self.train_progress_bar.reset(self.total_train_batches)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Training step")

    def get_metrics(self, trainer, pl_module):
        # 获取默认指标
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)  # 去掉 version
        # 格式化 float 指标
        formatted = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                if "lr" in k:
                    formatted[k] = f"{v:.7f}"
                elif "step" in k:
                    formatted[k] = int(v)
                else:
                    formatted[k] = f"{v:.3f}"
            else:
                formatted[k] = v
        return formatted


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False,
    is_global=False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                is_global=is_global,
            )
    return prompt_embeds

def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
    is_global=False,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
        is_global=is_global,
    )
    return prompt_embeds

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
    is_global=False
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        # breakpoint()
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")


    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # breakpoint()
    if is_global:
        prompt_embeds=prompt_embeds[:, 0, :].unsqueeze(1)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Dict,
    vae_scale_factor_spatial: int,
    device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

def masks_like(tensor, zero=False, generator=None, p=0.2):
    # breakpoint()
    if not isinstance(tensor, list):
        tensor = [tensor]
    # 生成全1矩阵
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
    #
    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            # breakpoint()
            # 走了这个分支,第一帧padding为0
            for u, v in zip(out1, out2):
                u[:, :, 0] = torch.zeros_like(u[:, :, 0])
                v[:, :, 0] = torch.zeros_like(v[:, :, 0])

    return out1, out2
