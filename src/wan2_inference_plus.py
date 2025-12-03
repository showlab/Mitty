import os
import argparse
import copy
import warnings
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf

import pytorch_lightning as L
from pytorch_lightning.utilities import rank_zero_only

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import FlowMatchEulerDiscreteScheduler

from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline
from models.wan2.attn_process import ConditionAttnProcessor2_0

from tools.my_schedule import FlowMatchScheduler, MyFlowMatchEulerDiscreteScheduler
from datasets.custom_dataset import CustomDataset


@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InteractionVideoSystemInfer(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hparams = opt
        self.is_configured = False

    def _align_frames(self, meta, gen, gt, is_one2three: bool):
        """对齐 meta / 生成 / GT 的帧数，输入 shape: [F, H, W, C]"""
        f_meta, f_gen, f_gt = meta.shape[0], gen.shape[0], gt.shape[0]
        if is_one2three and f_meta == f_gt and f_gen == f_meta - 1:
            meta = meta[1:]
            gt = gt[1:]
            return meta, gen, gt
        min_f = min(f_meta, f_gen, f_gt)
        return meta[:min_f], gen[:min_f], gt[:min_f]

    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True

        model_id = self.hparams.model_id

        # tokenizer / text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        )

        # VAE
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

        # schedulers（训练/采样与训练端对齐）
        if self.hparams.use_DiffSynth:
            self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
            self.train_scheduler.set_timesteps(1000, training=True)
        else:
            self.train_scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
        base_sampler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.sample_scheduler = UniPCMultistepScheduler.from_config(
            base_sampler.config, flow_shift=5
        )

        # transformer
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float32
        )

        # 冻结主干
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # gradient checkpoint（与训练端一致，可有可无）
        if getattr(self.hparams.training, "gradient_checkpointing", False):
            self.transformer.gradient_checkpointing = True
            self.transformer.enable_gradient_checkpointing()

        # latents 标准化参数
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )

        # 保存下 config（可用于 debug）
        self.vae_config = self.vae.config
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        # LoRA（仅在 use_lora=True 时准备 adapter 容器，具体权重后续再加载）
        self.using_lora = bool(self.hparams.use_lora)
        if self.using_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=96, lora_alpha=96, init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.transformer.add_adapter(transformer_lora_config)

        # 设置 ConditionAttnProcessor，与训练端一致
        for blk in self.transformer.blocks:
            blk.attn1.set_processor(ConditionAttnProcessor2_0())

        # 额外 patch embedding（与训练端一致）
        self.transformer.patch_embedding_extra = copy.deepcopy(self.transformer.patch_embedding).requires_grad_(True)

    @torch.no_grad()
    def encode_prompt(self, prompt_list, device):
        max_sequence_length = 512
        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = self.text_encoder(ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    def _load_lora_from_ckpt(self, ckpt_path, device):
        """
        只加载训练时保存的 LoRA / patch_embedding_extra：
        checkpoint['state_dict'] = {
            "transformer_processor": {...lora...},
            "patch_embedding_extra": {...}
        }
        - 自动过滤掉 shape 不匹配的键（打印跳过数量）
        """
        if ckpt_path in (None, "", "None", "null"):
            print("[Infer] No ckpt_path provided. Skip loading LoRA.")
            return

        print(f"[Infer] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" not in ckpt:
            print("[Infer] checkpoint has no 'state_dict' key; skip.")
            return

        sd_all = ckpt["state_dict"]

        # LoRA/attn processor
        if "transformer_processor" in sd_all:
            sd = sd_all["transformer_processor"]
            cur = self.transformer.state_dict()
            filtered = {k: v for k, v in sd.items() if (k in cur and cur[k].shape == v.shape)}
            skipped = [k for k in sd.keys() if k not in filtered]
            print(f"[Infer][LoRA] Load {len(filtered)}/{len(sd)} keys. Skipped {len(skipped)} mismatched keys.")
            missing_after = set(cur.keys()) - set(filtered.keys())
            # 使用 strict=False，允许缺失/额外键
            self.transformer.load_state_dict(filtered, strict=False)
        else:
            print("[Infer] 'transformer_processor' not found in ckpt.state_dict; skip LoRA.")

        # patch_embedding_extra
        if "patch_embedding_extra" in sd_all:
            sd2 = sd_all["patch_embedding_extra"]
            cur2 = self.transformer.state_dict()
            filtered2 = {k: v for k, v in sd2.items() if (k in cur2 and cur2[k].shape == v.shape)}
            skipped2 = [k for k in sd2.keys() if k not in filtered2]
            print(f"[Infer][patch_embedding_extra] Load {len(filtered2)}/{len(sd2)} keys. Skipped {len(skipped2)} mismatched keys.")
            self.transformer.load_state_dict(filtered2, strict=False)
        else:
            print("[Infer] 'patch_embedding_extra' not found in ckpt.state_dict; skip.")

    @torch.no_grad()
    def run_infer(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure_model()
        self.to(device)

        # 准备数据
        ds = CustomDataset(
            video_root=self.hparams.dataset.video_root,
            video_root2=self.hparams.dataset.video_root2,
            first_root=self.hparams.dataset.first_root,
            height=self.hparams.dataset.height,
            width=self.hparams.dataset.width,
            sample_n_frames=self.hparams.dataset.sample_n_frames,
            is_one2three=self.hparams.dataset.is_one2three,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=1, num_workers=self.hparams.dataset.num_workers,
            drop_last=False, pin_memory=self.hparams.dataset.pin_memory, shuffle=False
        )

        # 加载 LoRA/额外权重（来自训练时保存的 .ckpt）
        # 只有在 use_lora=True 时才尝试；否则跳过
        if self.using_lora:
            ckpt_path = getattr(self.hparams, "ckpt_path", None)
            self._load_lora_from_ckpt(ckpt_path, device)

        # 采样管线
        pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        )

        save_root = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'infer_samples')
        os.makedirs(save_root, exist_ok=True)
        print(f"[Infer] Save to: {save_root}")

        for batch_idx, batch in enumerate(dl):
            # 原图（用于可视化拼接）
            model_input = batch["pixel_values"].to(device)   # [1, C, F, H, W]
            model_input2 = batch["pixel_values2"].to(device) # [1, C, F, H, W]
            first_frames = batch.get('first_frames', None)
            if first_frames is not None:
                first_frames = first_frames.to(device).unsqueeze(2)  # [1, C, 1, H, W]
            prompts = batch["prompts"]

            is_one2three = self.hparams.dataset.is_one2three

            if is_one2three:
                # GT
                video_gt = model_input2.squeeze(0).permute(1, 0, 2, 3)
                video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
                video_gt = video_gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                # input 可视化
                meta = model_input.squeeze(0).permute(1, 0, 2, 3)
                meta = ((meta + 1) * 0.5).clamp(0, 1)
                meta = meta.permute(0, 2, 3, 1).detach().cpu().numpy()

                # latent
                model_input_lat = self.vae.encode(model_input).latent_dist.sample()
                model_input_lat = (model_input_lat - self.latents_mean) / self.latents_std

                # first_frames latent（缺失时占位）
                if first_frames is None:
                    first_frames_lat = model_input_lat[:, :, :1].detach() * 0
                else:
                    first_frames_lat = self.vae.encode(first_frames).latent_dist.sample()
                    first_frames_lat = (first_frames_lat - self.latents_mean) / self.latents_std

                attention_kwargs = {
                    'encoder_contion_states': model_input_lat,
                    'encoder_first_states': first_frames_lat,
                }
            else:
                # GT
                video_gt = model_input.squeeze(0).permute(1, 0, 2, 3)
                video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
                video_gt = video_gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                # input 可视化
                meta = model_input2.squeeze(0).permute(1, 0, 2, 3)
                meta = ((meta + 1) * 0.5).clamp(0, 1)
                meta = meta.permute(0, 2, 3, 1).detach().cpu().numpy()

                # latent
                model_input2_lat = self.vae.encode(model_input2).latent_dist.sample()
                model_input2_lat = (model_input2_lat - self.latents_mean) / self.latents_std
                attention_kwargs = {
                    'encoder_contion_states': model_input2_lat,
                }

            # prompt
            prompt_embeds = self.encode_prompt(prompts, device=device)  # 不需要手动传入到 pipeline，这里保留以防你扩展

            # 生成
            out = pipeline(
                prompt=prompts,
                height=self.hparams.dataset.height,
                width=self.hparams.dataset.width,
                num_frames=self.hparams.dataset.sample_n_frames,
                guidance_scale=5.0,
                attention_kwargs=attention_kwargs,
            )
            video_generate = out.frames[0]

            # 对齐帧
            meta, video_generate, video_gt = self._align_frames(meta, video_generate, video_gt, is_one2three)

            # 拼接保存
            concat = np.concatenate([meta, video_generate, video_gt], axis=1)
            save_path = os.path.join(save_root, f"batch_{batch_idx}.mp4")
            export_to_video(concat, output_video_path=save_path, fps=self.hparams.dataset.fps)
            print(f"[Infer] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_config.yaml", help="path to the yaml config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default="", help="path to a .ckpt with LoRA/extra weights")
    args, extras = parser.parse_known_args()
    args = vars(args)

    # 合并配置：YAML + CLI + 环境
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )

    # 归一化 ckpt 路径
    opt.ckpt_path = None if args['ckpt_path'] in ("", "null", "None") else args['ckpt_path']

    # 设定随机种
    L.seed_everything(opt.seed)

    # 跑推理
    system = InteractionVideoSystemInfer(opt)
    system.run_infer()


if __name__ == "__main__":
    main()
