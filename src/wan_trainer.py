import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import wandb
import copy
import numpy as np

import argparse
from omegaconf import OmegaConf
from tools.util import CustomProgressBar, CustomModelCheckpoint
from tools.util import compute_prompt_embeddings
from models.my_nets import FlowNet
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video, load_image, load_video

from models.cogvideox.custom_pipeline import InteractionVideoPipeline
from models.wan.transformer_wan import WanTransformer3DModel
from datasets.custom_dataset import CustomDataset
from tools.my_schedule import FlowMatchScheduler, MyFlowMatchEulerDiscreteScheduler
from diffusers import FlowMatchEulerDiscreteScheduler

from transformers import AutoTokenizer, UMT5EncoderModel
import torch
import random
import warnings
from einops import rearrange, repeat
from diffusers.models.attention_processor import Attention


@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore")
# silence_warnings()

os.environ["TOKENIZERS_PARALLELISM"] = "false"




class InteractionVideoSystem(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        # 自动保存配置文件，并且在训练过程可以动态访问，通过self.hparams访问
        self.save_hyperparameters(opt)
        #        
        self.is_configured = False

    # 导入模型
    def configure_model(self):
        if not self.is_configured:
            self.is_configured = True
            #
            model_id = self.hparams.model_id
            # breakpoint()
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch.float32
            )
            #
            self.vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32
            )
            # 
            if self.hparams.use_DiffSynth:
                self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
                self.train_scheduler.set_timesteps(1000, training=True) # Reset training scheduler
            else:
                self.train_scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.sample_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler") # set sample scheduler
            #
            self.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float32)
            #
            self.text_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.transformer.requires_grad_(False)
            #
            if self.hparams.training.gradient_checkpointing:
                self.transformer.gradient_checkpointing = True
                self.transformer.enable_gradient_checkpointing()


            self.register_buffer('latents_mean', torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1))
            self.register_buffer('latents_std', torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1))
            #
            self.vae_config = self.vae.config
            self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config


            # now we will add new LoRA weights to the attention layers
            if self.hparams.use_lora:
                # breakpoint()
                from peft import LoraConfig
                transformer_lora_config = LoraConfig(  # A矩阵用Kaiming Uniform，B矩阵用0
                    r=128,
                    lora_alpha=128,
                    init_lora_weights=True,
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
                self.transformer.add_adapter(transformer_lora_config) # 会先冻住所有的，然后再单独开启LoRA


            # set attn_processors
            from models.attn_process import ConditionAttnProcessor2_0
            for block_idx, attn_blcok in enumerate(self.transformer.blocks):
                # 设置控制条件的Processor
                attn_blcok.attn1.set_processor(
                    ConditionAttnProcessor2_0()
                )
            
            # 额外加一个patch embedding
            self.transformer.patch_embedding_extra = copy.deepcopy(self.transformer.patch_embedding).requires_grad_(True)


    # 定义前向过程
    def forward(self, model_input, model_input2, prompt_embeds):
        batch_size, num_channels, num_frames, height, width = model_input.shape
        #
        noise = torch.randn_like(model_input)
        timestep_id = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,))
        # 先试第三视角转第一视角，model_input2已知，求model_input
        if self.hparams.use_DiffSynth:
            timestep = self.train_scheduler.timesteps[timestep_id].to(dtype=model_input.dtype)  
            latent_noisy = self.train_scheduler.add_noise(model_input, noise, timestep)
            v_target = self.train_scheduler.training_target(model_input, noise, timestep)
            #
            attention_kwargs = {
                'encoder_contion_states': model_input2,
            }
            #
            v_pred = self.transformer(
                hidden_states=latent_noisy, # B, C, F, H, W
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                return_dict=False,
                attention_kwargs=attention_kwargs,
            )[0]
            # breakpoint()
            #
            loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float(), reduction='none')
            weight = self.train_scheduler.training_weight(timestep).to(loss.device)
            loss = (loss * weight[:, None, None, None, None]).mean()
        else:
            # breakpoint()
            timestep = self.train_scheduler.timesteps[timestep_id]
            latent_noisy = self.train_scheduler.scale_noise(model_input, timestep, noise)
            v_target = noise - model_input
            # For flow
            attention_kwargs = {
                'flow_embeds':flow_embeds,
                'encoder_history_states': history_input,
            }
            #
            v_pred = self.transformer(
                hidden_states=latent_noisy, # B, C, F, H, W
                encoder_hidden_states=prompt_embeds,
                timestep=timestep.to(latent_noisy.device),
                return_dict=False,
                attention_kwargs=attention_kwargs,
            )[0]
            loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return loss


    def encode_prompt(self, prompt):
        max_sequence_length = 512
        # breakpoint()
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # text_inputs = self.tokenizer(prompt, add_special_tokens=True)
        # ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        # breakpoint()
        # seq_lens = mask.gt(0).sum(dim=1).long()
        # # 
        # prompt_emb = self.text_encoder(ids.to(self.device), mask.to(self.device)).last_hidden_state
        # for i, v in enumerate(seq_lens):
        #     prompt_emb[:, v:] = 0
        # return prompt_emb
        # breakpoint()
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = self.text_encoder(ids.to(self.device), mask.to(self.device)).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    def process_data(self, batch, batch_idx):
        model_input = batch["pixel_values"] # B, C, F, H, W
        model_input2 = batch["pixel_values2"] # B, C, F, H, W
        first_frames = batch['first_frames'].unsqueeze(2) # # B, C, 1, H, W
        prompts = batch["prompts"]
        if self.hparams.use_drop_text:
            prompts = [prompt if random.random() < 0.5 else '' for prompt in prompts]
        # ---------------------------------------------------------------------------------------

        return model_input, model_input2, prompts, 

    # 模拟每个batch的循环
    def training_step(self, batch, batch_idx):
        model_input, model_input2, prompts = self.process_data(batch, batch_idx)
        # ---------------------------------------------------------------------------------------
        # model input
        model_input = self.vae.encode(model_input).latent_dist.sample() # [B, C, F, H, W]
        model_input = (model_input - self.latents_mean) / self.latents_std # scaling
        # history input
        model_input2 = self.vae.encode(model_input2).latent_dist.sample() # [B, C, F, H, W]
        model_input2 = (model_input2 - self.latents_mean) / self.latents_std # scaling
        # encode prompts
        # prompt_embeds.shape --> torch.Size([B, 512, 4096])
        prompt_embeds = self.encode_prompt(prompts)

        # forward
        loss = self.forward(model_input, model_input2, prompt_embeds)        
        # 记录loss
        self.log("train/loss", loss, prog_bar=True, on_step=True,
                logger=True, sync_dist=True if self.trainer.world_size > 1 else False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True,
                on_step=True, logger=True, sync_dist=True if self.trainer.world_size > 1 else False)

        return loss

    def on_validation_epoch_start(self): # 每次验证开始前（可以是训练epoch单位，也可以step单位，取决于验证频率粒度），初始化一下pipe，因为transformer在更新，每轮的权重都不一样。
        self.pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        )
        self.print(f"validation at step: {self.global_step}.")
        val_path = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'val_samples')
        os.makedirs(val_path, exist_ok=True)

    def validation_step(self, batch, batch_idx): # 每个batch的验证逻辑
        model_input = batch["pixel_values"] # B, C, F, H, W
        model_input2 = batch["pixel_values2"] # B, C, F, H, W
        # first_frames = batch['first_frames'].unsqueeze(2) # # B, C, 1, H, W
        prompts = batch["prompts"]
        #
        # transforms pixel_values to gt, for save.
        model_input = model_input.squeeze(0).permute(1, 0, 2, 3)
        model_input = ((model_input + 1) * 0.5).clamp(0, 1)
        model_input = model_input.permute(0, 2, 3, 1).cpu().numpy()
        # ---------------------------------------------------------------------------------
        model_input2 = self.vae.encode(model_input2).latent_dist.sample() # [B, C, F, H, W]
        model_input2 = (model_input2 - self.latents_mean) / self.latents_std # scaling
        attention_kwargs = {
            'encoder_contion_states': model_input2,
        }
        #
        # breakpoint()
        video_generate = self.pipeline(
            prompt=prompts,
            height=480,
            width=832,
            num_frames=41,
            guidance_scale=1.0,
            attention_kwargs=attention_kwargs,
        )
        video_generate = video_generate.frames[0]    
        # breakpoint()
        concatenated_video = np.concatenate([video_generate, model_input], axis=2)
        val_path = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'val_samples')
        val_video_path = os.path.join(val_path, f"val_{self.global_step}step-batch_{batch_idx}-rank{self.trainer.global_rank}.mp4")
        export_to_video(concatenated_video, output_video_path=val_video_path, fps=self.hparams.dataset.fps)
        # 只让主GPU记录日志
        if self.trainer.is_global_zero and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                f"val/video_{self.global_step}step-batch_{batch_idx}": wandb.Video(
                    val_video_path,
                    caption=f"Validation video - step {self.global_step}, batch {batch_idx}",
                    format="mp4"
                )
            })


    # 管理参数梯度，优化器，调度器
    def configure_optimizers(self):
        # 管理参数
        params_and_lrs = []
        modules = [self.transformer] # 只有一个transformer，没有额外的参数
        for module in modules:
            # 获取需要梯度的参数
            params = [p for p in module.parameters() if p.requires_grad]
            # 计算学习率
            learning_rate = self.hparams.training.learning_rate * (self.hparams.training.accumulate_grad_batches * self.hparams.num_gpus * self.hparams.num_nodes) ** 0.5
            params_and_lrs.append(
                {
                    "params": params, 
                    "lr": learning_rate
                }
            )
        # breakpoint()
        # 管理优化器
        optimizer = torch.optim.AdamW(
            params_and_lrs,
            betas=(0.9, 0.95), # 一般固定
            eps=1e-8, # 一般固定
            weight_decay=self.hparams.training.weight_decay,  # 默认 0.01
        )
        # 管理调度器
        def lr_fn(step, warmup_steps):
            if warmup_steps <= 0:
                return 1
            else:
                return min(step / warmup_steps, 1)
        #
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: lr_fn(step, warmup_steps=self.hparams.training.warmup_steps),
        )
        # 返回优化器和调度器的字典
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }

    # 自定义模型加载方式
    def load_state_dict(self, state_dict, strict: bool = True):
        # breakpoint()
        # load for attn-processor
        self.transformer.load_state_dict(state_dict['transformer_processor'], strict=False)

    # 保存前自定义调整checkpoint的state_dict
    def on_save_checkpoint(self, checkpoint):
        del checkpoint['hparams_name']
        del checkpoint['hparams_type']
        # reset model_state_dict
        model_state_dict = {}
        # transformer_processor
        tmp_dict = {}
        for name, param in self.transformer.state_dict().items():
            if "lora" in name:
                # breakpoint()
                tmp_dict[name] = param.cpu()
        model_state_dict["transformer_processor"] = tmp_dict
        # attn
        tmp_dict = {}
        for name, param in self.transformer.state_dict().items():
            # if "patch_history_embedding" in name or "norm_history" in name or "attn_history" in name:
            if "patch_history_embedding" in name:
                # breakpoint()
                tmp_dict[name] = param.cpu()
        model_state_dict["addition"] = tmp_dict      
        #
        checkpoint['state_dict'] = model_state_dict


def main(opt):
    # set seed
    L.seed_everything(opt.seed)
    # Dataset && Dataloader
    train_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        video_root2=opt.dataset.video_root2,
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
        training_len=opt.num_nodes * opt.num_gpus * opt.training.accumulate_grad_batches * opt.training.max_steps * opt.training.batch_size # 自动计算样本数
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.training.batch_size,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last, # 丢掉最后一个不符合batch数的样本，避免 batch size 不一致引发 BN 问题。
        pin_memory=opt.dataset.pin_memory, # 加快 CPU→GPU 的数据拷贝速度。
        shuffle=opt.dataset.shuffle, # 是否打乱，一般训练时候打乱，测试时候不打乱。
    )
    val_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        video_root2=opt.dataset.video_root2,
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last, # 丢掉最后一个不符合batch数的样本，避免 batch size 不一致引发 BN 问题。
        pin_memory=opt.dataset.pin_memory, # 加快 CPU→GPU 的数据拷贝速度。
        shuffle=False, # 是否打乱，一般训练时候打乱，测试时候不打乱。
    )
    # Custom System
    system = InteractionVideoSystem(opt)
    # Custom Logger
    wandb_logger = WandbLogger(
        project=opt.experiment_project,       # 项目名称（wandb 项目中显示）
        name=opt.experiment_name,             # 当前实验名（可选）
        save_dir=os.path.join(opt.output_root, opt.experiment_name),         # 日志保存路径（本地）
        log_model=False,                      # 是否保存模型 checkpoint 到 wandb
        offline=False,                         # 离线模式
    )
    # Define Trainer
    trainer = L.Trainer(
        # logger=wandb_logger,
        logger=False,
        max_steps=opt.training.max_steps, # 一共优化多少次
        precision=opt.training.precision,
        num_sanity_val_steps=1, #  训练前，val_dataloader() 中取 1 个 batch，执行 validation_step() 进行“预验证”
        limit_val_batches=1,  # 只用 1 批 batch 做验证，0表示跳过验证
        val_check_interval=opt.training.save_val_interval_steps * opt.training.accumulate_grad_batches, # 每多少个样本batch优化step验证一次
        accumulate_grad_batches=opt.training.accumulate_grad_batches, # 梯度累积
        gradient_clip_val=opt.training.gradient_clip_val, # 梯度裁剪
        gradient_clip_algorithm='value', # 按值裁剪 
        log_every_n_steps=1, # 多少个step记录一次
        accelerator=opt.training.accelerator, # 
        strategy=opt.training.strategy, # or 'ddp_find_unused_parameters_true' optioanl [deepspeed]
        benchmark=opt.training.benchmark,
        callbacks=[
            CustomProgressBar(), # 自定义显示条
            CustomModelCheckpoint(
                dirpath=os.path.join(opt.output_root, opt.experiment_name, 'checkpoints'),     # 模型保存路径
                filename="{step}",                                                             # 文件名包含step信息
                every_n_train_steps=opt.training.save_val_interval_steps,                      # 每间隔个训练步骤保存一次
                # every_n_train_steps=1,                                                         # 每间隔个训练步骤保存一次
                save_top_k=-1,                                                                 # 保存所有模型
                save_weights_only=False,                                                       # 是否只保存模型权重
                verbose=False,                                                                 # 开了没啥用
            )
        ],
        num_nodes=opt.num_nodes, # 多机训练，节点数量
    )
    # load model -- for debug
    # system.load_state_dict(torch.load("1234.ckpt"))
    # breakpoint()
    trainer.fit(system,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                # ckpt_path=opt.training.resume_path, # 断点恢复，单纯导入模型权重可以system.load_state_dict()
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml", help="path to the yaml config file")
    # Additional paras
    parser.add_argument("--seed", type=int, default=42)
    # ----------------------------------------------------------------------
    args, extras = parser.parse_known_args() # 将预定义的参数和命令行额外定义的参数，分离开。
    args = vars(args)
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']), # yaml文件中的参数
        OmegaConf.from_cli(extras), # 命令行额外传入的参数
        OmegaConf.create(args), # 额外定义的参数
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}), # 环境变量
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}), # 环境变量 
    )
    # ----------------------------------------------------------------------
    main(opt)
