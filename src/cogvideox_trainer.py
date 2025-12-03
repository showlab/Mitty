import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
import wandb

import argparse
from omegaconf import OmegaConf
from tools.util import CustomProgressBar, CustomModelCheckpoint
from tools.util import compute_prompt_embeddings
from models.my_nets import FlowNet

from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video

from models.cogvideox.custom_pipeline import InteractionVideoPipeline
from models.cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from datasets.custom_dataset import CustomDataset

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import torch
import warnings
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore")
silence_warnings()

os.environ["TOKENIZERS_PARALLELISM"] = "false"



class InteractionVideoSystem(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        # 自动保存配置文件，并且在训练过程可以动态访问，通过self.hparams访问
        self.save_hyperparameters(opt)
        # 损失函数
        self.loss_fn = F.mse_loss

        # 如果保存了 epoch，则恢复
        try:
            self.resume_epoch = torch.load(self.hparams.training.resume_ckpt, weights_only=True).get("epoch", 0) # 如果没存，就返回 0
        except:
            self.resume_epoch = 0


    # 管理生命周期，每个stage都会调用一次，optional: [fit, validate, test, predict]
    def setup(self, stage=None):
        if stage == 'fit':
            self.print(f"🕒 从 ckpt 中恢复的 epoch 是: {self.resume_epoch}")
        if True:
            model_id = self.hparams.model_id
            # breakpoint()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, subfolder="tokenizer", revision=None
            )
            #
            self.text_encoder = T5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder", revision=None
            )
            #
            self.vae = AutoencoderKLCogVideoX.from_pretrained(
                model_id, subfolder="vae", revision=None, variant=None
            )
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            #
            self.scheduler = CogVideoXDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
            load_dtype = torch.bfloat16 if "5b" in model_id.lower() else torch.float16
            #
            # 导入模型权重
            self.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                torch_dtype=torch.float32,
                revision=None,
                variant=None,
                )
            #
            self.text_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.transformer.requires_grad_(False)
            #
            if self.hparams.training.gradient_checkpointing:
                self.transformer.gradient_checkpointing = True
                self.transformer.enable_gradient_checkpointing()

            self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

            # set attn_processors
            attn_processors = {}
            from models.attn_process import FlowCogVideoXAttnProcessor2_0
            for key, value in self.transformer.attn_processors.items():
                block_idx = int(key.split(".")[1])
                if block_idx in list(range(self.transformer.config.num_layers)):
                    # breakpoint()
                    attn_processor = FlowCogVideoXAttnProcessor2_0(
                        block_index=block_idx
                    ).to(self.device)
                    # set learnable
                    for param in attn_processor.parameters():
                    # for param in attn_processor.parameters():
                        param.requires_grad_(True)
                    attn_processors[key] = attn_processor
                else:
                    attn_processors[key] = value
            self.transformer.set_attn_processor(attn_processors)
            
            # flow_net
            self.flow_net = FlowNet(entity_dim=4096, out_dim=512)
            

            self.load_ckpt()

        # 管理参数梯度
        if stage == 'fit':
            # now we will add new LoRA weights to the attention layers
            from peft import LoraConfig
            # A矩阵用Kaiming Uniform，B矩阵用0
            # output = base_layer_output + scaling * (lora_B(lora_A(x)))
            # transformer_lora_config = LoraConfig(
            #     r=16,
            #     lora_alpha=0.8,
            #     init_lora_weights=True,
            #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            # )
            # # 会先冻住所有的，然后再单独开启LoRA
            # self.transformer.add_adapter(transformer_lora_config)

            # for idx in range(5, 10):
            #     for param in self.transformer.transformer_blocks[idx].parameters():
            #         param.requires_grad_(True)
            #     self.transformer.transformer_blocks[idx].train()    
    
    def load_ckpt(self,):
        if self.hparams.training.resume_ckpt is not None:
            # load attn processor
            state_dict = torch.load(self.hparams.training.resume_ckpt, weights_only=True)['transformer_processor']
            # breakpoint()
            all_keys = self.transformer.state_dict()
            missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, strict=False)
            if not missing_keys and not unexpected_keys:
                print(f"✅ 成功加载 transformer 权重: {self.hparams.training.resume_ckpt}")
            else:
                print(f"加载 transformer 权重时存在部分不匹配:")
                print(f"一共需要{len(all_keys)}的参数")
                print(f"一共导入了{len(state_dict)}参数")
                if missing_keys:
                    print(f"- 缺失参数: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"  - 多余参数: {len(unexpected_keys)}")
            # load self.flow_net
            state_dict = torch.load(self.hparams.training.resume_ckpt, weights_only=True)['flow_net']
            try:
                self.flow_net.load_state_dict(state_dict)
                print(f"✅ 成功加载 flow_net 权重")
            except:
                raise ValueError

    # 定义前向过程
    def forward(self, model_input, image_latents, prompt_embeds, flow_embeds):
        batch_size, num_channels, num_frames, height, width = model_input.shape
        model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config
        # from [B, C, F, H, W] to [B, F, C, H, W]
        model_input = model_input.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        # breakpoint()
        assert (model_input.shape[0], *model_input.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        )
        # Add noise to latent
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=model_input.device,
        ).long()
        latent_noisy = self.scheduler.add_noise(model_input, torch.randn_like(model_input), timesteps)

        # image_latents只有一帧，要把它padding到和video token一样的帧，填0
        # torch.Size([2, 1, 16, 60, 90]) ---> torch.Size([2, 13, 16, 60, 90])
        padding_shape = (model_input.shape[0], model_input.shape[1] - 1, *model_input.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)
        

        # Concatenate latent and image_latents
        # latent_noisy.shape torch.Size([2, 13, 16, 60, 90]) | image_latents.shape torch.Size([2, 13, 16, 60, 90])
        # breakpoint()
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)


        # Prepare rotary embeds
        from tools.util import prepare_rotary_positional_embeddings
        # breakpoint()
        rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * self.vae_scale_factor_spatial,
                width=width * self.vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=self.model_config,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                device=self.device,
            )
            if model_config.use_rotary_positional_embeddings
            else None
        )
        # breakpoint()
        
        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.model_config.ofs_embed_dim is None
            else model_input.new_full((1,), fill_value=2.0)
        )
        # breakpoint()
        
        # For I2V
        attention_kwargs = {
            'flow_emb':flow_embeds
        }
        predicted_noise = self.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
            attention_kwargs=attention_kwargs,
        )[0]
        # For T2V
        # model_output = self.transformer(
        #     hidden_states=noisy_model_input,    # [B, F, C, H, W]
        #     encoder_hidden_states=prompt_embeds,
        #     timestep=timesteps,
        #     image_rotary_emb=image_rotary_emb,
        #     return_dict=False,
        # )[0]
        latent_pred = self.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        return latent_pred, model_input, timesteps, batch_size

    # 模拟每个batch的循环
    def training_step(self, batch, batch_idx):
        model_input = batch["pixel_values"] # B, C, F, H, W
        first_frames = batch['first_frames'].unsqueeze(2) # # B, C, 1, H, W
        prompts = batch["prompts"]
        mask_values = batch['mask_values'] # B, C, F, H, W
        flow_values = batch['flow_values'] # B, 2, F, H, W，最后一帧padding 0
        masked_flow = batch['masked_flow'] # B, 2, F, H, W
        # ---------------------------------------------------------------------------------------
        # 处理masked_flow 有问题！！！
        # breakpoint()
        area_s=mask_values[:, 0, :, :, :].unsqueeze(1).sum(dim=[-1, -2]) # B, 1, F
        refined_flow = masked_flow.sum(dim=(-1, -2)) / (area_s + 1e-8)  # # B, 2, F
        norm = torch.norm(refined_flow, dim=1, keepdim=True) + 1e-8  # [B, 1, N]
        # 单位方向 + 长度 # [B, 3, N] ---> [B, N, 3]
        refined_flow = torch.cat([refined_flow / norm, norm], dim=1).permute(0, 2, 1) # [B, N, 3]
        # refined_flow = torch.clamp(refined_flow, min=-2.0, max=2.0)
        # ---------------------------------------------------------------------------------------
        # model input
        model_input = self.vae.encode(model_input).latent_dist.sample() * self.vae.config.scaling_factor # [B, C, F, H, W]
        # Add noise to images
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=first_frames.dtype)
        noisy_images = (
            first_frames + torch.randn_like(first_frames) * image_noise_sigma[:, None, None, None, None]
        )
        # breakpoint()
        image_latents = self.vae.encode(
            noisy_images.to(dtype=self.vae.dtype)
        ).latent_dist.sample() * self.vae.config.scaling_factor
        # breakpoint()

        # encode prompts
        # prompt_embeds.shape --> torch.Size([B, 226, 4096])
        prompt_embeds = compute_prompt_embeddings(
            self.tokenizer,
            self.text_encoder,
            prompts,
            self.model_config.max_text_seq_length,
            self.device,
            torch.bfloat16,
        )
        # prompt_embeds.shape --> torch.Size([B, 226, 4096])
        # entity_embeds = compute_prompt_embeddings(
        #     self.tokenizer,
        #     self.text_encoder,
        #     ["squirrel", "squirrel"],
        #     self.model_config.max_text_seq_length,
        #     self.device,
        #     torch.bfloat16,
        #     is_global=True,
        # )
        # breakpoint()
        # prompt_embeds.shape ---> torch.Size([2, 226, 4096]) | torch.Size([2, F, 3])
        flow_embeds = self.flow_net(
            inputs=refined_flow,
        )
        # breakpoint()
        # 考虑这个flow_embeds怎么用上去
        # forward
        latent_pred, latent, timesteps, batch_size = self.forward(model_input, image_latents, prompt_embeds, flow_embeds)        
        #
        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)
        # breakpoint()
        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = 1.0 * loss.mean()

        # 记录loss
        self.log("train/loss", loss, prog_bar=True, on_step=True,
                logger=True, sync_dist=True if self.trainer.world_size > 1 else False)
        self.log("global/step", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True,
                prog_bar=True, logger=True, sync_dist=True if self.trainer.world_size > 1 else False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True,
                prog_bar=True, logger=True, sync_dist=True if self.trainer.world_size > 1 else False)

        return loss

    def on_validation_epoch_start(self): # 每轮验证开始前，初始化一下pipe，因为transformer在更新，每轮的权重都不一样。
        load_dtype = torch.bfloat16 if "5b" in self.hparams.model_id.lower() else torch.float16
        # ori_dtype = self.transformer.dtype
        # breakpoint()
        # self.transformer.to(load_dtype)
        self.pipeline = InteractionVideoPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.scheduler,
        )

        os.makedirs(self.hparams.val_path, exist_ok=True)


    
    def validation_step(self, batch, batch_idx): # 每个batch的验证逻辑
        
        image = load_image(image="/data/lxy/sqj/code/Wan2.1/my_case_img/resize1.png")
        model_input = batch["pixel_values"]
        B, C, F, H, W = model_input.shape
        
        refined_flow = torch.tensor([0.5, 0.5]).unsqueeze(0).unsqueeze(-1).expand(B, 2, F).to(device=self.device, dtype=torch.bfloat16) # B, 2, F
        norm = torch.norm(refined_flow, dim=1, keepdim=True) + 1e-8  # [B, 1, N]
        # 单位方向 + 长度 # [B, 3, N] ---> # [B, N, 3]
        refined_flow = torch.cat([refined_flow / norm, norm], dim=1).permute(0, 2, 1)  # [B, 3, N]

        flow_embeds = self.flow_net(
            inputs=refined_flow,
        )

        attention_kwargs = {
            'flow_emb':flow_embeds
        }


        with torch.no_grad():
            video_generate = self.pipeline(
                height=self.hparams.dataset.height,
                width=self.hparams.dataset.width,
                prompt="squirrel", # 多个batch会自动广播
                image=image,
                # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
                num_videos_per_prompt=1,  # Number of videos to generate per prompt
                num_inference_steps=50,  # Number of inference steps
                num_frames=49,  # Number of frames to generate
                use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(self.hparams.seed),  # Set the seed for reproducibility
                attention_kwargs=attention_kwargs,
            )
        video_generate = video_generate.frames[0]    
        # breakpoint()
        output_video_path = os.path.join(self.hparams.val_path, f"test_{self.current_epoch}epoch-batch_{batch_idx}-rank{self.trainer.global_rank}.mp4")
        export_to_video(video_generate, output_video_path=output_video_path, fps=self.hparams.dataset.fps)
        
        # 只让主GPU记录日志
        if self.trainer.is_global_zero and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                f"val/video_epoch{self.current_epoch}_b{batch_idx}": wandb.Video(
                    output_video_path,
                    caption=f"Validation video - epoch {self.current_epoch}, batch {batch_idx}",
                    fps=self.hparams.dataset.fps,
                    format="mp4"
                )
            })


    # 管理参数梯度，优化器，调度器
    def configure_optimizers(self):
        # 管理参数
        params_and_lrs = []
        modules = [self.transformer, self.flow_net]
        for module in modules:
            params_and_lrs.append(
                {"params": filter(lambda p: p.requires_grad, module.parameters()), 
                "lr": self.hparams.training.learning_rate * \
                    (self.hparams.training.accumulate_grad_batches ** 0.5)}
            )
        # 管理优化器
        optimizer = torch.optim.AdamW(
            params_and_lrs,
            betas=(0.9, 0.95), # 一般固定
            eps=1e-8, # 一般固定
            weight_decay=self.hparams.training.weight_decay,  # 默认 0.01
        )
        # 管理调度器
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(step / 50, 1)
        )

        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.9,
        )

        # 返回优化器和调度器的字典
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }


def main(opt):
    # Dataset && Dataloader
    train_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        mask_root=opt.dataset.mask_root,
        flow_root=opt.dataset.flow_root,
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.training.batch_size,
        num_workers=opt.dataset.num_workers,
        shuffle=opt.dataset.shuffle, # 是否打乱，一般训练时候打乱，测试时候不打乱。
        drop_last=opt.dataset.drop_last, # 丢掉最后一个不符合batch数的样本，避免 batch size 不一致引发 BN 问题。
        pin_memory=opt.dataset.pin_memory, # 加快 CPU→GPU 的数据拷贝速度。
    )
    val_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        mask_root=opt.dataset.mask_root,
        flow_root=opt.dataset.flow_root,
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.training.batch_size,
        num_workers=opt.dataset.num_workers,
        shuffle=opt.dataset.shuffle, # 是否打乱，一般训练时候打乱，测试时候不打乱。
        drop_last=opt.dataset.drop_last, # 丢掉最后一个不符合batch数的样本，避免 batch size 不一致引发 BN 问题。
        pin_memory=opt.dataset.pin_memory, # 加快 CPU→GPU 的数据拷贝速度。
    )
    # Custom System
    system = InteractionVideoSystem(opt)
    # Custom Logger
    wandb_logger = WandbLogger(
        project=opt.experiment_project,       # 项目名称（wandb 项目中显示）
        name=opt.experiment_name,             # 当前实验名（可选）
        save_dir=opt.experiment_path,         # 日志保存路径（本地）
        log_model=False,                      # 是否保存模型 checkpoint 到 wandb
        offline=False,                         # 离线模式
    )
    # Define Trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        # logger=False,
        max_epochs=opt.training.max_epochs,
        precision=opt.training.precision,
        limit_train_batches=1.0, # 限制训练数据占比，用来debug
        limit_val_batches=2,  # 只用 2 个 batch 做验证
        num_sanity_val_steps=1, #  训练前，val_dataloader() 中取 2 个 batch，执行 validation_step() 进行“预验证”
        accumulate_grad_batches=opt.training.accumulate_grad_batches, # 梯度累积
        gradient_clip_val=opt.training.gradient_clip_val, # 梯度裁剪
        log_every_n_steps=opt.training.log_every_n_steps, # 多少个step记录一次
        check_val_every_n_epoch=3, # 多少个epoch验证一次
        accelerator=opt.training.accelerator, # 
        # strategy=opt.training.strategy, # or 'ddp_find_unused_parameters_true' optioanl [deepspeed]
        strategy='deepspeed', # 注意开多卡部分无法breakpoint debug
        benchmark=opt.training.benchmark,
        enable_checkpointing=False, # 关闭系统自带的保存策略
        callbacks=[
            CustomProgressBar(), # 自定义显示条
            CustomModelCheckpoint(save_dir=opt.save_path,
                                    save_every_n_steps=1, # 每几轮保存一次
                                ),
        ],
        num_nodes=opt.num_nodes, # 多机训练，节点数量
    )
    # 设置resume_epoch
    trainer.fit_loop.epoch_progress.current.completed = system.resume_epoch
    # Start training!!!
    trainer.fit(system, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


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
    )
    # ----------------------------------------------------------------------
    main(opt)
