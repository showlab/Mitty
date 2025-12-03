from src.wan2_trainer import InteractionVideoSystem
import pytorch_lightning as L
import argparse
from omegaconf import OmegaConf
import os
import torch
from datasets.custom_dataset import CustomDataset
from torch.utils.data import DataLoader


def main(opt):
    L.seed_everything(opt.seed)
    test_dataset = CustomDataset(
        video_root=opt.dataset.video_root,
        video_root2=opt.dataset.video_root2,
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last,
        pin_memory=opt.dataset.pin_memory,
        shuffle=False,
    )
    system = InteractionVideoSystem.load_from_checkpoint("/opt/liblibai-models/user-workspace2/users/sqj/code/InteractionVideo/outputs/wan2/use_DiffSynth-1e4-condition-lora-v6/checkpoints/step=1800.ckpt", opt=opt)
    trainer = L.Trainer(
        logger=False,
        precision=opt.training.precision,
        log_every_n_steps=1,
        accelerator=opt.training.accelerator, # 
        strategy=opt.training.strategy,
        benchmark=opt.training.benchmark,
        num_nodes=opt.num_nodes,
    )
    trainer.predict(system, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml", help="path to the yaml config file")
    # Additional paras
    parser.add_argument("--seed", type=int, default=42)
    # ----------------------------------------------------------------------
    args, extras = parser.parse_known_args()
    args = vars(args)
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )
    # ----------------------------------------------------------------------
    main(opt)
