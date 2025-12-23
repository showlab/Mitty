# Mitty
> **Mitty: Diffusion-based Human-to-Robot Video Generation**
> <br>
> [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), 
> [Cheng Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=TvdVuAYAAAAJ), 
> [Weijia Mao](https://scholar.google.com.hk/citations?hl=zh-CN&user=TvdVuAYAAAAJ), 
> and 
> [Mike Zheng Shou](https://scholar.google.com/citations?user=S7bGBmkyNtEC)
> <br>
> [Show Lab](https://sites.google.com/view/showlab), National University of Singapore
> <br>

<a href="https://arxiv.org/abs/xxx"><img src="https://img.shields.io/badge/ariXv-xxxx.xxxx-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/spaces/xxxx/xxxx"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/xxxx/xxxx"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/xxxx/xxxx"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/xxxx/xxxx/"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>

<br>

<img src='./assets/teaser.png' width='100%' />


## 🔧 Environment & Installation

### 1. Create environment

```bash
conda create -n mitty python=3.10 -y
conda activate mitty
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 HuggingFace Models & Datasets

### 1. Download Pretrained Model

The fine-tuned Mitty model is available at: [showlab/Mitty_Model](https://huggingface.co/showlab/Mitty_Model)

Download to `ckpt/` directory:

```bash
mkdir -p ckpt
huggingface-cli download showlab/Mitty_Model --local-dir ckpt/Mitty_Model
```

**Note**: This codebase uses the Diffusers-integrated model format (e.g., `Wan2.2-TI2V-5B-Diffusers`) which includes proper subfolder structure (tokenizer/, text_encoder/, vae/, transformer/, scheduler/).

### 2. Download and Prepare Dataset

The paired human–robot dataset is available at: [showlab/Mitty_Dataset](https://huggingface.co/datasets/showlab/Mitty_Dataset)

Download and extract to `dataset/` directory:

```bash
# Download dataset
mkdir -p dataset
huggingface-cli download showlab/Mitty_Dataset --repo-type dataset --local-dir dataset/tmp

# Extract dataset
cd dataset
unzip -q tmp/Human2Robot.zip
unzip -q tmp/EPIC-KITCHENS.zip
rm -rf tmp
cd ..
```

The dataset will be organized as:

```text
dataset/
  ├── Human2Robot/
  │   ├── human/
  │   │   ├── xxx_00001.mp4
  │   │   ├── xxx_00001.txt # prompt
  │   │   └── ...
  │   └── robot/
  │       ├── xxx_00001.mp4
  │       └── ...
  └── EPIC-KITCHENS/
      ├── human/
      │   ├── xxx_00001.mp4
      │   └── ...
      └── robot/
          ├── xxx_00001.mp4
          └── ...
```

---

## 🚀 Training & 🎬 Inference

We provide simple shell scripts to launch training and inference.

### 1. Training

Edit `_scripts/train.sh` to set your dataset paths, output directory, and training hyperparameters.  
Then run:

```bash
_scripts/train.sh
```

This will start training Mitty on the paired human–robot dataset.

### 2. Inference

Edit `_scripts/inference.sh` to point to your trained checkpoint (or the released pretrained model) and specify the input human video / prompt and output directory.  
Then run:

```bash
_scripts/inference.sh
```

This will generate corresponding robot videos from the human inputs using the Mitty model.



---

## 📚 Citation

If you use this codebase or the released models / dataset in your research, please cite the Mitty paper.  

```bibtex
@article{mitty2025,
    title  = {Mitty: Diffusion-based Human-to-Robot Video Generation},
    author = {Yiren Song and Cheng Liu and Weijia Mao and Mike Zheng Shou},
    year   = {2025},
}
```

