> **Mitty: Diffusion-based Human-to-Robot Video Generation**
> <br>
> [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), 
> [Cheng Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=TvdVuAYAAAAJ), 
> and 
> [Mike Zheng Shou](https://sites.google.com/view/showlab)
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

### 1. Pretrained model

The pretrained / fine-tuned Mitty H2R model will be available at:

- **Model:**
  - `https://huggingface.co/<YOUR_HF_USERNAME>/mitty-h2r`

Example usage with a simple `MittyPipeline` wrapper (adjust to match your actual code):

```python
import torch
from mitty.engine.inference import MittyPipeline

pipe = MittyPipeline.from_pretrained(
    "<YOUR_HF_USERNAME>/mitty-h2r",
    torch_dtype=torch.float16,
).to("cuda")

robot_video = pipe(
    human_video_path="examples/human_demo.mp4",
    mode="h2r",  # or "hi2r"
    robot_first_frame="examples/robot_init.png",  # required when mode == "hi2r"
)
robot_video.save("outputs/robot_demo.mp4")
```

### 2. Dataset

The paired human–robot dataset will be released as a HuggingFace dataset:

- **Dataset:**
  - `https://huggingface.co/datasets/<YOUR_HF_USERNAME>/mitty-h2r-dataset`

A recommended format is:

```text
dataset/
  ├── human/
  │   ├── xxx_00001.mp4
  │   ├── xxx_00001.txt # prompt
  │   └── ...
  ├── robot/
  │   ├── xxx_00001.mp4
  │   └── ...
```

---

## 🚀 Training



---

## 🎬 Inference



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

