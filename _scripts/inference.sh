export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python src/wan2_inference_plus.py --config=./configs/inference.yaml --seed=1234 --ckpt_path PATH_TO_CHECKPOINT