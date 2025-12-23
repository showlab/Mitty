export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python src/wan2_trainer_plus.py --config=./_configs/train.yaml --seed=1234
