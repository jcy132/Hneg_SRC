#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_hSRC.py --mode eval --num_domains 4 --w_hpf 0 \
--checkpoint_dir ./expr/checkpoints_seasons --eval_dir ./expr/eval_seasons \
--sample_dir ./expr/samples_seasons \
--train_img_dir ./data/seasons/train \
--val_img_dir ./data/seasons/val --result_dir ./expr/results_seasons --resume_iter 0

# CUDA_VISIBLE_DEVICES=0 python main_hSRC.py --mode eval --num_domains 5 --w_hpf 0 \
# --checkpoint_dir ./expr/checkpoints_weather --eval_dir ./expr/eval_weather \
# --sample_dir ./expr/samples_weather \
# --train_img_dir ./data/weather/train \
# --val_img_dir ./data/weather/train --result_dir ./expr/results_weather --resume_iter 0