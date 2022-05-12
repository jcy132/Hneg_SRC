#!/usr/bin/env bash

python main_hSRC.py --mode train --num_domains 4 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 --lambda_src 0.1 --lambda_nce 0.0 --lambda_dce 1.0 \
--checkpoint_dir ./expr/checkpoints_seasons --eval_dir ./expr/eval_seasons \
--sample_dir ./expr/samples_seasons \
--train_img_dir ./data/seasons/train \
--val_img_dir ./data/seasons/val --result_dir ./expr/results_seasons \
--use_curriculum --n_patch 128 --use_hard

# python main_hSRC.py --mode train --num_domains 4 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 --lambda_src 0.1 --lambda_nce 0.0 --lambda_dce 0.1 \
# --checkpoint_dir ./expr/checkpoints_weather --eval_dir ./expr/eval_weather \
# --sample_dir ./expr/samples_weather \
# --train_img_dir ./data/weather/train \
# --val_img_dir ./data/weather/val --result_dir ./expr/results_weather \
# --use_curriculum --n_patch 128 --use_hard
