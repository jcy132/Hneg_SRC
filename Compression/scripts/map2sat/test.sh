#!/usr/bin/env bash
python test.py --dataroot database/maps \
  --results_dir results-pretrained/pix2pix/map2sat_fast/finetune_map \
  --restore_G_path logs/pix2pix_map2sat/compressed/latest_net_G.pth \
  --config_str $1 \
  --real_stat_path real_stat/maps_A.npz \
  --direction BtoA \
  --need_profile --num_test 200

#16_16_48_32_32_40_24_16