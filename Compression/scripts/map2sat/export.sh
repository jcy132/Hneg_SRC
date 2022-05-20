#!/usr/bin/env bash
python export.py \
  --input_path logs/pix2pix_map2sat/finetune/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix_map2sat/compressed/latest_net_G.pth \
  --ngf 64 --config_str $1
  
#16_16_48_32_32_40_24_16