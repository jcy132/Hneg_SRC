python test.py --dataroot ./database/cityscapes \
  --results_dir results-pretrained/pix2pix_city/compressed \
  --config_str 16_48_48_24_40_48_24_24 \
  --restore_G_path logs/pix2pix_city/compressed/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path /database/cityscape_origin \
  --table_path datasets/val_table.txt \
  --direction BtoA --need_profile