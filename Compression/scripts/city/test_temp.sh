python test.py --dataroot ../GAN_Compression/database/cityscapes \
  --results_dir logs/results/compressed_e2/result \
  --config_str 40_16_48_16_48_48_16_16 \
  --restore_G_path logs/pix2pix_cityscapes/compressed/latest_net_G.pth \
  --real_stat_path ../GAN_Compression/real_stat/cityscapes_A.npz \
  --drn_path ../GAN_Compression/drn-d-105_ms_cityscapes.pth \
  --cityscapes_path mnt/F/cityscape \
  --table_path ../GAN_Compression/datasets/val_table.txt \
  --direction BtoA --need_profile