python test.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir logs/cycle_gan_h2z/compressed \
  --config_str $1 \
  --restore_G_path logs/cycle_gan_h2z/compressed/latest_net_G.pth \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz
  
# 16_16_32_16_16_64_16_24