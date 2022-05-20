python export.py \
  --input_path logs/cycle_gan_h2z/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/cycle_gan_h2z/compressed/latest_net_G.pth \
  --ngf 64 --config_str $1
  
#16_16_32_16_32_64_16_16