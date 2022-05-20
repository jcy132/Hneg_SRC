#!/usr/bin/env bash
python train_supernet.py --dataroot database/maps \
  --supernet resnet \
  --log_dir logs/pix2pix_map2sat/finetune \
  --teacher_ngf 64 --student_ngf 64 --teacher_netG resnet_9blocks \
  --nepochs 200 --nepochs_decay 200 \
  --save_epoch_freq 50 --save_latest_freq 20000 \
  --eval_batch_size 16 \
  --restore_teacher_G_path pretrained/map2sat/full/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix_map2sat/supernet/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix_map2sat/supernet/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/maps_A.npz \
  --direction BtoA --config_str $1 \
  --lambda_recon 10 --lambda_distill 0.01
