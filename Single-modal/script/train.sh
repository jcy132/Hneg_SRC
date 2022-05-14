set -ex

##########
# Exp
##########

### horse2zebra
#python train.py --name exp_h2z \
#--dataroot ./datasets/horse2zebra \
#--CUT_mode CUT --dce_idt --lambda_HDCE 0.1 --lambda_SRC 0.05 \
#--use_curriculum --HDCE_gamma 50 --HDCE_gamma_min 10 \
#--gpu_ids 0


### cityscapes
#python train.py --name exp_city \
#--dataroot ./datasets/cityscapes_cut \
#--CUT_mode CUT --dce_idt --lambda_HDCE 0.1 --lambda_SRC 0.1 --direction BtoA \
#--use_curriculum --HDCE_gamma 50 --HDCE_gamma_min 10 --step_gamma --step_gamma_epoch 200 \
#--gpu_ids 0

### Monet
#python train.py --name exp_monet \
#--dataroot ./datasets/single_image_monet_etretat \
#--CUT_mode CUT --dce_idt --model sincut --lambda_HDCE 4.0 --lambda_SRC 10.0 \
#--use_curriculum --HDCE_gamma 200 --HDCE_gamma_min 40 \
#--gpu_ids 0


