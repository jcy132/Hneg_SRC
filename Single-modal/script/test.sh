set -ex


###############
# Exp
###############

### pretrained ####

## horse2zebra best
#python test.py --dataroot ./datasets/horse2zebra --name h2z-best --CUT_mode CUT --phase test --num_test 120 --epoch best
#python -m pytorch_fid ./results/h2z-best/test_best/images/fake_B ./datasets/horse2zebra/testB --device cuda:1

## cityscapes best
#python test.py --dataroot ./datasets/cityscapes_cut --name city-best --CUT_mode CUT --phase test --num_test 500 --epoch best --direction BtoA
#python -m pytorch_fid ./results/city-best/test_best/images/fake_B ./datasets/cityscapes_cut/testA

## Style best
#python test.py --name monet-best --model sincut --dataroot ./datasets/single_image_monet_etretat

