## Single-modal Image translation

This repository provides Pytorch Implementation of Single-modal Image translation.

Our source code is based on official implementation of [CUT](https://github.com/taesungp/contrastive-unpaired-translation)

### Pre-trained model download & evaluate

We provide the pretrained model for 3 datasets: \
[horse-to-zebra](https://drive.google.com/file/d/11N8KXWSS4m6o-oTeQOO7KahPp7Nd7BaE/view?usp=sharing) /
[Cityscapes](https://drive.google.com/file/d/1oPIyWwLEtBIKaO4vZZVijMHJD7xICBsh/view?usp=sharing) /
[Monet](https://drive.google.com/file/d/1fzkME3D-g8tztdr8rotucbtdxSe7YrIy/view?usp=sharing)

To get results of pretrained model, 
1. Locate the pretrained model to the ```./checkpoints``` directory
2. Refer the ```./script/test.sh``` file, or the following command
```
## horse2zebra best
python test.py --dataroot ./datasets/horse2zebra --name h2z-best --CUT_mode CUT --phase test --epoch best
python -m pytorch_fid ./results/h2z-best/test_best/images/fake_B ./datasets/horse2zebra/testB

## cityscapes best
python test.py --dataroot ./datasets/cityscapes_cut --name city-best --CUT_mode CUT --phase test --epoch best --direction BtoA
python -m pytorch_fid ./results/city-best/test_best/images/fake_B ./datasets/cityscapes_cut/testA

## Style best
python test.py --name monet-best --model sincut --dataroot ./datasets/single_image_monet_etretat
```
