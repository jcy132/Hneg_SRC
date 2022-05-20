## Single-modal Image translation

This repository provides Pytorch Implementation of Single-modal Image translation.

Our source code is based on official implementation of [CUT](https://github.com/taesungp/contrastive-unpaired-translation)

### Pre-trained model

We provide the pretrained model for 3 datasets: \
[horse-to-zebra](https://drive.google.com/file/d/11N8KXWSS4m6o-oTeQOO7KahPp7Nd7BaE/view?usp=sharing) /
[Cityscapes](https://drive.google.com/file/d/1oPIyWwLEtBIKaO4vZZVijMHJD7xICBsh/view?usp=sharing) /
[Monet](https://drive.google.com/file/d/1fzkME3D-g8tztdr8rotucbtdxSe7YrIy/view?usp=sharing)

We also provied drn-d-22 model for cityscapes: \
[drn-d-22](https://drive.google.com/file/d/1nNV_tQ4PhEyF6TohgYOaZ51qUYvXoMMB/view?usp=sharing)

To get results of pretrained model, 
1. Locate the pretrained model to the ```./checkpoints``` directory
2. Refer the ```./script/test.sh``` file, or the following command
```
## horse2zebra 
python test.py --dataroot ./datasets/horse2zebra --name h2z-best --CUT_mode CUT --phase test --epoch best
python -m pytorch_fid ./results/h2z-best/test_best/images/fake_B ./datasets/horse2zebra/testB

## cityscapes
python test.py --dataroot ./datasets/cityscapes --name city-best --CUT_mode CUT --phase test --epoch best --direction BtoA
python -m pytorch_fid ./results/city-best/test_best/images/fake_B ./datasets/cityscapes/testA

## Monet 
python test.py --name monet-best --model sincut --dataroot ./datasets/single_image_monet_etretat
```


### Training & Evaluation
#### Training 
Refer the ```./script/train.sh``` file, or the following command
```
### horse2zebra
python train.py --name exp_h2z \
--dataroot ./datasets/horse2zebra \
--CUT_mode CUT --dce_idt --lambda_HDCE 0.1 --lambda_SRC 0.05 \
--use_curriculum --HDCE_gamma 50 --HDCE_gamma_min 10 \
--gpu_ids 0


### cityscapes
python train.py --name exp_city \
--dataroot ./datasets/cityscapes_cut \
--CUT_mode CUT --dce_idt --lambda_HDCE 0.1 --lambda_SRC 0.1 --direction BtoA \
--use_curriculum --HDCE_gamma 50 --HDCE_gamma_min 10 --step_gamma --step_gamma_epoch 200 \
--gpu_ids 0

### Monet
python train.py --name exp_monet \
--dataroot ./datasets/single_image_monet_etretat \
--CUT_mode CUT --dce_idt --model sincut --lambda_HDCE 4.0 --lambda_SRC 10.0 \
--use_curriculum --HDCE_gamma 200 --HDCE_gamma_min 40 \
--gpu_ids 0
```

#### Evaluation
Please refer the following command:
```
python test.py --dataroot [path-to-dataset] --name [experiment-name] --CUT_mode CUT --phase test --epoch [epoch-for-test]
python -m pytorch_fid [path-to-output] [path-to-input]
```

For the segmentation scores, please refer the following command:
```
python python get_mAP.py test -d [path-to-result-folder] \
-c 19 --arch drn_d_22 --phase val --batch-size 1 --pretrained [path-to-pretrained-drn]

python get_Acc.py \
--gt_path [path-to-result-folder]/images/real_A --output_path [path-to-result-folder]/segmapResult_color
```

### Acknowledgement
Our source code is based on [CUT](https://github.com/taesungp/contrastive-unpaired-translation). \
We thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID calculation, [drn](https://github.com/fyu/drn) and [GcGAN](https://github.com/hufu6371/GcGAN) for the computation of segmentation scores. 
