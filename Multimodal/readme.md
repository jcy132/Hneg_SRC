## Multi domain - Multi modal Image translation

This repository provides Pytorch Implementation of Multidomain-Multimodal Image translation.

Our source code is based on official implementation of [StarGANv2](https://github.com/clovaai/stargan-v2)

### Dataset

For dataset download, please refer to the following link 

[Seasons](https://github.com/AAnoosheh/ComboGAN) / [Weather](https://ieee-dataport.org/documents/five-class-weather-image-dataset)

Before traning / evaluation, split the data into train & validation set,

then save the data in ```data``` folder.

To access the train&val dataset that we used, contact : cyclomon@kaist.ac.kr 

### Pre-trained model download

To download our pre-trained model, please refer to the following link

[Pre-trained Seasons](https://drive.google.com/file/d/1885ZJk4wFI5UKWW_o-2seF8vkMu1jXGt/view?usp=sharing) / [Pre-trained Weather](https://drive.google.com/file/d/106Y5ssK5WickKsttPYhZgx8OtYYSkLOy/view?usp=sharing)

### Training 

For training, use ```bash run_train.sh``` or use the command

or run the command for training models on Seasons dataset

```
python main_hSRC.py --mode train --num_domains 4 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
--lambda_src 0.1 --lambda_nce 0.0 --lambda_dce 1.0 \
--checkpoint_dir ./expr/checkpoints_seasons --eval_dir ./expr/eval_seasons \
--sample_dir ./expr/samples_seasons \
--train_img_dir ./data/alps_proc/train \
--val_img_dir ./data/alps_proc/val --result_dir ./expr/results_seasons \
--use_curriculum --n_patch 128 --use_hard
```

for Weather dataset

```
python main_hSRC.py --mode train --num_domains 5 --w_hpf 0 --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
--lambda_src 0.1 --lambda_nce 0.0 --lambda_dce 0.1 \
--checkpoint_dir ./expr/checkpoints_seasons --eval_dir ./expr/eval_seasons \
--sample_dir ./expr/samples_seasons \
--train_img_dir ./data/alps_proc/train \
--val_img_dir ./data/alps_proc/val --result_dir ./expr/results_seasons \
--use_curriculum --n_patch 128 --use_hard
```

### Evaluation

For evaluation, use ```bash run_eval.sh``` or use the command

```
CUDA_VISIBLE_DEVICES=0 python main_hSRC.py --mode eval --num_domains 4 --w_hpf 0 \
--checkpoint_dir ./expr/checkpoints_seasons --eval_dir ./expr/eval_seasons \
--sample_dir ./expr/samples_seasons \
--train_img_dir ./data/seasons/train \
--val_img_dir ./data/seasons/val --result_dir ./expr/results_seasons --resume_iter 0
```

or 

```
CUDA_VISIBLE_DEVICES=0 python main_hSRC.py --mode eval --num_domains 5 --w_hpf 0 \
--checkpoint_dir ./expr/checkpoints_weather --eval_dir ./expr/eval_weather \
--sample_dir ./expr/samples_weather \
--train_img_dir ./data/weather/train \
--val_img_dir ./data/weather/train --result_dir ./expr/results_weather --resume_iter 0
```
