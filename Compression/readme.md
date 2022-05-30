## GAN Compression Code

Our implementation is based on official repository of [Fast GAN Compression](https://github.com/mit-han-lab/gan-compression). 

### Additional environment setting
```pip install torchprofile```
```pip install tensorboard```

### Data prepare

You can download the training dataset & data stats for metric calculation.

e.g) For horse2zebra, download the image dataset with following command

```
bash datasets/download_cyclegan_dataset.sh horse2zebra
```

For calculation of FID score, download the real stat dataset with following command

```
bash datasets/download_real_stat.sh horse2zebra A

bash datasets/download_real_stat.sh horse2zebra B
```

For the Cityscapes dataset, we cannot provide the full dataset due to license issue. 

Please download the dataset from https://cityscapes-dataset.com and use the script prepare_cityscapes_dataset.py to preprocess it. 

You need to download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip and unzip them in the same folder. 

For example, you may put gtFine and leftImg8bit in database/cityscapes_origin. You need to prepare the dataset with the following commands:

```
python datasets/get_trainIds.py database/cityscapes-origin/gtFine/
python datasets/prepare_cityscapes_dataset.py \
--gtFine_dir database/cityscapes-origin/gtFine \
--leftImg8bit_dir database/cityscapes-origin/leftImg8bit \
--output_dir database/cityscapes \
--train_table_path datasets/train_table.txt \
--val_table_path datasets/val_table.txt
```

To support mIoU computation, you need to download a pre-trained DRN model ```drn-d-105_ms_cityscapes.pth``` from http://go.yf.io/drn-cityscapes-models. 

By default, we put the drn model in the root directory of the repo. 

Then you can test our compressed models on cityscapes after you have downloaded our models.

### Download Teacher Model

We provide pre-trained teacher models in the following links.

[CycleGAN-horse2zebra](https://drive.google.com/file/d/1Y7QAySP1ZC4WqZbszl9FKs0nVyFaqyhZ/view?usp=sharing)

[Pix2Pix-map2sat](https://drive.google.com/file/d/1EfV0goGJB_koozqQAWyR87qNjdOXwxpD/view?usp=sharing)

[Pix2Pix-cityscapes](https://drive.google.com/file/d/1eCd6NPPNGOacqjaiE8HbXZ4NbFpYyzbg/view?usp=sharing)

Please save the models in directory

```
pretrained/horse2zebra/full
pretrained/map2sat/full
pretrained/cityscapes/full
```

### Supernet Training

For training supernet(Once-for-all model) distillation with our code, please run the following command.

e.g) For horse2zebra cycleGAN,

```bash scripts/h2z/train_supernet.sh```

### Evolution Search

For searching the optimal model architecture with evolution search, 

please run the following bash file

e.g.) For horse2zebra cycleGAN,

```bash scripts/h2z/evolution_search.sh```

### (Optional) Fine-tuning the compressed model

For better performance of compressed model, we can fine-tune the model with additional training

To reproduce the results on pix2pix-map2sat, please run the fine-tuning step.

```bash scripts/map2sat/finetune.sh $CONFIG```

Again, in ```$CONFIG```, please write the channel configuration e.g. ```16_16_32_16_32_64_16_16```

### Compressed model export

After evolution search step, we can obtain the optimized channel configuration for compressed model.

Please run the following bash file to obtain compressed model.

```bash scripts/h2z/export.sh $CONFIG```

in ```$CONFIG```, please write the channel configuration e.g. ```16_16_32_16_32_64_16_16```


### Test

For calculating metrics on compressed model,

please run the following code e.g.)

```bash scripts/h2z/test.sh $CONFIG```

in ```$CONFIG```, please write the channel configuration

We provide the pre-trained compressed models in the follwing link.

we also provide pre-trained network configuration in the bash file.

[Map2sat](https://drive.google.com/file/d/1GQMkUFGHdPorOdZCIkd_BKGLx6qJDOqo/view?usp=sharing)

[Horse2zebra](https://drive.google.com/file/d/1NQs-cwhTZvjAPALMT_rNqH0eJZfNkCaa/view?usp=sharing)

[Cityscapes](https://drive.google.com/file/d/1h3ePQKDpzpFSRp88gYF8Y630bnLBmXHk/view?usp=sharing)



