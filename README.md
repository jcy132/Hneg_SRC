# Hneg_SRC
## Official Pytorch implementation of "Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks" (CVPR 2022)
(https://arxiv.org/abs/2203.01532)

[Chanyong Jung*](https://sites.google.com/view/jcy132), [Gihyun Kwon*](https://sites.google.com/view/gihyunkwon), [Jong Chul Ye](https://bispl.weebly.com/professor.html) (* Equally Contributed)

![github](https://user-images.githubusercontent.com/52989204/163760314-97f79169-9405-436f-9fe0-e8f50d3b9f9b.jpg)

### Environment
```
$ conda create -n SRC python=3.6
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ conda install -c conda-forge packaging 
$ conda install -c conda-forge visdom 
$ conda install -c conda-forge gputil 
$ conda install -c conda-forge dominate 
```
