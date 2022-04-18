# Hneg_SRC
## Official Pytorch implementation of "Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks" (CVPR 2022)
(https://arxiv.org/abs/2203.01532)

[Chanyong Jung*](https://sites.google.com/view/jcy132), [Gihyun Kwon*](https://sites.google.com/view/gihyunkwon), [Jong Chul Ye](https://bispl.weebly.com/professor.html) (* Equally Contributed)


<p align="center">
<img src="https://user-images.githubusercontent.com/52989204/163760524-55805b04-2324-421c-8ede-bfbd328cd930.png" width="800"/>
</p> 

### Cite
```
@article{jung2022exploring,
  title={Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks},
  author={Jung, Chanyong and Kwon, Gihyun and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2203.01532},
  year={2022}
}
```

### Environment
```
$ conda create -n SRC python=3.6
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ conda install -c conda-forge packaging 
$ conda install -c conda-forge visdom 
$ conda install -c conda-forge gputil 
$ conda install -c conda-forge dominate 
```
