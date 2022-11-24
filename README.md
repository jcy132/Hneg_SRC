## Official Pytorch implementation of "Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks" (CVPR 2022)
[Chanyong Jung*](https://sites.google.com/view/jcy132), [Gihyun Kwon*](https://sites.google.com/view/gihyunkwon), [Jong Chul Ye](https://bispl.weebly.com/professor.html) (* co-first author)

Link: https://arxiv.org/abs/2203.01532

Supplementary Material:
https://openaccess.thecvf.com/content/CVPR2022/supplemental/Jung_Exploring_Patch-Wise_Semantic_CVPR_2022_supplemental.pdf



<p align="center">
<img src="https://user-images.githubusercontent.com/52989204/163761652-cc999aa5-db8f-4e34-be4e-8fa4fa706c2e.png" width="900"/>
</p> 

![Result_sincut](https://user-images.githubusercontent.com/52989204/165891864-1f7bbb8f-937e-496e-86e1-7ba9ebc48001.jpg)


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


