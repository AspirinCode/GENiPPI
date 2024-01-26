[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](https://github.com/AspirinCode/GENiPPI)
[![bioRxiv](https://img.shields.io/badge/bioRxiv%202023.10.10.557742-green)](https://doi.org/10.1101/2023.10.10.557742)

# GENiPPI

**Interface-aware molecular generative framework for protein-protein interaction modulators**

Protein-protein interactions (PPIs) play a crucial role in many biochemical processes and biological processes. Recently, many structure-based molecular generative models have been proposed. However, PPI sites and compounds targeting PPIs have distinguished physicochemical properties compared to traditional binding pockets and drugs, it is still a challenging task to generate compounds targeting PPIs by considering PPI complexes or interface hotspot residues. In this work, we propose a specifically molecular generative framework based on PPI interfaces, named GENiPPI. We evaluated the framework and found it can capture the implicit relationship between the PPI interface and the active molecules, and can generate novel compounds that target the PPI interface. Furthermore, the framework is able to generate diverse novel compounds with limited PPI interface inhibitors. The results show that PPI interface-based molecular generative model enriches structure-based molecular generative models and facilitates the design of modulators based on PPI structures.


## Framework of GENiPPI
![Model Architecture of GENiPPI](https://github.com/AspirinCode/GENiPPI/blob/latest_branch/figure/GENiPPI_framework.jpg)


## Acknowledgements
The code in this repository is based on their source code release (https://github.com/AspirinCode/iPPIGAN and https://github.com/kiharalab/GNN_DOVE). If you find this code useful, please consider citing their work.

## Requirements
```python
Python==3.6
pytorch==1.7.1
torchvision==0.8.2
tensorflow==2.5
keras==2.2.2
RDKit==2020.09.1.0
HTMD==1.13.9
Multiwfn==3.7
moleculekit==0.6.7
```

https://github.com/rdkit/rdkit

https://github.com/Acellera/htmd

https://github.com/Acellera/moleculekit

http://sobereva.com/multiwfn/


## Training


```python

#the training model
# 0 : train
python train.py [File Index] 0

#example
python train.py 1 0
python train.py 2 0
...

#fine-tuning
# 1 : fine tuning
python train.py [File Index] 1

#example
python train.py 2 1
python train.py 3 1
python train.py 4 1
...
```

For the generation stage the model files are available. It is possible to use the ones that are generated during the training step or you can download the ones that we have already generated model files from Google Drive. 



## Generation
novel compound generation please follow notebook:

```python
python gen_wgan.py

or

GENiPPI_generate.ipynb
```

## Model Metrics
### MOSES
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI

*  Kosugi T, Ohue M. Quantitative estimate index for early-stage screening of compounds targeting protein-protein interactions. International Journal of Molecular Sciences, 22(20): 10925, 2021. doi: 10.3390/ijms222010925
Another QEPPI publication (conference paper)

*  Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. In Proceedings of The 18th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2021), 2021. doi: 10.1109/CIBCB49929.2021.9562931 (PDF) * © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


## Cite:

*  Under Review

*  Jianmin Wang, Jiashun Mao, Chunyan Li, Hongxin Xiang, Xun Wang, Shuang Wang, Zixu Wang, Yangyang Chen, Yuquan Li, Heqi Sun, Kyoung Tai No, Tao Song, Xiangxiang Zeng; An interface-based molecular generative framework for protein-protein interaction inhibitors. bioRxiv 2023.10.10.557742; doi: https://doi.org/10.1101/2023.10.10.557742

*  Jianmin Wang, Yanyi Chu, Jiashun Mao, Hyeon-Nae Jeon, Haiyan Jin, Amir Zeb, Yuil Jang, Kwang-Hwi Cho, Tao Song, Kyoung Tai No, De novo molecular design with deep molecular generative models for PPI inhibitors, Briefings in Bioinformatics, 2022;, bbac285, https://doi.org/10.1093/bib/bbac285

