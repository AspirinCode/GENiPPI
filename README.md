[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](https://github.com/AspirinCode/GENiPPI)
[![ J. Cheminform.](https://img.shields.io/badge/%20J%20Cheminform%20(2024)-red)](https://doi.org/10.1186/s13321-024-00930-0)
[![bioRxiv](https://img.shields.io/badge/bioRxiv%202023.10.10.557742-green)](https://doi.org/10.1101/2023.10.10.557742)
[![Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.13968591-gray)](https://zenodo.org/records/13968592)


# GENiPPI

**Interface-aware molecular generative framework for protein-protein interaction modulators**

Protein-protein interactions (PPIs) play a crucial role in numerous biochemical and biological processes. Although several structure-based molecular generative models have been developed, PPI interfaces and compounds targeting PPIs exhibit distinct physicochemical properties compared to traditional binding pockets and small-molecule drugs. As a result, generating compounds that effectively target PPIs, particularly by considering PPI complexes or interface hotspot residues, remains a significant challenge. In this work, we constructed a comprehensive dataset of PPI interfaces with active and inactive compound pairs. Based on this, we propose a novel molecular generative framework tailored to PPI interfaces, named GENiPPI. Our evaluation demonstrates that GENiPPI captures the implicit relationships between the PPI interfaces and the active molecules, and can generate novel compounds that target these interfaces. Moreover, GENiPPI can generate structurally diverse novel compounds with limited PPI interface modulators. To the best of our knowledge, this is the first exploration of a structure-based molecular generative model focused on PPI interfaces, which could facilitate the design of PPI modulators. The PPI interface-based molecular generative model enriches the existing landscape of structure-based (pocket/interface) molecular generative model.


## Framework of GENiPPI
![Model Architecture of GENiPPI](https://github.com/AspirinCode/GENiPPI/blob/main/figure/GENiPPI_framework.png)


## Acknowledgements
The code in this repository is based on their source code release (https://github.com/AspirinCode/iPPIGAN and https://github.com/kiharalab/GNN_DOVE). If you find this code useful, please consider citing their work.


## News!

**[2024/11/20]** Available online **Journal of Cheminformatics**, 2024.  

**[2024/11/11]** Accepted in **Journal of Cheminformatics**, 2024.  

**[2024/03/15]** submission to **Journal of Cheminformatics**, 2024.  

**[2023/10/10]** submission to **bioRxiv**, 2023.  


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


## Installation

```python
conda env create -f environment.yml

OR

conda create --name GENiPPI python=3.6 conda

pip install -r requirements.txt


OR

conda config --add channels acellera
conda install -c acellera htmd=1.13.9

#pytorch==1.7.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
#keras
conda install keras==2.2.2

```



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



## Analysis


### Molecular shape

**calculate NPR and PMI descriptors**

```python
Chem.Descriptors3D.NPR1(mol)
Chem.Descriptors3D.NPR2(mol)

Chem.rdMolDescriptors.CalcPMI1
Chem.rdMolDescriptors.CalcPMI2
Chem.rdMolDescriptors.CalcPMI3
```

https://greglandrum.github.io/rdkit-blog/posts/2022-06-22-variability-of-pmi-descriptors.html


**calculate PBF descriptors**

```python
Chem.rdMolDescriptors.CalcPBF(mol)
```

**reference**  
Firth, N.C., Brown, N. and Blagg, J., 2012. Plane of best fit: a novel method to characterize the three-dimensionality of molecules. Journal of chemical information and modeling, 52(10), pp.2516-2525.


### TMAP visualization of chemical space

https://github.com/reymond-group/mhfp

https://github.com/reymond-group/faerun-python



```python

pip install mhfp
pip install faerun

```

**reference code**  
https://tmap.gdb.tools/?ref=gdb.unibe.ch#ex-chembl


## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


## Cite:


*  Jianmin Wang, Jiashun Mao, Chunyan Li, Hongxin Xiang, Xun Wang, Shuang Wang, Zixu Wang, Yangyang Chen, Yuquan Li, Kyoung Tai No, Tao Song, Xiangxiang Zeng; Interface-aware molecular generative framework for protein-protein interaction modulators.  J Cheminform (2024). doi: https://doi.org/10.1186/s13321-024-00930-0

*  Jianmin Wang, Yanyi Chu, Jiashun Mao, Hyeon-Nae Jeon, Haiyan Jin, Amir Zeb, Yuil Jang, Kwang-Hwi Cho, Tao Song, Kyoung Tai No, De novo molecular design with deep molecular generative models for PPI inhibitors, Briefings in Bioinformatics, 2022;, bbac285, https://doi.org/10.1093/bib/bbac285

* J. Wang, P. Zhou, Z. Wang, W. Long, Y. Chen, K.T. No, D. Ouyang, J. Mao, X. Zeng, Diffusion-based generative drug-like molecular editing with chemical natural language, Journal of Pharmaceutical Analysis, https://doi.org/10.1016/j.jpha.2024.101137.  
