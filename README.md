# GENiPPI
Molecular generative framework for protein-protein interface inhibitors

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

http://sobereva.com/multiwfn/

## Training


```python

#the training model
python train.py

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
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI

Kosugi T, Ohue M. Quantitative estimate index for early-stage screening of compounds targeting protein-protein interactions. International Journal of Molecular Sciences, 22(20): 10925, 2021. doi: 10.3390/ijms222010925
Another QEPPI publication (conference paper)

Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. In Proceedings of The 18th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2021), 2021. doi: 10.1109/CIBCB49929.2021.9562931 (PDF) * © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


## Cite:




