# Publication:  "Protein Docking Model Evaluation by Graph Neural Networks", Xiao Wang, Sean T Flannery and Daisuke Kihara,  (2020)

#GNN-Dove is a computational tool using graph neural network that can evaluate the quality of docking protein-complexes.

#Copyright (C) 2020 Xiao Wang, Sean T Flannery, Daisuke Kihara, and Purdue University.

#License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)

#Contact: Daisuke Kihara (dkihara@purdue.edu)

#

# This program is free software: you can redistribute it and/or modify

# it under the terms of the GNU General Public License as published by

# the Free Software Foundation, version 3.

#

# This program is distributed in the hope that it will be useful,

# but WITHOUT ANY WARRANTY; without even the implied warranty of

# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

# GNU General Public License V3 for more details.

#

# You should have received a copy of the GNU v3.0 General Public License

# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.
import sys
sys.path.append('/home/jmwang/WorkSpace/GENiPPI/gPPMol')
import torch
import os
import numpy as np
from math import sqrt  
from ppi_processing.Prepare_Input import Prepare_Input

# 质数判断  
def isPrime(n):  
    for i in range(2, int(sqrt(n))+1):  
        if n % i == 0:  
            return False  
    return True  
  
def load_interface_feature(structure_path="./Input.pdb"):
    input_file=Prepare_Input(structure_path=structure_path)
    data=np.load(input_file)

    H=data['H']
    A1=data['A1']
    A2 = data['A2']
    V=data['V']
    #print(H,A1,A2,V,H.shape,A1.shape,A2.shape,V.shape)  #(402, 56) (402, 402) (402, 402) (402,)
    A2 = A2.reshape(-1)
    
    index = 2  
    num = A2.shape[0]
    maxPrime = None  
  
    while index <= num:  
        if isPrime(index) and num % index == 0:  
            num /= index  
            maxPrime = index  
        index += 1  
  
    print (maxPrime) #67

    A2 = A2.reshape(-1,maxPrime)

    np.savez('feature_reshape',  H=H, A1=A1, A2=A2, V=V)

    print("A2:",A2.shape) #(2412, 67)

    print("V:",V.shape)

    H = torch.Tensor( np.reshape(H, (1, H.shape[0], H.shape[1])) )
    A2 = torch.Tensor( np.reshape(A2, (1, A2.shape[0], A2.shape[1])) )
    V = torch.Tensor( np.reshape(V, (1, V.shape[0])) )

    return H,A1,A2,V,A2.shape[-1]

if __name__ == "__main__":

    load_interface_feature()
    
