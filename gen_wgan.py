# Copyright (C) 2024 by mao jiashun and wang jianming
# Copying and distribution is allowed under AGPLv3 license

import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np
import pandas as pd
import sys
sys.path.append(r'/home/jmwang/WorkSpace/GENiPPI/')
from gPPMol.nets import EncoderCNN, DecoderRNN, GNN_Model

from gPPMol.utils import *
from tqdm import tqdm
import argparse
import multiprocessing
from gPPMol.ppi_processing.collate_fn import collate_fn
from gPPMol.ppi_processing.Prepare_Input import Prepare_Input
from gPPMol.ppi_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader

from gPPMol.models_3d.wgan_clipping import WGAN_CP
from gPPMol.models_3d.wgan_gradient_penalty import WGAN_GP

batch_size = 8
savedir = "model"
os.makedirs(savedir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)
is_cuda = True if device == "cuda" else False

structure_path="./interface.pdb"
input_file=Prepare_Input(structure_path)
list_npz = [input_file]
dataset = Single_Dataset(list_npz)
dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=1,
                            drop_last=False, collate_fn=collate_fn)
for batch_idx, sample in enumerate(dataloader):
    H, A1, A2, V, Atom_count = sample
    break

smiles = np.load("./gPPMol/example.npy") #(305584, 242)   105
y = np.load("./gPPMol/example_y.npy") #(305584,)

from gPPMol.comgen import decode_smiles
from gPPMol.gene import queue_datagen

multiproc = multiprocessing.Pool(1)
my_gen = queue_datagen(smiles, y, batch_size=batch_size, mp_pool=multiproc)
test_loader  = my_gen

from gPPMol.comgen import decode_smiles

# Define the networks
params={}
params['n_graph_layer'] =4  #图卷积的层数
params['d_graph_layer'] =140  #图卷积每一层有多少个节点
params['n_FC_layer'] =4    #全连接层数
params['d_FC_layer'] =128    #全连接层每一层多少个节点
params['dropout_rate'] =0.3
params['initial_mu'] =0.0
params['initial_dev'] =1.0
params['N_atom_features'] = 28     #每个原子有28个特征值  不可修改
params['final_dimension'] = 4* 4* 4    #将 interface feature map to a 128 size of vector  can be changed
gnn_interface_model = GNN_Model(params)

encoder = EncoderCNN(9)
decoder = DecoderRNN(512, 1024, 37, 1)

gnn_interface_model.to(device)
encoder.to(device)
decoder.to(device)

model_name = 'WGAN-CP'
channels = 9
generator_iters = 50000
batch_size = 8

if model_name == 'WGAN-CP':
    model = WGAN_CP(channels,is_cuda,generator_iters,gnn_interface_model,encoder,decoder,device,batch_size=8)
elif model_name == 'WGAN-GP':
    model = WGAN_GP(channels,is_cuda,generator_iters,gnn_interface_model,encoder,decoder,device,batch_size=8)        
else:
    print("Model type non-existing. Try again.")
    exit(-1)


load_D = './discriminator.pkl'
load_G = './generator.pkl'

# Start model training
#captions1,captions2 = model.evaluate( test_loader,load_D, load_G,H, A1, A2, V, Atom_count)
#aa = decode_smiles(captions1,captions2)

from rdkit import Chem

def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [x for x in set(xresults)]  # Check for duplicates and filter out invalids

#aa=filter_unique_canonical(aa)
#print(aa,len(aa))

#df = pd.DataFrame(aa, columns=['smile']) 

#df.to_csv("predict_result.csv",index=None)
# Start model generating
for i in range(1):
    captions1,captions2 = model.evaluate( test_loader,load_D, load_G,H, A1, A2, V, Atom_count)
    smi = decode_smiles(captions1,captions2)
    smi=filter_unique_canonical(smi)
    #print(smi,len(smi))
    df = pd.DataFrame(smi, columns=['smile']) 
    df.to_csv("gens.csv",index=None,mode='a')
