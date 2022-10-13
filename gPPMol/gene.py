# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from gPPMol.utils import *
import numpy as np
import multiprocessing
import math
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchio.transforms import RandomAffine
from gPPMol.mol_vox import generate_vox
import os
vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", #"X", "Y", "Z",
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",
              "/","\\", "[", "]", "+", "@", "H", "7"
]

vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}


def generate_representation(in_smile,small_pdb,public_path):
    """
    Makes embeddings of Molecule.
    """
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh,randomSeed=1,maxAttempts=5000,useRandomCoords=True)
        Chem.AllChem.MMFFOptimizeMolecule(mh,maxIters=200)
        #m = Chem.RemoveHs(mh)
        Chem.rdmolfiles.MolToPDBFile(mh, filename=public_path+small_pdb+".pdb")
        return public_path+small_pdb+".pdb"
    except:  # Rarely the conformer generation fails
        return None

def resatom(in_smile):
    public_path = "/home/jmwang/WorkSpace/GENiPPI/gPPMol/tmp"

    tem_floder = public_path
    if not os.path.exists(tem_floder):
        os.mkdir(tem_floder)
        
    in_smile_ask = "/mjs"
    small_pdb = generate_representation(in_smile,in_smile_ask,public_path)
    if small_pdb is  None:
        print("error happen in generate_representation() from smile to pdb")
        return None

    output_path = public_path+in_smile_ask+".npy"
    


    finish_combine = generate_vox(small_pdb,output_path,tem_floder)  #(9, 35, 35, 35)
    if finish_combine is None:
        return None

    #file = torch.Tensor(np.squeeze(np.load(output_path)))
    file = torch.Tensor(np.squeeze(finish_combine))
    #print("finish_combine:",file.shape,torch.max(file),torch.min(file))

    #RandomAffine：随机仿射变换，包括尺度（scale，需要指定缩放的比例，可以设置缩放时的差值策略），旋转（degrees，需要指定每个轴旋转的角度范围，可以设置旋转时pad的数值），平移（translation），还支持各向同性和设置以中心为基准进行变换
    trans = RandomAffine(scales=(1, 1), translation=5, isotropic=True, degrees=180)
    tensor_trans = trans(file)
    #tensor_trans2 = tensor_trans.unsqueeze(0)
    #print("RandomAffine:",tensor_trans.shape,torch.max(tensor_trans),torch.min(tensor_trans))

    return tensor_trans

def resatom1(in_smile):
    public_path = "/home/jmwang/WorkSpace/GENiPPI/gPPMol/"

    in_smile_ask = "".join([str(ord(j)) for j in in_smile])
    small_pdb = generate_representation(in_smile,in_smile_ask,public_path)
    if small_pdb is  None:
        print("error happen in generate_representation() from smile to pdb")
        return None

    output_path = public_path+in_smile_ask+".npy"

    tem_floder = public_path+in_smile_ask
    if not os.path.exists(tem_floder):
        os.mkdir(tem_floder)

    finish_combine = generate_vox(small_pdb,output_path,tem_floder)  #(9, 35, 35, 35)
    if finish_combine is None:
        return None

    if os.path.exists(tem_floder):        
        os.remove(small_pdb)
        os.remove(output_path)

        import shutil
        shutil.rmtree(tem_floder)

    #file = torch.Tensor(np.squeeze(np.load(output_path)))
    file = torch.Tensor(np.squeeze(finish_combine))
    #print("finish_combine:",file.shape,torch.max(file),torch.min(file))

    #RandomAffine：随机仿射变换，包括尺度（scale，需要指定缩放的比例，可以设置缩放时的差值策略），旋转（degrees，需要指定每个轴旋转的角度范围，可以设置旋转时pad的数值），平移（translation），还支持各向同性和设置以中心为基准进行变换
    trans = RandomAffine(scales=(1, 1), translation=5, isotropic=True, degrees=180)
    tensor_trans = trans(file)
    #tensor_trans2 = tensor_trans.unsqueeze(0)
    #print("RandomAffine:",tensor_trans.shape,torch.max(tensor_trans),torch.min(tensor_trans))

    return tensor_trans

name_list = ["./gPPMol/smile_str_resatom_dict_rest.npy"]+["./gPPMol/smile_str_resatom_dict_"+str(int(10000*i))+".npy" for i in range(1,31)]
file_id =0
with open("./gPPMol/file_id.txt","r") as f :
    file_id = int(f.readline().strip())

data = np.load(name_list[file_id], allow_pickle=True).item()

print("read file_id:",file_id,name_list[file_id])

def smile_to_f(smile_str):
    global data

    smile_str = list(data.keys())
    smile_lenth = len(smile_str)-1

    smile_id  = random.randint(0,smile_lenth)
    return data[smile_str[smile_id]],smile_str[smile_id]

def generate_image_representation(smile,y,id):
    """
    Generate resatom feature and string representation of a molecule
    """
    # Convert smile to 3D structure

    smile_str = list(smile)
    end_token = smile_str.index(2)
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])
    mol = getRDkitMol(smile_str)

    vocab_list__ = ["pad", "start", "end",
        "C", "c", "N", "n", "S", "s", "P", "O", "o",
        "B", "F", "I",
        "X", "Y", "Z",
        "1", "2", "3", "4", "5", "6",
        "#", "=", "-", "(", ")",
        "/", "\\", "[", "]", "+", "@", "H", "7"
    ]
    vocab_i2c_v1__ = {i: x for i, x in enumerate(vocab_list__)}
    vocab_c2i_v1__ = {vocab_i2c_v1__[i]: i for i in vocab_i2c_v1__}

    if mol is not None:
        if os.path.isfile("./gPPMol/mygen.py"):
            img_,sstring = smile_to_f(smile_str)
            y = img_[1]
            img_ = img_[0]

            sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
            smile = [1] + [vocab_c2i_v1__[xchar] for xchar in sstring] + [2]
            end_token = smile.index(2)

            if img_ is None:
                img_ = resatom(smile_str)
        else:
            img_ = resatom(smile_str)
        if img_ is None:
            return None

        return img_, torch.Tensor(smile), y, end_token + 1
    else:
        return None

def gather_fn(in_data):
    """
    Collects and creates a batch.
    """
    # Sort a data list by smiles length (descending order)
    in_data.sort(key=lambda x: x[3], reverse=True)
    images, smiles, y, lengths = zip(*in_data)
    only_y = y
    condition_f = []
    protein_ori_f = np.ones((10736, 29),dtype=np.float32)  #(10736, 29)
    negative_f = np.zeros(protein_ori_f.shape,dtype=np.float32)
    for i in range(len(y)):
        if y[i]==0:
            condition_f.append(negative_f)
        elif y[i]==1:
            condition_f.append(protein_ori_f)

    condition_f = torch.Tensor(np.array(condition_f))

    images = torch.stack(images, 0)  # Stack images
    y = torch.LongTensor(y)
    c = torch.sparse.torch.eye(2)
    y = c.index_select(0,y)
    # y = torch.stack(y, 0)
    # Merge smiles (from tuple of 1D tensor to 2D tensor).
    # lengths = [len(smile) for smile in smiles]
    targets = torch.zeros(len(smiles), max(lengths)).long()
    for i, smile in enumerate(smiles):
        end = lengths[i]
        targets[i, :end] = smile[:end]
    return images, condition_f, y, only_y, targets, lengths


class Batch_prep:
    def __init__(self, n_proc=6, mp_pool=None):
        if mp_pool:
            self.mp = mp_pool
        elif n_proc > 1:
            self.mp = multiprocessing.Pool(n_proc)  #n_proc
        else:
            raise NotImplementedError("Use multiprocessing for now!")

        self.total_number = 0

    def transform_data(self, smiles, y, batch_idx):

        inputs = self.mp.starmap(generate_image_representation, zip(smiles, y, batch_idx))

        # Sometimes representation generation fails
        inputs = list(filter(lambda x: x is not None, inputs))

        self.total_number = self.total_number+len(inputs)

        return gather_fn(inputs)

def queue_datagen(smiles, y, batch_size=128, n_proc=12, mp_pool=None):
    """
    Continuously produce representations.
    """
    n_batches = math.ceil(len(smiles) / batch_size)
    sh_indencies = np.arange(len(smiles))
    my_batch_prep = Batch_prep(n_proc=n_proc, mp_pool=mp_pool)
    while True:
        np.random.shuffle(sh_indencies)
        for i in range(n_batches):
            batch_idx = sh_indencies[i * batch_size:(i + 1) * batch_size]
            yield my_batch_prep.transform_data(smiles[batch_idx], y[batch_idx], batch_idx)
