import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from moleculekit.util import uniformRandomRotation
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import multiprocessing
import math, os
import random

from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.home import home
import moleculekit

vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", # "X", "Y", "Z",
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",  # Misc
              "/","\\", "[", "]", "+", "@", "H", "7"
]

vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}

vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}


def prepDatasets(smiles, np_save_path, y, y_path, length):
    from tqdm import tqdm

    strings = np.zeros((len(smiles), length+2), dtype='uint8')     # 103+2

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


    for i, sstring in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(sstring)
        aa = sstring
        if sstring.find('.') != -1:
            lfc = MolStandardize.fragment.LargestFragmentChooser()
            mol = lfc.choose(mol)

        if not mol:
            raise ValueError("Failed to parse molecule '{}'".format(mol))

        sstring = Chem.MolToSmiles(mol)  # Make the SMILES canonical.
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
        try:
            vals = [1] + [vocab_c2i_v1__[xchar] for xchar in sstring] + [2]
        except KeyError:
            raise ValueError(("Unkown SMILES tokens: {} in string '{}'."
                              .format(", ".join([x for x in sstring if x not in vocab_c2i_v1__]),
                                                                          sstring)))
        strings[i, :len(vals)] = vals

        if i>999999:
            break

    np.save(np_save_path, strings)
    np.save(y_path, y)


def getRDkitMol(in_smile):
    try:
        m = Chem.MolFromSmiles(in_smile)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.AllChem.MMFFOptimizeMolecule(mh)
        m = Chem.RemoveHs(mh)
        mol = SmallMol(m)
        return mol
    except:
        return None


def rotate(coords, rotMat, center=(0,0,0)):
    """ Rotate a selection of atoms by a given rotation around a center """
    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center


def length_func(list_or_tensor):
    if type(list_or_tensor)==list:
        return len(list_or_tensor)
    return list_or_tensor.shape[0]
