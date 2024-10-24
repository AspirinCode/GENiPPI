# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

# This script implements functions to generate 3D molecular structures from SMILES strings,
# applies random affine transformations to 3D representations, and prepares molecular data for deep learning models.

import torch
import rdkit  # RDKit library for handling chemical molecules and SMILES strings
from rdkit import Chem  # RDKit functions for molecule creation
from rdkit.Chem import AllChem  # RDKit functions for 3D conformer generation
from gPPMol.utils import *  # Custom utilities from gPPMol
import numpy as np  # Numpy for numerical operations
import multiprocessing  # Multiprocessing for parallelizing data preparation
import math  # Math operations
import random  # Random operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
import matplotlib.image as mpimg  # Matplotlib for image handling
from torchio.transforms import RandomAffine  # TorchIO for applying random affine transformations
from gPPMol.mol_vox import generate_vox  # Custom voxel generation function
import os  # OS module for file handling

# Vocabulary for SMILES tokenization and decoding
vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br",
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",
              "/","\\", "[", "]", "+", "@", "H", "7"
]

# Map indices to characters and vice versa for SMILES tokenization
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}  # Index to character
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}  # Character to index

# Function to generate 3D molecular representation from SMILES and convert to PDB format
def generate_representation(in_smile, small_pdb, public_path):
    """
    Converts SMILES to a 3D molecular structure and saves it as a PDB file.
    
    Args:
        in_smile (str): SMILES string of the molecule.
        small_pdb (str): Name for the output PDB file.
        public_path (str): Path to save the PDB file.
    
    Returns:
        str: Path to the generated PDB file.
    """
    try:
        m = Chem.MolFromSmiles(in_smile)  # Convert SMILES to RDKit molecule
        mh = Chem.AddHs(m)  # Add hydrogen atoms to the molecule
        AllChem.EmbedMolecule(mh, randomSeed=1, maxAttempts=5000, useRandomCoords=True)  # Generate 3D coordinates
        Chem.AllChem.MMFFOptimizeMolecule(mh, maxIters=200)  # Optimize 3D geometry using MMFF
        Chem.rdmolfiles.MolToPDBFile(mh, filename=public_path + small_pdb + ".pdb")  # Save as PDB
        return public_path + small_pdb + ".pdb"  # Return path to PDB file
    except:  # Handle errors during conformer generation
        return None

# Function to convert SMILES to 3D voxel representation and apply random transformations
def resatom(in_smile):
    """
    Converts a SMILES string to a 3D voxel representation and applies random affine transformations.
    
    Args:
        in_smile (str): SMILES string of the molecule.
    
    Returns:
        torch.Tensor: Affine-transformed tensor representation of the molecule.
    """
    public_path = "/home/jmwang/WorkSpace/GENiPPI/gPPMol/tmp"  # Temporary directory for storing files

    tem_floder = public_path
    if not os.path.exists(tem_floder):
        os.mkdir(tem_floder)  # Create the temporary folder if it doesn't exist
        
    in_smile_ask = "/mjs"
    small_pdb = generate_representation(in_smile, in_smile_ask, public_path)  # Generate the PDB file
    if small_pdb is None:
        print("Error in generate_representation() from SMILES to PDB")
        return None

    output_path = public_path + in_smile_ask + ".npy"
    finish_combine = generate_vox(small_pdb, output_path, tem_floder)  # Generate voxel data from the PDB file
    if finish_combine is None:
        return None

    file = torch.Tensor(np.squeeze(finish_combine))  # Convert voxel data to PyTorch tensor

    # Apply random affine transformations (rotation, scaling, translation)
    trans = RandomAffine(scales=(1, 1), translation=5, isotropic=True, degrees=180)
    tensor_trans = trans(file)

    return tensor_trans

# Another version of the resatom function with different paths
def resatom1(in_smile):
    """
    Converts SMILES string to voxel representation and applies random affine transformations (alternative version).
    
    Args:
        in_smile (str): SMILES string of the molecule.
    
    Returns:
        torch.Tensor: Affine-transformed tensor representation of the molecule.
    """
    public_path = "/home/jmwang/WorkSpace/GENiPPI/gPPMol/"

    in_smile_ask = "".join([str(ord(j)) for j in in_smile])  # Create a unique name for the file
    small_pdb = generate_representation(in_smile, in_smile_ask, public_path)
    if small_pdb is None:
        print("Error in generate_representation() from SMILES to PDB")
        return None

    output_path = public_path + in_smile_ask + ".npy"
    tem_floder = public_path + in_smile_ask
    if not os.path.exists(tem_floder):
        os.mkdir(tem_floder)

    finish_combine = generate_vox(small_pdb, output_path, tem_floder)  # Generate voxel data
    if finish_combine is None:
        return None

    if os.path.exists(tem_floder):  # Clean up temporary files
        os.remove(small_pdb)
        os.remove(output_path)

        import shutil
        shutil.rmtree(tem_floder)

    file = torch.Tensor(np.squeeze(finish_combine))  # Convert to tensor

    # Apply random affine transformations
    trans = RandomAffine(scales=(1, 1), translation=5, isotropic=True, degrees=180)
    tensor_trans = trans(file)

    return tensor_trans

# List of file names and the current file ID being processed
name_list = ["./gPPMol/smile_str_resatom_dict_rest.npy"] + ["./gPPMol/smile_str_resatom_dict_" + str(int(10000 * i)) + ".npy" for i in range(1, 31)]
file_id = 0
with open("./gPPMol/file_id.txt", "r") as f:
    file_id = int(f.readline().strip())  # Read the current file ID

data = np.load(name_list[file_id], allow_pickle=True).item()  # Load data

print("Read file_id:", file_id, name_list[file_id])

# Function to randomly select a SMILES string and its corresponding feature from the dataset
def smile_to_f(smile_str):
    """
    Randomly selects a SMILES string and its corresponding features.
    
    Args:
        smile_str (str): Placeholder argument (not used in this function).
    
    Returns:
        Tuple: Selected feature and SMILES string.
    """
    global data

    smile_str = list(data.keys())
    smile_length = len(smile_str) - 1

    smile_id = random.randint(0, smile_length)  # Select a random SMILES string
    return data[smile_str[smile_id]], smile_str[smile_id]

# Function to generate molecular image representation and SMILES string representation
def generate_image_representation(smile, y, id):
    """
    Generates both the 3D voxel representation and tokenized SMILES string for a given molecule.
    
    Args:
        smile (torch.Tensor): Tokenized SMILES string.
        y (int): Label of the molecule.
        id (int): Unique ID for the molecule.
    
    Returns:
        Tuple: (voxel representation, tokenized SMILES, label, length of SMILES string)
    """
    smile_str = list(smile)
    end_token = smile_str.index(2)  # End token in SMILES
    smile_str = "".join([vocab_i2c_v1[i] for i in smile_str[1:end_token]])  # Decode SMILES

    mol = getRDkitMol(smile_str)  # Convert SMILES to RDKit molecule

    if mol is not None:
        if os.path.isfile("./gPPMol/mygen.py"):  # Check if the file exists
            img_, sstring = smile_to_f(smile_str)  # Fetch the precomputed feature
            y = img_[1]
            img_ = img_[0]

            # Replace specific atoms with placeholders
            sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")
            smile = [1] + [vocab_c2i_v1__[xchar] for xchar in sstring] + [2]
            end_token = smile.index(2)

            if img_ is None:
                img_ = resatom(smile_str)  # Generate the voxel representation
        else:
            img_ = resatom(smile_str)
        if img_ is None:
            return None

        return img_, torch.Tensor(smile), y, end_token + 1
    else:
        return None

# Function to prepare a batch of data by collecting features and organizing into tensors
def gather_fn(in_data):
    """
    Prepares a batch of data by collecting images, SMILES, and labels and organizing them into tensors.
    
    Args:
        in_data (list): List of tuples containing (image, SMILES, label, SMILES length).
    
    Returns:
        Tuple: Stacked and processed tensors for images, conditions, labels, SMILES, and lengths.
    """
    in_data.sort(key=lambda x: x[3], reverse=True)  # Sort by SMILES length (descending order)
    images, smiles, y, lengths = zip(*in_data)  # Unpack data
    only_y = y

    # Generate condition tensors based on the labels
    condition_f = []
    protein_ori_f = np.ones((10736, 29), dtype=np.float32)  # Positive condition tensor
    negative_f = np.zeros(protein_ori_f.shape, dtype=np.float32)  # Negative condition tensor
    for i in range(len(y)):
        if y[i] == 0:
            condition_f.append(negative_f)
        elif y[i] == 1:
            condition_f.append(protein_ori_f)

    condition_f = torch.Tensor(np.array(condition_f))

    images = torch.stack(images, 0)  # Stack images into a tensor
    y = torch.LongTensor(y)  # Convert labels to tensor
    c = torch.sparse.torch.eye(2)
    y = c.index_select(0, y)

    # Prepare SMILES tensor
    targets = torch.zeros(len(smiles), max(lengths)).long()
    for i, smile in enumerate(smiles):
        end = lengths[i]
        targets[i, :end] = smile[:end]

    return images, condition_f, y, only_y, targets, lengths

# Class for preparing batches of data in parallel using multiprocessing
class Batch_prep:
    def __init__(self, n_proc=6, mp_pool=None):
        """
        Initializes the Batch_prep class for preparing batches in parallel using multiprocessing.
        
        Args:
            n_proc (int): Number of processes to use for multiprocessing.
            mp_pool (multiprocessing.Pool): Optional pre-existing multiprocessing pool.
        """
        if mp_pool:
            self.mp = mp_pool
        elif n_proc > 1:
            self.mp = multiprocessing.Pool(n_proc)  # Create multiprocessing pool
        else:
            raise NotImplementedError("Use multiprocessing for now!")

        self.total_number = 0

    def transform_data(self, smiles, y, batch_idx):
        """
        Uses multiprocessing to generate 3D representations and tokenized SMILES strings.
        
        Args:
            smiles (list): List of SMILES strings.
            y (list): List of labels.
            batch_idx (list): Batch indices.
        
        Returns:
            Tuple: Processed batch of data.
        """
        inputs = self.mp.starmap(generate_image_representation, zip(smiles, y, batch_idx))  # Parallel processing

        # Filter out failed representations
        inputs = list(filter(lambda x: x is not None, inputs))

        self.total_number = self.total_number + len(inputs)

        return gather_fn(inputs)

# Function to generate batches of data continuously using multiprocessing
def queue_datagen(smiles, y, batch_size=128, n_proc=12, mp_pool=None):
    """
    Continuously generates batches of molecular representations using multiprocessing.
    
    Args:
        smiles (list): List of SMILES strings.
        y (list): List of labels.
        batch_size (int): Number of samples per batch.
        n_proc (int): Number of processes to use for parallelization.
        mp_pool (multiprocessing.Pool): Optional pre-existing multiprocessing pool.
    
    Yields:
        Tuple: Generated batch of data.
    """
    n_batches = math.ceil(len(smiles) / batch_size)
    sh_indencies = np.arange(len(smiles))
    my_batch_prep = Batch_prep(n_proc=n_proc, mp_pool=mp_pool)  # Initialize batch preparation with multiprocessing
    while True:
        np.random.shuffle(sh_indencies)  # Shuffle indices
        for i in range(n_batches):
            batch_idx = sh_indencies[i * batch_size:(i + 1) * batch_size]
            yield my_batch_prep.transform_data(smiles[batch_idx], y[batch_idx], batch_idx)  # Yield processed batch
