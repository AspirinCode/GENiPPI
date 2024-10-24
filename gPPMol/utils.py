# Importing necessary libraries for molecular processing, voxelization, rotation, and data preparation.
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from moleculekit.util import uniformRandomRotation  # For random 3D rotations
from moleculekit.smallmol.smallmol import SmallMol  # Small molecule handling in moleculekit
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures  # Voxelization tools

# Additional imports for numerical operations, plotting, and multiprocessing
import matplotlib as mpl
mpl.use('Agg')  # Non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix  # Sparse matrix handling for large datasets
import multiprocessing  # For parallel processing
import math, os
import random

# Importing functions and classes related to moleculekit
from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping  # Atom typing for proteins
from moleculekit.home import home
import moleculekit

# Vocabulary for SMILES tokenization
vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", 
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",  
              "/","\\", "[", "]", "+", "@", "H", "7"]

# Mapping index to SMILES characters and reverse mapping (char to index)
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}

# PrepDatasets: Prepares datasets by tokenizing SMILES strings into indices and saving them as numpy arrays.
def prepDatasets(smiles, np_save_path, y, y_path, length):
    """
    Prepares and processes a list of SMILES strings, tokenizes them based on a predefined vocabulary,
    and stores them in numpy arrays for further use. The function also handles token padding.

    Args:
        smiles (list of str): List of SMILES strings to be tokenized.
        np_save_path (str): Path to save the tokenized SMILES numpy array.
        y (list): List of labels associated with the molecules.
        y_path (str): Path to save the numpy array of labels.
        length (int): Maximum length for the tokenized SMILES strings.
    """
    from tqdm import tqdm  # Progress bar for iteration

    strings = np.zeros((len(smiles), length+2), dtype='uint8')  # Initialize a numpy array to store SMILES tokens
    
    # Vocabulary for SMILES tokenization (extended)
    vocab_list__ = ["pad", "start", "end",
        "C", "c", "N", "n", "S", "s", "P", "O", "o",
        "B", "F", "I",
        "X", "Y", "Z",
        "1", "2", "3", "4", "5", "6",
        "#", "=", "-", "(", ")",
        "/", "\\", "[", "]", "+", "@", "H", "7"]

    # Create mappings from indices to characters and vice versa
    vocab_i2c_v1__ = {i: x for i, x in enumerate(vocab_list__)}
    vocab_c2i_v1__ = {vocab_i2c_v1__[i]: i for i in vocab_i2c_v1__}

    # Process each SMILES string and convert to tokenized form
    for i, sstring in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(sstring)  # Convert SMILES to RDKit Mol object
        
        if sstring.find('.') != -1:  # Handle cases with disconnected fragments
            lfc = MolStandardize.fragment.LargestFragmentChooser()
            mol = lfc.choose(mol)  # Choose the largest fragment for consistency

        if not mol:
            raise ValueError(f"Failed to parse molecule '{mol}'")

        sstring = Chem.MolToSmiles(mol)  # Canonicalize the SMILES string for consistency
        # Replace special characters for specific atoms
        sstring = sstring.replace("Cl", "X").replace("[nH]", "Y").replace("Br", "Z")

        try:
            # Convert SMILES string to a list of token indices
            vals = [1] + [vocab_c2i_v1__[xchar] for xchar in sstring] + [2]  # Add start (1) and end (2) tokens
        except KeyError:
            raise ValueError(f"Unknown SMILES tokens: {', '.join([x for x in sstring if x not in vocab_c2i_v1__])}, SMILES: {sstring}")

        strings[i, :len(vals)] = vals  # Fill the array with tokenized values

        if i > 999999:  # Limit processing for large datasets
            break

    # Save the tokenized SMILES strings and labels as numpy arrays
    np.save(np_save_path, strings)
    np.save(y_path, y)

# getRDkitMol: Converts SMILES strings into RDKit molecule objects and generates 3D conformers.
def getRDkitMol(in_smile):
    """
    Converts a SMILES string into an RDKit molecule object and generates its 3D conformer. 
    Hydrogens are added and MMFF force field optimization is applied.

    Args:
        in_smile (str): The SMILES string to be converted into a 3D conformer.
    
    Returns:
        SmallMol: A SmallMol object with the optimized 3D conformation.
    """
    try:
        m = Chem.MolFromSmiles(in_smile)  # Parse the SMILES into an RDKit Mol object
        mh = Chem.AddHs(m)  # Add hydrogen atoms to the molecule
        AllChem.EmbedMolecule(mh)  # Generate 3D coordinates using embedding
        Chem.AllChem.MMFFOptimizeMolecule(mh)  # Optimize geometry using MMFF force field
        m = Chem.RemoveHs(mh)  # Remove hydrogens after optimization
        mol = SmallMol(m)  # Convert to a SmallMol object from Moleculekit
        return mol  # Return the optimized SmallMol object
    except:
        return None  # Return None if the process fails

# rotate: Applies a rotation matrix to a set of atomic coordinates around a specified center.
def rotate(coords, rotMat, center=(0, 0, 0)):
    """
    Rotates a set of atomic coordinates by a given rotation matrix around a specified center.

    Args:
        coords (numpy array): Array of atomic coordinates to be rotated.
        rotMat (numpy array): 3x3 rotation matrix to apply to the coordinates.
        center (tuple): The center point around which the rotation is applied (default: origin (0, 0, 0)).
    
    Returns:
        numpy array: The rotated coordinates.
    """
    newcoords = coords - center  # Translate coordinates to the origin (center)
    return np.dot(newcoords, np.transpose(rotMat)) + center  # Apply rotation and translate back

# length_func: Returns the length of a list or tensor. 
def length_func(list_or_tensor):
    """
    Returns the length of a list or tensor. Supports both Python lists and PyTorch tensors.

    Args:
        list_or_tensor (list or tensor): The input object whose length is to be measured.
    
    Returns:
        int: Length of the input object.
    """
    if type(list_or_tensor) == list:
        return len(list_or_tensor)  # Return length for Python lists
    return list_or_tensor.shape[0]  # Return the first dimension size for tensors
