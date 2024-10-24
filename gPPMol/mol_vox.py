# Copyright (C) 2019 by Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
#
# This script generates 3D voxel-based molecular representations from ligand PDB files.
# It uses the RDKit library for chemical processing, MoleculeKit for voxel descriptor generation,
# and TorchIO for random affine transformations. The resulting voxel descriptors can be used
# in machine learning models for molecular property prediction.

import torch  # PyTorch for tensor manipulations
import torch.nn as nn  # PyTorch neural network utilities
from rdkit import Chem  # RDKit for handling chemical files (SMILES, PDB, etc.)
from moleculekit.molecule import Molecule  # MoleculeKit for molecular manipulations
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures  # Voxel descriptor tools
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping  # Atom typing utilities for proteins
from moleculekit.smallmol.smallmol import SmallMol  # Handling small molecules
from moleculekit.home import home  # Home directory for MoleculeKit
import moleculekit  # Main MoleculeKit package
import math  # For mathematical operations
import numpy  # For numerical computations and array manipulations
import os  # For file and directory operations
import shutil  # For file operations like copying and moving
import sys  # For system-level operations
from torchio.transforms import RandomAffine  # TorchIO library for random affine transformations (data augmentation)
import numpy as np  # NumPy for matrix and array operations

# Function to convert a scientific notation string (e.g., "1e+5") to a numerical value
def str2num(string):
    """
    Converts a string in scientific notation format into a numerical float value.

    Args:
        string (str): The string to convert, usually in scientific notation (e.g., "1e+5").
    
    Returns:
        float: The corresponding float value after conversion.
    """
    try:
        temp = string.split('+')  # Split the string at '+', separating base and exponent
        base, index = temp[0], temp[1]  # Base value and exponent
        base = base.split('e')[0]  # Handle scientific notation formatted as 'e'
        return float(base) * (10 ** int(index))  # Recalculate number from base and exponent
    except:
        return float(string)  # If conversion fails, return the float value of the string directly

# Main function to generate voxel descriptors for a ligand
def generate_vox(ligand_name, output_path, tem_floder):
    """
    Generates voxel descriptors and electrostatic potential data for a ligand using MoleculeKit and Multiwfn.

    Args:
        ligand_name (str): Path to the ligand's PDB file.
        output_path (str): Path where the final voxel data (numpy array) will be saved.
        tem_floder (str): Path to a temporary folder for storing intermediate files.

    Returns:
        numpy.ndarray: The combined voxel descriptor and electrostatic data in a 3D array.
    """
    # Step 1: Load the ligand molecule and calculate its center of mass
    try:
        mol = SmallMol(ligand_name, force_reading=True)  # Load ligand using SmallMol
        center = mol.getCenter()  # Get center of mass of the ligand molecule
    except:
        print("Cannot read the file!")  # Error if the file cannot be read
        return None

    # Step 2: Create a configuration file for running Multiwfn (an external tool)
    try:
        with open(tem_floder + "/run.txt", "w") as txt_file:
            txt_file.write("5\n")  # Example parameters for running Multiwfn
            txt_file.write("1\n")
            txt_file.write("6\n")
            txt_file.write(f"{center[0]},{center[1]},{center[2]}\n")  # Molecule center (x, y, z)
            txt_file.write("35,35,35\n")  # Box size (grid dimensions)
            txt_file.write("66,66,66\n")  # Grid size for the voxel grid
            txt_file.write("3\n")  # Additional parameters for Multiwfn
    except:
        print("Cannot create run file!")  # Error if the file cannot be created
        return None

    # Step 3: Run Multiwfn to calculate electrostatic properties and move the result file
    try:
        cmd = f"Multiwfn {ligand_name} < {tem_floder}/run.txt > {tem_floder}/medinfo.txt > out.out"
        os.system(cmd)  # Run Multiwfn with the generated input file
        shutil.move("./output.txt", tem_floder + "/output_ligand.txt")  # Move the result file to the temporary folder
    except:
        print("Cannot run Multiwfn_ligand")  # Error if Multiwfn fails to run
        return None

    # Step 4: Read the output file and process the electrostatic potential data
    try:
        file = open(tem_floder + "/output_ligand.txt")  # Open the Multiwfn output file
        file_list = file.readlines()  # Read all lines from the file

        # Initialize a list to store the 3D electrostatic data
        new_list = []
        for a_i in range(35):  # Iterate over the voxel grid (35x35x35)
            tem_list_x = []
            for b_i in range(35):
                tem_list_y = []
                for c_i in range(35):
                    number_tem = a_i * 35 * 35 + b_i * 35 + c_i  # Calculate the index in the file
                    tem = (file_list[number_tem]).split()[3]  # Extract the electrostatic value
                    tem = str2num(tem)  # Convert the value from string to number
                    if tem != 0:
                        tem = math.log10(tem)  # Apply log transformation to the value
                    tem_list_y.append(tem)  # Append the transformed value to the y-dimension
                tem_list_x.append(tem_list_y)  # Append to the x-dimension
            new_list.append(tem_list_x)  # Build the full 3D grid
        elect_np_ligand = numpy.array(new_list)  # Convert the list into a numpy array
    except:
        print("Cannot create ligand_elect_den")  # Error if file reading or data processing fails
        return None

    # Step 5: Generate voxel descriptors for the ligand using MoleculeKit
    try:
        mol = SmallMol(ligand_name, force_reading=True)  # Reload the ligand
        center = mol.getCenter()  # Get the molecular center again
        box = [35, 35, 35]  # Voxel grid dimensions

        # Get voxel descriptors: molecular features like hydrophobicity, aromaticity, etc.
        mol_vox, mol_centers, mol_N = getVoxelDescriptors(mol, boxsize=box, voxelsize=1, buffer=0, center=center, validitychecks=False)

        # Reshape the voxel descriptor array to the desired format
        mol_vox_t = mol_vox.transpose().reshape([1, mol_vox.shape[1], mol_N[0], mol_N[1], mol_N[2]])
    except:
        print("Cannot perform voxelization")  # Error if voxel generation fails
        return None

    # Step 6: Combine electrostatic potential data with the voxel descriptors
    finish_combine = numpy.squeeze(mol_vox_t)  # Squeeze out unnecessary dimensions
    finish_combine = numpy.insert(finish_combine, 8, elect_np_ligand, axis=0)  # Insert the electrostatic data as a feature
    numpy.save(output_path, finish_combine)  # Save the final combined array as a .npy file
    
    print("Voxelization complete")  # Indicate successful completion
    return finish_combine  # Return the combined 3D voxel array

# Main block to execute the script
if __name__ == "__main__":
    ligand_name = "./1a30_ligand.pdb"  # Path to the input ligand PDB file
    output_path = "./1a30.npy"  # Path to save the output voxel descriptors
    tem_floder = "./tmp"  # Temporary folder for intermediate files

    # Step 1: Generate voxel descriptors for the ligand
    finish_combine = generate_vox(ligand_name, output_path, tem_floder)

    # Step 2: Print the size of the generated voxel array
    print("Voxel size:", finish_combine.shape)

    # Step 3: Convert the voxel array to a PyTorch tensor
    file = torch.Tensor(np.squeeze(finish_combine))

    # Step 4: Apply random affine transformations to the tensor (for data augmentation)
    trans = RandomAffine(scales=(1, 1), translation=5, isotropic=True, degrees=180)  # Set random affine parameters
    tensor_trans = trans(file)  # Apply transformation

    # Step 5: Add an extra dimension to the transformed tensor
    tensor_trans2 = tensor_trans.unsqueeze(0)

    # Step 6: Print the shape of the final tensor after transformation
    print(tensor_trans2.shape)
