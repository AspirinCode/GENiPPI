import rdkit
from rdkit import Chem
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.home import home
import moleculekit
import math
import numpy
import os
import sys
import shutil
from torchio.transforms import RandomAffine
import torch
import numpy as np

def str2num(string):
    try:
        temp = string.split('+')
        base, index = temp[0], temp[1]
        base = base.split('e')[0]
        return float(base)*(10**int(index))
    except:
        return float(string)


def generate_vox(ligand_name,output_path,tem_floder):
    try:
        mol = SmallMol(ligand_name,force_reading=True)
        center = mol.getCenter()
    except:
        print("Can not read file!")
        #sys.exit()
        return None
    try:
        with open(tem_floder + "/run.txt", "w") as txt_file:
            txt_file.write("5\n")
            txt_file.write("1\n")
            txt_file.write("6\n")
            txt_file.write(str(center[0]) + "," + str(center[1]) + "," + str(center[2]) + "\n")
            txt_file.write("35,35,35\n")
            txt_file.write("66,66,66\n")
            txt_file.write("3\n")
    except:
        print("Can not create run file!")
        #sys.exit()
        return None

    try:
        cmd = "Multiwfn " + ligand_name + " < " + tem_floder + "/run.txt > " + tem_floder + "/medinfo.txt > out.out"
        os.system(cmd)
        shutil.move("./output.txt", tem_floder+"/output_ligand.txt")
    except:
        print("Can not run Multiwfn_ligand")
        #sys.exit()
        return None

    try:
        file = open(tem_floder + "/output_ligand.txt")
        file_list = file.readlines()
        new_list = []
        for a_i in range(35):
            tem_list_x = []
            for b_i in range(35):
                tem_list_y = []
                for c_i in range(35):
                    number_tem = a_i * 35 * 35 + b_i * 35 + c_i
                    tem = (file_list[number_tem]).split()[3]
                    tem = str2num(tem)
                    if tem == 0:
                        pass
                    else:
                        tem = math.log10(tem)
                    tem_list_y.append(tem)
                tem_list_x.append(tem_list_y)
            new_list.append(tem_list_x)
        elect_np_ligand = numpy.array(new_list)
    except:
        print("Can not create ligand_elect_den")
        #sys.exit()
        return None
    try:
        mol = SmallMol(ligand_name, force_reading=True)
        center = mol.getCenter()
        box = [35, 35, 35]
        #getVoxelDescriptors calculates feature of the mol object and return features as array, centers of voxel
        #The features define the 8 feature of the voxel, (‘hydrophobic’, ‘aromatic’, ‘hbond_acceptor’, ‘hbond_donor’, ‘positive_ionizable’, ‘negative_ionizable’, ‘metal’, ‘occupancies’).
        mol_vox, mol_centers, mol_N = getVoxelDescriptors(mol, boxsize=box, voxelsize=1, buffer=0, center=center,validitychecks =False)
        #print(mol_vox, mol_centers, mol_N ) #mol_N = [35 35 35]

        #print(mol_vox.shape,mol_centers.shape,mol_N.shape)  #(42875, 8) (42875, 3) (3,)

        mol_vox_t = mol_vox.transpose().reshape([1, mol_vox.shape[1], mol_N[0], mol_N[1], mol_N[2]])
    except:
        print("Can not Voxelization")
        #sys.exit()
        return None

    finish_combine = numpy.squeeze(mol_vox_t)
    finish_combine = numpy.insert(finish_combine, 8, elect_np_ligand, axis=0)
    numpy.save(output_path,finish_combine)
    
    print("finish vox")

    return finish_combine

if __name__ == "__main__":

    ligand_name = "./1a30_ligand.pdb"
    output_path = "./1a30.npy"
    tem_floder = "./tmp"

    finish_combine = generate_vox(ligand_name,output_path,tem_floder)

    print("voxelsize:",finish_combine.shape)

    file = torch.Tensor(np.squeeze(finish_combine))

    trans = RandomAffine(scales=(1, 1), translation=5, isotropic=True, degrees=180)
    tensor_trans = trans(file)
    tensor_trans2 = tensor_trans.unsqueeze(0)

    print(tensor_trans2.shape)