# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
import sys
sys.path.append(r'/home/jmwang/WorkSpace/GENiPPI/')

from gPPMol.nets import EncoderCNN, DecoderRNN, generator, discriminator, GNN_Model
from gPPMol.gene import *
from gPPMol.gene import queue_datagen
from gPPMol.utils import *
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import tqdm
from rdkit import Chem
from torch.autograd import Variable
import torch
import time
from gPPMol.ppi_processing.Prepare_Input import Prepare_Input
from gPPMol.ppi_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
import multiprocessing
from gPPMol.utils import prepDatasets

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

def decode_smiles(in_tensor1, in_tensor2):
    """
    Decodes input tensor to a list of strings.
    :param in_tensor:
    :return:
    """
    gen_smiles = []
    for sample in in_tensor1:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)

    for sample in in_tensor2:
        csmile = ""
        for xchar in sample[1:]:
            if xchar == 2:
                break
            csmile += vocab_i2c_v1[xchar]
        gen_smiles.append(csmile)

    return gen_smiles

def filter_unique_canonical(in_mols):
    """
    :param in_mols - list of SMILES strings
    :return: list of uinique and valid SMILES strings in canonical form.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert to RDKit Molecule
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Filter out invalids
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Check for duplicates and filter out invalids


def gen_seq(smile_str=[],y= [1],np_save_path = "./example.npy",y_path = "./example_y.npy"):
    max_length = 0
    for i in smile_str:
        if len(i)>max_length:
            max_length = len(i)
    if len(smile_str)!=len(y):
        print("len(smile_str)!=len(y)")
        return None

    prepDatasets(smile_str, np_save_path, y, y_path, max_length)

    return np_save_path,y_path


class CompoundGenerator:
    def __init__(self, use_cuda=True,params={}):

        self.use_cuda = False
        self.encoder = EncoderCNN(9)
        self.decoder = DecoderRNN(512, 1024, 37, 1)
        self.G       = generator(nc=9,use_cuda=use_cuda)
        self.gnn_interface_model = GNN_Model(params)

        self.G.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.gnn_interface_model.eval()

        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.G.cuda()
            self.gnn_interface_model.cuda()
            self.use_cuda = True

    def load_weight(self, G_weights, encoder_weights, decoder_weights, gnn_interface_weights):
        """
        Load the weights of the models.
        :param vae_weights: str - VAE model weights path
        :param encoder_weights: str - captioning model encoder weights path
        :param decoder_weights: str - captioning model decoder model weights path
        :return: None
        """
        self.gnn_interface_model.load_state_dict(torch.load(gnn_interface_weights, map_location='cpu'))
        self.G.load_state_dict(torch.load(G_weights, map_location='cpu'))
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))


    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from in_shapes
        """
        embedding = self.encoder(in_shapes)
        if probab:
            captions1,captions2 = self.decoder.sample_prob(embedding)
        else:
            captions = self.decoder.sample(embedding)

        captions1 = torch.stack(captions1, 1)
        captions2 = torch.stack(captions2, 1)
        if self.use_cuda:
            captions1 = captions1.cpu().data.numpy()
            captions2 = captions2.cpu().data.numpy()
        else:
            captions1 = captions1.data.numpy()
            captions2 = captions2.data.numpy()
        return decode_smiles(captions1,captions2)

    def generate_molecules(self, smile_str, n_attemps=300, probab=False, filter_unique_valid=True,device="cpu"):
        """
        Generate novel compounds from a seed compound.
        :param smile_str: string - SMILES representation of a molecule
        :param n_attemps: int - number of decoding attempts
        :param probab: boolean - use probabilistic decoding
        :param filter_unique_canonical: boolean - filter for valid and unique molecules
        :return: list of RDKit molecules.
        """
        structure_path="./interface.pdb"
        input_file=Prepare_Input(structure_path)
        list_npz = [input_file]
        dataset = Single_Dataset(list_npz)
        dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=4,
                            drop_last=False, collate_fn=collate_fn)
        for batch_idx, sample in enumerate(dataloader):
            H, A1, A2, V, Atom_count = sample
            break

        np_save_path,y_path = gen_seq(smile_str=[smile_str],y= [1],np_save_path = "./smile_str.npy",y_path = "./smile_str_y.npy")

        smiles = np.load(np_save_path)
        y = np.load(y_path)

        multiproc = multiprocessing.Pool(1)
        my_gen = queue_datagen(smiles, y, batch_size=n_attemps, mp_pool=multiproc)
        mg = GeneratorEnqueuer(my_gen)
        mg.start()
        mt_gen = mg.get()
        tq_gen = tqdm(enumerate(mt_gen))
        for i, (mol_batch, condition, y, only_y, caption, lengths) in tq_gen:
            break

        H_new  = Variable(H.to(device))
        A1_new = Variable(A1.to(device))
        A2_new = Variable(A2.to(device))
        V_new  = Variable(V.to(device))

        num_img = mol_batch.size(0)

        c_output1 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), device)

        z = Variable(torch.randn(num_img, 9, 35, 35, 35))
        
        recoded_shapes = self.G(z.detach(), c_output1, only_y)        
        smiles = self.caption_shape(recoded_shapes, probab=probab)

        if filter_unique_valid:
            return filter_unique_canonical(smiles)

        return [Chem.MolFromSmiles(x) for x in smiles]