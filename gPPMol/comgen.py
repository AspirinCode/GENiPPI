# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

# This script defines various functions and classes for generating molecular compounds from SMILES representations.
# It includes SMILES decoding, filtering valid molecules, sequence generation, and model loading for GNN-based generation.
# It also integrates an encoder-decoder architecture for molecule generation and captioning.

import sys
sys.path.append(r'/home/jmwang/WorkSpace/GENiPPI/')  # Append path for loading custom modules

from gPPMol.nets import EncoderCNN, DecoderRNN, generator, discriminator, GNN_Model  # Importing models for GNN, Generator, Encoder, Decoder
from gPPMol.gene import *  # Import utility functions for generating molecules
from gPPMol.gene import queue_datagen  # Import the data generator
from gPPMol.utils import *  # Import additional utilities
from keras.utils.data_utils import GeneratorEnqueuer  # Enqueuer for Keras-style data generators
from tqdm import tqdm  # Progress bar for long iterations
from rdkit import Chem  # RDKit for molecular manipulations
from torch.autograd import Variable  # For creating PyTorch Variables
import torch  # PyTorch for deep learning
import time  # For tracking execution time
from gPPMol.ppi_processing.Prepare_Input import Prepare_Input  # Custom input preparation for PPI
from gPPMol.ppi_processing.Single_Dataset import Single_Dataset  # Custom dataset loader
from torch.utils.data import DataLoader  # Dataloader for PyTorch
import multiprocessing  # Multiprocessing for parallel data processing
from gPPMol.utils import prepDatasets  # Utility function for preparing datasets

# Define vocabulary for tokenizing and decoding SMILES strings
vocab_list = ["pad", "start", "end",
              "C", "c", "N", "n", "S", "s", "P", "O", "o",
              "B", "F", "I",
              "Cl", "[nH]", "Br", 
              "1", "2", "3", "4", "5", "6",
              "#", "=", "-", "(", ")",  
              "/","\\", "[", "]", "+", "@", "H", "7"
]
# Map indices to characters and vice versa
vocab_i2c_v1 = {i: x for i, x in enumerate(vocab_list)}  # Index to character mapping
vocab_c2i_v1 = {vocab_i2c_v1[i]: i for i in vocab_i2c_v1}  # Character to index mapping

# Function to decode SMILES from tensor format
def decode_smiles(in_tensor1, in_tensor2):
    """
    Decodes input tensors into SMILES string representations.
    
    Args:
        in_tensor1 (torch.Tensor): Tensor containing SMILES indices.
        in_tensor2 (torch.Tensor): Additional tensor containing SMILES indices.
    
    Returns:
        List[str]: List of decoded SMILES strings.
    """
    gen_smiles = []  # List to hold the generated SMILES strings
    for sample in in_tensor1:
        csmile = ""
        for xchar in sample[1:]:  # Skip the start token
            if xchar == 2:  # End token
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

# Function to filter unique and canonical SMILES strings
def filter_unique_canonical(in_mols):
    """
    Filters unique and valid SMILES strings from a list and converts them into their canonical form.
    
    Args:
        in_mols (List[str]): List of SMILES strings.
    
    Returns:
        List[rdkit.Chem.Mol]: List of unique and valid RDKit molecules.
    """
    xresults = [Chem.MolFromSmiles(x) for x in in_mols]  # Convert SMILES to RDKit Molecule objects
    xresults = [Chem.MolToSmiles(x) for x in xresults if x is not None]  # Convert valid molecules back to SMILES
    return [Chem.MolFromSmiles(x) for x in set(xresults)]  # Remove duplicates and return RDKit molecules

# Function to prepare datasets by tokenizing SMILES strings and saving them to numpy arrays
def gen_seq(smile_str=[], y=[1], np_save_path="./example.npy", y_path="./example_y.npy"):
    """
    Generates tokenized SMILES sequences and prepares them for training.
    
    Args:
        smile_str (List[str]): List of SMILES strings.
        y (List[int]): Labels corresponding to the SMILES strings.
        np_save_path (str): Path to save the tokenized SMILES sequences.
        y_path (str): Path to save the labels.
    
    Returns:
        Tuple[str, str]: Paths where the SMILES sequences and labels are saved.
    """
    max_length = 0
    for i in smile_str:
        if len(i) > max_length:  # Determine the maximum length of SMILES strings
            max_length = len(i)
    if len(smile_str) != len(y):  # Ensure SMILES strings and labels have the same length
        print("len(smile_str) != len(y)")
        return None

    # Call utility to prepare datasets (tokenizes the SMILES and saves as numpy arrays)
    prepDatasets(smile_str, np_save_path, y, y_path, max_length)
    
    return np_save_path, y_path

# Class to generate compounds based on GNN and encoder-decoder models
class CompoundGenerator:
    def __init__(self, use_cuda=True, params={}):
        """
        Initializes the compound generator with necessary models: encoder, decoder, generator, and GNN.
        
        Args:
            use_cuda (bool): Whether to use GPU for computation.
            params (dict): Parameters for the GNN model.
        """
        self.use_cuda = False
        self.encoder = EncoderCNN(9)  # Initialize CNN-based encoder
        self.decoder = DecoderRNN(512, 1024, 37, 1)  # Initialize RNN-based decoder
        self.G = generator(nc=9, use_cuda=use_cuda)  # Initialize the GAN generator
        self.gnn_interface_model = GNN_Model(params)  # Initialize the GNN for molecular representation

        # Set models to evaluation mode
        self.G.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.gnn_interface_model.eval()

        # Move models to CUDA if available
        if use_cuda:
            assert torch.cuda.is_available()
            self.encoder.cuda()
            self.decoder.cuda()
            self.G.cuda()
            self.gnn_interface_model.cuda()
            self.use_cuda = True

    def load_weight(self, G_weights, encoder_weights, decoder_weights, gnn_interface_weights):
        """
        Load pre-trained weights for the models.
        
        Args:
            G_weights (str): Path to the generator weights.
            encoder_weights (str): Path to the encoder weights.
            decoder_weights (str): Path to the decoder weights.
            gnn_interface_weights (str): Path to the GNN model weights.
        
        Returns:
            None
        """
        # Load weights for all models
        self.gnn_interface_model.load_state_dict(torch.load(gnn_interface_weights, map_location='cpu'))
        self.G.load_state_dict(torch.load(G_weights, map_location='cpu'))
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))

    def caption_shape(self, in_shapes, probab=False):
        """
        Generates SMILES representation from molecular shapes using the encoder-decoder model.
        
        Args:
            in_shapes (torch.Tensor): Input molecular shapes.
            probab (bool): Whether to use probabilistic decoding.
        
        Returns:
            List[str]: List of generated SMILES strings.
        """
        embedding = self.encoder(in_shapes)  # Encode the input shapes
        if probab:
            captions1, captions2 = self.decoder.sample_prob(embedding)  # Probabilistic decoding
        else:
            captions = self.decoder.sample(embedding)  # Greedy decoding

        captions1 = torch.stack(captions1, 1)
        captions2 = torch.stack(captions2, 1)

        # Move data to CPU if necessary and decode into SMILES strings
        if self.use_cuda:
            captions1 = captions1.cpu().data.numpy()
            captions2 = captions2.cpu().data.numpy()
        else:
            captions1 = captions1.data.numpy()
            captions2 = captions2.data.numpy()
        return decode_smiles(captions1, captions2)

    def generate_molecules(self, smile_str, n_attemps=300, probab=False, filter_unique_valid=True, device="cpu"):
        """
        Generate novel compounds starting from a seed SMILES string.
        
        Args:
            smile_str (str): Seed SMILES string.
            n_attemps (int): Number of attempts to generate molecules.
            probab (bool): Whether to use probabilistic decoding.
            filter_unique_valid (bool): Whether to filter out invalid and duplicate molecules.
            device (str): The device to use for computation ("cpu" or "cuda").
        
        Returns:
            List[rdkit.Chem.Mol]: List of valid, unique RDKit molecule objects.
        """
        structure_path = "./interface.pdb"
        input_file = Prepare_Input(structure_path)
        list_npz = [input_file]  # Prepare molecular structure data
        dataset = Single_Dataset(list_npz)  # Create dataset
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=4, drop_last=False, collate_fn=collate_fn)

        # Load the first batch of molecular data
        for batch_idx, sample in enumerate(dataloader):
            H, A1, A2, V, Atom_count = sample
            break

        # Tokenize the input SMILES string
        np_save_path, y_path = gen_seq(smile_str=[smile_str], y=[1], np_save_path="./smile_str.npy", y_path="./smile_str_y.npy")

        # Load the tokenized SMILES strings and labels
        smiles = np.load(np_save_path)
        y = np.load(y_path)

        # Start multiprocessing for data generation
        multiproc = multiprocessing.Pool(1)
        my_gen = queue_datagen(smiles, y, batch_size=n_attemps, mp_pool=multiproc)
        mg = GeneratorEnqueuer(my_gen)  # Queue the data generator
        mg.start()
        mt_gen = mg.get()  # Get the generator
        tq_gen = tqdm(enumerate(mt_gen))  # Progress bar for data generation
        for i, (mol_batch, condition, y, only_y, caption, lengths) in tq_gen:
            break

        # Convert input to PyTorch variables
        H_new = Variable(H.to(device))
        A1_new = Variable(A1.to(device))
        A2_new = Variable(A2.to(device))
        V_new = Variable(V.to(device))

        num_img = mol_batch.size(0)

        # Pass the molecular data through the GNN model
        c_output1 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), device)

        # Sample latent vectors for the GAN
        z = Variable(torch.randn(num_img, 9, 35, 35, 35))

        # Generate molecular shapes using the GAN
        recoded_shapes = self.G(z.detach(), c_output1, only_y)

        # Decode the generated shapes into SMILES strings
        smiles = self.caption_shape(recoded_shapes, probab=probab)

        # Optionally filter out invalid and duplicate molecules
        if filter_unique_valid:
            return filter_unique_canonical(smiles)

        # Return generated RDKit molecules
        return [Chem.MolFromSmiles(x) for x in smiles]
