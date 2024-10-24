# Copyright (C) 2024 by mao jiashun and wang jianming
# Copying and distribution is allowed under AGPLv3 license

# This script implements the training of a WGAN (either WGAN-CP or WGAN-GP) on molecular data,
# leveraging graph neural networks (GNN) for 3D molecular structure representations and SMILES-based data.
# The model involves an encoder-decoder architecture and uses datasets of SMILES strings for training.

import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append(r'/home/jmwang/WorkSpace/GENiPPI/')  # Append custom workspace path for loading modules
from gPPMol.nets import EncoderCNN, DecoderRNN, GNN_Model  # Importing necessary neural network models

from gPPMol.utils import *  # Import utility functions from gPPMol package
from tqdm import tqdm  # Progress bar for training
import argparse
import multiprocessing  # For parallel processing
from gPPMol.ppi_processing.collate_fn import collate_fn  # Import custom collate function for dataloader
from gPPMol.ppi_processing.Prepare_Input import Prepare_Input  # For preparing input data
from gPPMol.ppi_processing.Single_Dataset import Single_Dataset  # Dataset class for processing inputs
from torch.utils.data import DataLoader  # Dataloader for batching and parallel processing

from gPPMol.models_3d.wgan_clipping import WGAN_CP  # Import WGAN model with clipping
from gPPMol.models_3d.wgan_gradient_penalty import WGAN_GP  # Import WGAN model with gradient penalty

# Set the batch size and create directory to save models
batch_size = 8
savedir = "model"
os.makedirs(savedir, exist_ok=True)  # Create directory if it doesn't exist

# Set the device to CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)  # Print the selected device (GPU/CPU)
is_cuda = True if device == "cuda" else False

# Load the molecular structure and prepare it for processing
structure_path = "./interface.pdb"  # Path to the PDB structure file
input_file = Prepare_Input(structure_path)  # Prepare the input using a custom class
list_npz = [input_file]  # List of prepared inputs

# Load the dataset and create a DataLoader for batching data
dataset = Single_Dataset(list_npz)
dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=1, drop_last=False, collate_fn=collate_fn)

# Load the first batch of data from the DataLoader
for batch_idx, sample in enumerate(dataloader):
    H, A1, A2, V, Atom_count = sample  # Load graph data (H, adjacency matrices A1, A2, V, Atom count)
    break

# Load SMILES strings and corresponding labels
smiles = np.load("./gPPMol/example.npy")  # Load SMILES data
y = np.load("./gPPMol/example_y.npy")  # Load labels for SMILES strings

file_id = 0  # Default file identifier

# Check if a file ID is passed via command line arguments
if len(sys.argv) > 1:
    file_id = sys.argv[1]

# Write the file ID to a text file
with open("./gPPMol/file_id.txt", "w") as f:
    f.write(str(file_id) + "\n")

print("write file_id:", file_id)

# Importing additional utilities for SMILES decoding and data generation
from gPPMol.comgen import decode_smiles
from gPPMol.gene import queue_datagen

# Set up multiprocessing for data generation
multiproc = multiprocessing.Pool(1)
my_gen = queue_datagen(smiles, y, batch_size=batch_size, mp_pool=multiproc)  # Data generator

# Define parameters for the GNN model
params = {
    'n_graph_layer': 4,  # Number of graph layers in GNN
    'd_graph_layer': 140,  # Number of nodes per graph layer
    'n_FC_layer': 4,  # Number of fully connected layers
    'd_FC_layer': 128,  # Number of nodes in each fully connected layer
    'dropout_rate': 0.3,  # Dropout rate for regularization
    'initial_mu': 0.0,  # Initial mean for weight initialization
    'initial_dev': 1.0,  # Initial deviation for weight initialization
    'N_atom_features': 28,  # Number of atom features (fixed)
    'final_dimension': 4 * 4 * 4  # Final feature map dimension (size of feature vector)
}

# Instantiate the GNN model and Encoder-Decoder networks
gnn_interface_model = GNN_Model(params)  # Graph Neural Network for interface modeling
encoder = EncoderCNN(9)  # CNN-based encoder for 3D data with 9 input channels
decoder = DecoderRNN(512, 1024, 37, 1)  # RNN-based decoder for sequence generation

# Move models to the device (GPU/CPU)
gnn_interface_model.to(device)
encoder.to(device)
decoder.to(device)

# Set models to training mode
gnn_interface_model.train()
encoder.train()
decoder.train()

# Select the model type (WGAN-CP or WGAN-GP)
model_name = 'WGAN-CP'
channels = 9  # Number of input channels
generator_iters = 50000  # Number of iterations for training the generator

# Instantiate the WGAN model based on the selected type
if model_name == 'WGAN-CP':
    model = WGAN_CP(channels, is_cuda, generator_iters, gnn_interface_model, encoder, decoder, device, batch_size=8)
elif model_name == 'WGAN-GP':
    model = WGAN_GP(channels, is_cuda, generator_iters, gnn_interface_model, encoder, decoder, device, batch_size=8)
else:
    print("Model type non-existing. Try again.")  # Error message if invalid model type is specified
    exit(-1)

# Load datasets to train and test loaders (data is generated using queue_datagen)
train_loader = my_gen
test_loader = my_gen

# Feature extraction is commented out for now
# feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

# Control flow for training and fine-tuning
is_train = 'True'
load_D = './discriminator.pkl'  # Path to load a pre-trained discriminator
load_G = './generator.pkl'  # Path to load a pre-trained generator
is_fineturn = 0  # Fine-tuning flag

# Check if fine-tuning flag is passed via command line arguments
if len(sys.argv) > 2:
    is_fineturn = int(sys.argv[2])

# Start training if is_train is set to True
if is_train == 'True':
    if is_fineturn:
        model.load_model()  # Load pre-trained models if fine-tuning
    model.train(train_loader, H, A1, A2, V, Atom_count)  # Train the model
# If not training, start evaluation
else:
    captions1, captions2 = model.evaluate(test_loader, load_D, load_G, H, A1, A2, V, Atom_count)  # Evaluate the model
    aa = decode_smiles(captions1, captions2)  # Decode the generated SMILES strings
    print(aa)  # Print the decoded SMILES strings

# Close the multiprocessing pool after processing is complete
multiproc.close()
