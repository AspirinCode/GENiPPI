# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
#
# This script implements conditional Generative Adversarial Networks (cGANs) for 3D molecular structure generation,
# as well as graph-based neural networks (GNNs) for learning molecular features from graph-structured data.
# The cGAN consists of a generator and a discriminator, while the GNN model handles graph data using attention mechanisms.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time
from multiprocessing import Pool
from functools import partial
import torch.utils.data

########################
# Conditional GAN (cGAN) #
########################

# Discriminator for cGAN that evaluates whether 3D voxel structures are real or fake
class discriminator(nn.Module):
    def __init__(self, nc=9, ngf=128, ndf=128, latent_variable_size=512, use_cuda=False):
        """
        Initializes the Discriminator network for cGAN. The network processes 3D voxel data
        and classifies whether it is real or generated (fake).

        Args:
            nc (int): Number of input channels (typically 9 for molecular features).
            ngf (int): Number of generator features.
            ndf (int): Number of discriminator features.
            latent_variable_size (int): Size of the latent variable.
            use_cuda (bool): Whether to use CUDA (GPU acceleration).
        """
        super(discriminator, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        
        # Convolutional layers followed by batch normalization
        self.e1 = nn.Conv3d(nc, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(32)

        self.e2 = nn.Conv3d(32, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm3d(32)

        self.e3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(64)

        self.e4 = nn.Conv3d(64, ndf * 4, 3, 2, 1)
        self.bn4 = nn.BatchNorm3d(ndf * 4)

        self.e5 = nn.Conv3d(ndf * 4, ndf * 4, 3, 2, 1)
        self.bn5 = nn.BatchNorm3d(ndf * 4)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(512 * 125, latent_variable_size)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        """
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): Input tensor representing 3D voxel data.
        
        Returns:
            torch.Tensor: The output score indicating whether the input is real or fake.
        """
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        
        h5 = h5.view(-1, 512 * 125)  # Flatten the tensor
        h6 = self.fc1(h5)
        h7 = self.fc2(h6)
        h8 = self.fc3(h7)
        
        return self.sigmoid(h8)  # Return the classification score


# Generator for cGAN that generates 3D voxel structures conditioned on molecular features
class generator(nn.Module):
    def __init__(self, nc=9, ngf=128, ndf=10, latent_variable_size=512, use_cuda=False):
        """
        Initializes the Generator network for cGAN. The network generates 3D voxel structures
        conditioned on molecular features.

        Args:
            nc (int): Number of input channels (typically 9 for molecular features).
            ngf (int): Number of generator features.
            ndf (int): Number of discriminator features.
            latent_variable_size (int): Size of the latent variable.
            use_cuda (bool): Whether to use CUDA (GPU acceleration).
        """
        super(generator, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
     
        # Transposed convolution layer for generating 3D voxel data
        self.d5 = nn.ConvTranspose3d(ndf, 32, 3, 1, padding=1, output_padding=0)
        self.bn9 = nn.BatchNorm3d(32, 1.e-3)

        # Output layer
        self.d6 = nn.Conv3d(32, nc, 3, 1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.output = nn.Tanh()

    def forward(self, x, c2, only_y):
        """
        Forward pass of the generator network.

        Args:
            x (torch.Tensor): Latent input tensor.
            c2 (torch.Tensor): Conditioning features for generation.
            only_y (torch.Tensor): Labels or conditioning input.

        Returns:
            torch.Tensor: Generated 3D voxel data.
        """
        c2 = c2.view(c2.shape[0], 35, 35, 35)  # Reshape conditioning feature
        c2_condition_f = []
        c2_negative_f = torch.zeros(c2.shape, dtype=torch.float32)

        if self.use_cuda:
            c2_negative_f = Variable(c2_negative_f.cuda())
        else:
            c2_negative_f = Variable(c2_negative_f)

        for i in range(len(only_y)):
            if only_y[i] == 0:
                c2_condition_f.append(c2_negative_f)
            elif only_y[i] == 1:
                c2_condition_f.append(c2)

        c2_condition = torch.stack(c2_condition_f, 0)  # Stack condition features

        cat_h3 = torch.cat([x, c2_condition], dim=1)  # Concatenate with latent vector

        h5 = self.leakyrelu(self.bn9(self.d5(cat_h3)))  # Pass through layers
        return self.output(self.d6(h5))  # Output the generated 3D voxel data


######################
# Encoder-Decoder RNN #
######################

# CNN-based Encoder for extracting features from 3D voxel data
class EncoderCNN(nn.Module):
    def __init__(self, in_layers):
        """
        Initializes the EncoderCNN for encoding 3D voxel data into a fixed-size vector.

        Args:
            in_layers (int): Number of input channels.
        """
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        # Construct a CNN with increasing depth
        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                out_layers *= 2  # Double the number of output layers every two layers
                layers.append(self.pool)

        layers.pop()  # Remove the last max pooling layer
        self.fc1 = nn.Linear(256, 512)  # Fully connected layer
        self.network = nn.Sequential(*layers)  # Sequential container for layers

    def forward(self, x):
        """
        Forward pass of the encoder network.

        Args:
            x (torch.Tensor): Input 3D voxel data.

        Returns:
            torch.Tensor: Encoded feature vector.
        """
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)  # Global averaging
        x = self.relu(self.fc1(x))
        return x


# LSTM-based Decoder for generating SMILES strings from the encoded feature vectors
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """
        Initializes the DecoderRNN for generating SMILES strings using an LSTM-based decoder.

        Args:
            embed_size (int): Size of the embedding vectors.
            hidden_size (int): Size of the hidden state of the LSTM.
            vocab_size (int): Size of the vocabulary for SMILES tokens.
            num_layers (int): Number of LSTM layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initializes weights for the embedding and linear layers."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """
        Forward pass of the LSTM-based decoder.

        Args:
            features (torch.Tensor): Encoded feature vectors from the EncoderCNN.
            captions (torch.Tensor): Ground-truth SMILES sequences.
            lengths (list): Lengths of each SMILES sequence.

        Returns:
            torch.Tensor: Predicted SMILES tokens.
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """
        Samples SMILES tokens using greedy search based on the feature vectors.

        Args:
            features (torch.Tensor): Encoded feature vectors from the EncoderCNN.
            states: Hidden and cell states for LSTM (default: None).

        Returns:
            list: Sampled SMILES tokens.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(105):  # Set maximum SMILES length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]  # Greedily choose the token with max probability
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """
        Samples SMILES tokens using probabilistic sampling instead of greedy search.

        Args:
            features (torch.Tensor): Encoded feature vectors from the EncoderCNN.
            states: Hidden and cell states for LSTM (default: None).

        Returns:
            tuple: Two lists of sampled SMILES tokens using probabilistic sampling.
        """
        sampled1_ids, sampled2_ids = [], []
        inputs = features.unsqueeze(1)

        # First sampling round
        for i in range(105):  # Set maximum SMILES length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)
                predicted = self.probabilistic_sample(probs)
            sampled1_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        # Second sampling round
        inputs = features.unsqueeze(1)
        for i in range(128):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)
                predicted = self.probabilistic_sample(probs)
            sampled2_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return sampled1_ids, sampled2_ids

    def probabilistic_sample(self, probs):
        """
        Probabilistically samples a token from the probability distribution.

        Args:
            probs (torch.Tensor): Softmax probabilities for the SMILES tokens.

        Returns:
            torch.Tensor: Probabilistically sampled token.
        """
        if probs.is_cuda:
            probs_np = probs.data.cpu().numpy()
        else:
            probs_np = probs.data.numpy()

        rand_num = np.random.rand(probs_np.shape[0])
        iter_sum = np.zeros((probs_np.shape[0],))
        tokens = np.zeros(probs_np.shape[0], dtype=np.int)

        for i in range(probs_np.shape[1]):
            c_element = probs_np[:, i]
            iter_sum += c_element
            valid_token = rand_num < iter_sum
            update_indices = np.logical_and(valid_token, np.logical_not(tokens.astype(np.bool)))
            tokens[update_indices] = i

        if probs.is_cuda:
            return Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
        else:
            return Variable(torch.LongTensor(tokens.astype(np.int)))


########################
# Graph Neural Network #
########################

# GNN Model that processes molecular graphs with attention-based mechanisms
class GNN_Model(nn.Module):
    def __init__(self, params):
        """
        Initializes the GNN_Model for processing graph-structured molecular data.

        Args:
            params (dict): Dictionary containing hyperparameters for the model.
        """
        super(GNN_Model, self).__init__()
        n_graph_layer = params['n_graph_layer']
        d_graph_layer = params['d_graph_layer']
        n_FC_layer = params['n_FC_layer']
        d_FC_layer = params['d_FC_layer']
        N_atom_features = params['N_atom_features']
        final_dimension = params['final_dimension']
        self.dropout_rate = params['dropout_rate']

        # Define GNN layers
        self.layers1 = [d_graph_layer for i in range(n_graph_layer + 1)]
        self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i + 1]) for i in range(len(self.layers1) - 1)])

        # Fully connected layers
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i == 0 else
                                 nn.Linear(d_FC_layer, final_dimension) if i == n_FC_layer - 1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        # Learnable parameters for GNN
        self.mu = nn.Parameter(torch.Tensor([params['initial_mu']]).float())
        self.dev = nn.Parameter(torch.Tensor([params['initial_dev']]).float())
        self.embede = nn.Linear(2 * N_atom_features, d_graph_layer, bias=False)
        self.params = params

    # Additional methods for GNN_Model are defined below
    # These methods include functions for embedding graphs, formulating adjacency matrices,
    # predicting outputs, and extracting features from graph-structured data.
    ...
    
# GAT Gate Layer for Graph Attention Network
class GAT_gate(nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        """
        Initializes the GAT_gate layer, which is a single attention layer for the Graph Attention Network (GAT).

        Args:
            n_in_feature (int): Number of input features for each node in the graph.
            n_out_feature (int): Number of output features for each node in the graph.
        """
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, 1)  # Default bias=True
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj, request_attention=False):
        """
        Forward pass of the GAT_gate layer, applying attention mechanism to graph nodes.

        Args:
            x (torch.Tensor): Node feature matrix.
            adj (torch.Tensor): Adjacency matrix representing edges between nodes.
            request_attention (bool): Whether to return attention values (default: False).

        Returns:
            torch.Tensor: Updated node features after attention is applied.
        """
        h = self.W(x)  # Linear transformation of node features
        batch_size = h.size()[0]
        N = h.size()[1]  # Number of nodes

        # Attention mechanism
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))  # Attention scores between nodes
        e = e + e.permute((0, 2, 1))  # Symmetrize the attention scores
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # Mask out non-existent edges
        attention = F.softmax(attention, dim=1)

        output_attention = attention
        attention = attention * adj  # Apply attention to edges
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))  # Update node features

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))  # Gate mechanism
        retval = coeff * x + (1 - coeff) * h_prime  # Update node features using gated mechanism

        if request_attention:
            return output_attention, retval  # Return attention values if requested
        else:
            return retval  # Return updated node features
