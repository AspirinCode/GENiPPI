# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
# 
# This script implements a Wasserstein GAN (WGAN) with Gradient Penalty (WGAN-GP)
# for generating 3D voxel data using a Generator and Discriminator model.
# It includes functions for training the GAN, evaluating models, and saving/loading weights.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')  # Switch to non-interactive mode for saving plots
sys.path.append('/home/jmwang/WorkSpace/GENiPPI/gPPMol/models_3d')  # Add custom path for models
import os
from itertools import chain
from torchvision import utils
from tensorboard_logger import Logger  # Logger for TensorBoard visualization
from tqdm import tqdm  # Progress bar for loops
import numpy as np

SAVE_PER_TIMES = 1000  # Interval for saving models during training

# Generator Model Definition
class Generator(torch.nn.Module):
    """
    Generator model for 3D voxel data generation.
    
    This model uses ConvTranspose3d layers to upsample a latent vector into a
    3D voxel grid with specified output channels.
    """
    def __init__(self, channels, is_cuda):
        """
        Initializes the generator model.

        Args:
            channels (int): Number of output channels (i.e., the depth of the voxel grid).
            is_cuda (bool): Flag to indicate whether to use GPU acceleration.
        """
        super().__init__()
        self.use_cuda = is_cuda  # Store if CUDA is being used
        
        # Generator architecture using ConvTranspose3d layers
        self.main_module = nn.Sequential(
            # First layer: ConvTranspose3d (upsample) from 10 to 1024 channels
            nn.ConvTranspose3d(in_channels=10, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=1024),  # Normalize to stabilize training
            nn.ReLU(True),  # Activation
            
            # Second layer: ConvTranspose3d from 1024 to 512 channels
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(True),

            # Third layer: ConvTranspose3d from 512 to 256 channels
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(True),

            # Final layer: Output from 256 to the desired number of channels
            nn.ConvTranspose3d(in_channels=256, out_channels=channels, kernel_size=8, stride=1, padding=0)
        )

        self.output = nn.Tanh()  # Output activation function

    def forward(self, x, c2, only_y):
        """
        Forward pass for generating a 3D voxel grid from a latent vector.

        Args:
            x (torch.Tensor): Latent vector input (noise).
            c2 (torch.Tensor): Condition tensor for conditional generation.
            only_y (list): List to control which condition to apply.

        Returns:
            torch.Tensor: Generated 3D voxel grid.
        """
        # Prepare condition tensor c2 (which will be concatenated with input x)
        c2 = c2.view(c2.shape[0], 4, 4, 4)  # Reshape c2 to match input size
        
        # Create a negative tensor with zeros
        c2_negative_f = np.zeros(c2.shape, dtype=np.float32)
        c2_negative_f = torch.Tensor(c2_negative_f)

        if self.use_cuda:
            c2_negative_f = Variable(c2_negative_f.cuda())  # Move to GPU if available
        else:
            c2_negative_f = Variable(c2_negative_f)

        # Prepare conditional input for generator based on 'only_y' values
        c2_condition_f = []
        for i in range(len(only_y)):
            if only_y[i] == 0:
                c2_condition_f.append(c2_negative_f)  # Add negative condition
            elif only_y[i] == 1:
                c2_condition_f.append(c2)  # Add actual condition

        # Stack the conditions and concatenate with input x
        c2_condition = torch.stack(c2_condition_f, 0)
        x = torch.cat([x, c2_condition], dim=1)

        # Pass through the main generator network
        x = self.main_module(x)
        return self.output(x)  # Apply final output activation


# Discriminator Model Definition
class Discriminator(torch.nn.Module):
    """
    Discriminator model for distinguishing between real and generated 3D voxel grids.

    This model uses Conv3d layers to downsample 3D voxel data and output a scalar prediction.
    """
    def __init__(self, channels):
        """
        Initializes the discriminator model.

        Args:
            channels (int): Number of input channels (depth of the voxel grid).
        """
        super().__init__()
        
        # Discriminator architecture using Conv3d layers
        self.main_module = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),  # Use instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Activation function

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        """
        Forward pass to classify a 3D voxel grid as real or generated.

        Args:
            x (torch.Tensor): Input voxel grid.

        Returns:
            torch.Tensor: Output scalar prediction.
        """
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        """
        Extracts features from the discriminator for use in feature matching or evaluation.

        Args:
            x (torch.Tensor): Input voxel grid.

        Returns:
            torch.Tensor: Flattened feature vector.
        """
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4 * 4)  # Flatten output


# WGAN with Gradient Penalty training class
class WGAN_GP(object):
    """
    WGAN with Gradient Penalty (WGAN-GP) for training a generator and discriminator to generate 3D voxel data.
    """
    def __init__(self, channels, is_cuda, generator_iters, gnn_interface_model, encoder, decoder, device, batch_size=64):
        """
        Initializes the WGAN-GP model.

        Args:
            channels (int): Number of input/output channels for generator and discriminator.
            is_cuda (bool): Flag to enable CUDA.
            generator_iters (int): Number of generator iterations during training.
            gnn_interface_model: GNN model for additional feature generation.
            encoder: Encoder model for captions.
            decoder: Decoder model for captions.
            device (torch.device): Device to run the training (CPU/GPU).
            batch_size (int): Batch size for training.
        """
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(channels, is_cuda)  # Initialize Generator
        self.D = Discriminator(channels)  # Initialize Discriminator
        self.gnn_interface_model = gnn_interface_model  # GNN interface model (custom)
        self.encoder = encoder  # Encoder model for captions
        self.decoder = decoder  # Decoder model for captions
        self.device = device  # Device (CPU/GPU)
        self.C = channels  # Number of channels

        self.check_cuda(is_cuda)  # Set up CUDA if available

        # WGAN-GP hyperparameters and optimizer setup
        self.learning_rate = 1e-4  # Learning rate for both generator and discriminator
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = batch_size  # Batch size

        # Optimizers for generator and discriminator
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()) + list(self.gnn_interface_model.parameters()), 
                                               lr=self.learning_rate)

        # Logger for TensorBoard visualization
        self.logger = Logger('./logs')
        self.logger.writer.flush()

        self.generator_iters = generator_iters  # Number of generator iterations
        self.critic_iter = 50  # Number of critic iterations per generator update
        self.lambda_term = 10  # Lambda for gradient penalty

    def get_torch_variable(self, arg):
        """
        Utility function to return torch Variables (with CUDA support if available).

        Args:
            arg: Tensor to be converted to Variable.

        Returns:
            torch.Variable: Converted Variable with GPU support if enabled.
        """
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        """
        Checks and enables CUDA if available.

        Args:
            cuda_flag (bool): Flag indicating whether CUDA should be enabled.

        Returns:
            None
        """
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print(f"CUDA enabled: {self.cuda}")
        else:
            self.cuda = False

    def train(self, train_loader, H, A1, A2, V, Atom_count):
        """
        Trains the WGAN-GP model.

        Args:
            train_loader: DataLoader for training.
            H, A1, A2, V, Atom_count: GNN input tensors.

        Returns:
            None
        """
        self.t_begin = t.time()  # Start time for training

        # Infinite data loader
        self.data = self.get_infinite_batches(train_loader)

        # Labels for real and fake images
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        # Loss function for captions
        criterion = nn.CrossEntropyLoss()
        caption_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

        caption_start = 10000  # Start training captions after certain iterations

        for g_iter in range(self.generator_iters):
            # Update discriminator multiple times per generator update
            for p in self.D.parameters():
                p.requires_grad = True

            # Train discriminator (critic) multiple times
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Get batch data
                (images, condition, y, only_y, caption, lengths) = self.data.__next__()

                if images.size()[0] != self.batch_size:
                    continue

                # Process GNN input and get conditional output
                H_new = Variable(H.to(self.device))
                A1_new = Variable(A1.to(self.device))
                A2_new = Variable(A2.to(self.device))
                V_new = Variable(V.to(self.device))
                c_output2 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), self.device)

                images = self.get_torch_variable(images)

                # Train discriminator on real images
                d_loss_real = self.D(images).mean()
                d_loss_real.backward(mone)

                # Train discriminator on fake images generated by the generator
                z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
                fake_images = self.G(z, c_output2.detach(), only_y)
                d_loss_fake = self.D(fake_images).mean()
                d_loss_fake.backward(one)

                # Apply gradient penalty (WGAN-GP specific)
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

                print(f'Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad()

            # Generate fake images and update generator
            z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
            fake_images = self.G(z, c_output2.detach(), only_y)
            g_loss = self.D(fake_images).mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()

            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')

            # Train autoencoder for captions if applicable
            if g_iter >= caption_start:
                recon_batch = Variable(recon_batch.to(self.device))
                captions = Variable(caption.to(self.device))
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                self.decoder.zero_grad()
                self.encoder.zero_grad()
                features = self.encoder(recon_batch)
                outputs = self.decoder(features, captions, lengths)
                cap_loss = criterion(outputs, targets)
                cap_loss.backward()
                caption_optimizer.step()

                print("cap_loss:", cap_loss.data.item())

            # Save models at specific intervals
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                time_elapsed = t.time() - self.t_begin
                print(f"Generator iter: {g_iter}, Time {time_elapsed}")

                # TensorBoard logging
                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_cost.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data
                }
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value.cpu(), g_iter + 1)

        self.t_end = t.time()
        print(f'Time of training: {self.t_end - self.t_begin}')
        self.save_model()

    # Function to evaluate trained model
    def evaluate(self, test_loader, D_model_path, G_model_path, H, A1, A2, V, Atom_count, probab=True):
        """
        Evaluates the WGAN-GP model.

        Args:
            test_loader: DataLoader for evaluation.
            D_model_path: Path to discriminator weights.
            G_model_path: Path to generator weights.
            H, A1, A2, V, Atom_count: GNN input tensors.
            probab (bool): Whether to use probabilistic sampling.

        Returns:
            captions1, captions2: Generated captions (if applicable).
        """
        self.load_model(D_model_path, G_model_path)

        H_new = Variable(H.to(self.device))
        A1_new = Variable(A1.to(self.device))
        A2_new = Variable(A2.to(self.device))
        V_new = Variable(V.to(self.device))

        c_output2 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), self.device)
        z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
        only_y = [1 for i in range(self.batch_size)]
        recon_batch = self.G(z, c_output2, only_y)
        features = self.encoder(recon_batch)
        if probab:
            captions1, captions2 = self.decoder.sample_prob(features)
        else:
            captions = self.decoder.sample(features)

        captions1 = torch.stack(captions1, 1)
        captions2 = torch.stack(captions2, 1)
        if self.cuda:
            captions1 = captions1.cpu().data.numpy()
            captions2 = captions2.cpu().data.numpy()
        else:
            captions1 = captions1.data.numpy()
            captions2 = captions2.data.numpy()
        return captions1, captions2

    # Gradient penalty calculation for WGAN-GP
    def calculate_gradient_penalty(self, real_images, fake_images):
        """
        Calculates gradient penalty for WGAN-GP.

        Args:
            real_images: Real images batch.
            fake_images: Generated fake images batch.

        Returns:
            grad_penalty: Computed gradient penalty term.
        """
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)

        # Interpolate between real and fake images
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = Variable(interpolated, requires_grad=True)

        # Get discriminator output for interpolated images
        prob_interpolated = self.D(interpolated)

        # Compute gradients of interpolated images
        gradients = autograd.grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(prob_interpolated.size()),
            create_graph=True, 
            retain_graph=True
        )[0]

        # Compute gradient penalty term
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    # Function to save models
    def save_model(self):
        """
        Saves the generator, discriminator, GNN interface, encoder, and decoder models to files.
        """
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        torch.save(self.gnn_interface_model.state_dict(), './gnn_interface_model.pkl')
        torch.save(self.encoder.state_dict(), './encoder.pkl')
        torch.save(self.decoder.state_dict(), './decoder.pkl')
        print('Models saved to ./generator.pkl & ./discriminator.pkl & ./gnn_interface_model.pkl & ./encoder.pkl & ./decoder.pkl')

    # Function to load saved models
    def load_model(self, D_model_filename='./discriminator.pkl', G_model_filename='./generator.pkl',
                  gnn_model_filename='./gnn_interface_model.pkl', encoder_file='./encoder.pkl', decoder_file='./decoder.pkl'):
        """
        Loads the saved generator, discriminator, GNN interface, encoder, and decoder models.

        Args:
            D_model_filename (str): Filename of the saved discriminator model.
            G_model_filename (str): Filename of the saved generator model.
            gnn_model_filename (str): Filename of the saved GNN interface model.
            encoder_file (str): Filename of the saved encoder model.
            decoder_file (str): Filename of the saved decoder model.

        Returns:
            None
        """
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))

        self.gnn_interface_model.load_state_dict(torch.load(gnn_model_filename))
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))

    # Infinite batch loader for continuous training
    def get_infinite_batches(self, data_loader):
        """
        Generates an infinite stream of batches from a DataLoader.

        Args:
            data_loader: The DataLoader to get batches from.

        Yields:
            A batch of data.
        """
        while True:
            for i, data in tqdm(enumerate(data_loader)):
                yield data
