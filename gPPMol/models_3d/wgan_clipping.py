# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license
# 
# This script implements a Wasserstein GAN (WGAN) with weight clipping (WGAN-CP) 
# for generating 3D voxel data using a Generator and Discriminator model.
# It includes functions for training the GAN, evaluating models, and saving/loading weights.

import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')  # Switch to non-interactive mode for saving plots
sys.path.append('/home/jmwang/WorkSpace/GENiPPI/gPPMol/models_3d')  # Add custom path for models
import os
from torchvision import utils
from tensorboard_logger import Logger  # Logger for TensorBoard visualization
from tqdm import tqdm  # Progress bar for loops
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

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
        self.use_cuda = is_cuda  # Use CUDA if available

        # Define the generator architecture (ConvTranspose3d for upsampling)
        self.main_module = nn.Sequential(
            nn.ConvTranspose3d(in_channels=10, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=1024),
            nn.ReLU(True),
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(True),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(True),
            nn.ConvTranspose3d(in_channels=256, out_channels=channels, kernel_size=8, stride=1, padding=0)
        )

        # Output activation: Tanh to normalize voxel values between -1 and 1
        self.output = nn.Tanh()

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
        c2 = c2.view(c2.shape[0], 4, 4, 4)

        c2_condition_f = []
        c2_negative_f = np.zeros(c2.shape, dtype=np.float32)
        c2_negative_f = torch.Tensor(c2_negative_f)

        if self.use_cuda:
            c2_negative_f = Variable(c2_negative_f.cuda())
            c2 = Variable(c2.cuda())
        else:
            c2_negative_f = Variable(c2_negative_f)
            c2 = Variable(c2)

        for i in range(len(only_y)):
            if only_y[i] == 0:
                c2_condition_f.append(c2_negative_f)
            elif only_y[i] == 1:
                c2_condition_f.append(c2)

        c2_condition = torch.stack(c2_condition_f, 0)

        if c2_condition.shape[0] != x.shape[0]:
            print("Batch size error", c2_condition.shape, x.shape)

        cat_h3 = torch.cat([x, c2_condition], dim=1)
        cat_h3 = self.main_module(cat_h3)
        return self.output(cat_h3)


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

        # Define the discriminator architecture (Conv3d for downsampling)
        self.main_module = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final output layer: Single value prediction
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
        return x.view(-1, 1024 * 4 * 4 * 4)


# WGAN with Clipping (WGAN-CP)
class WGAN_CP(object):
    """
    WGAN with weight clipping for training a generator and discriminator to generate 3D voxel data.
    """
    def __init__(self, channels, is_cuda, generator_iters, gnn_interface_model, encoder, decoder, device, batch_size=64):
        """
        Initializes the WGAN-CP model.

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
        print("WGAN_CP initialized.")
        self.G = Generator(channels, is_cuda)  # Generator model
        self.D = Discriminator(channels)  # Discriminator model
        self.gnn_interface_model = gnn_interface_model  # GNN interface model
        self.encoder = encoder  # Encoder model
        self.decoder = decoder  # Decoder model
        self.device = device  # Device to use (CPU/GPU)
        self.C = channels  # Number of input/output channels

        self.check_cuda(is_cuda)  # Check if CUDA is enabled

        # WGAN values and optimizers
        self.learning_rate = 0.00005
        self.batch_size = batch_size
        self.weight_cliping_limit = 0.01

        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()) + list(self.gnn_interface_model.parameters()),
                                               lr=self.learning_rate)

        # Logger for TensorBoard visualization
        self.logger = Logger('./logs')
        self.logger.writer.flush()

        self.generator_iters = generator_iters  # Number of generator iterations
        self.critic_iter = 30  # Number of critic iterations per generator update

    def get_torch_variable(self, arg):
        """
        Converts a tensor to a torch Variable with CUDA support if enabled.

        Args:
            arg (torch.Tensor): Input tensor.

        Returns:
            torch.Variable: Torch variable with or without CUDA.
        """
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        """
        Checks if CUDA is enabled and moves models to GPU if available.

        Args:
            cuda_flag (bool): Whether to enable CUDA.
        """
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
        Trains the WGAN-CP model.

        Args:
            train_loader: DataLoader for training.
            H, A1, A2, V, Atom_count: GNN input tensors.

        Returns:
            None
        """
        self.t_begin = t.time()

        # Infinite data loader to loop through training data
        self.data = self.get_infinite_batches(train_loader)

        # Labels for real and fake images
        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        # Loss function for captions (optional)
        criterion = nn.CrossEntropyLoss()
        caption_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

        caption_start = 1  # Start caption training after 1st generator iteration

        for g_iter in range(self.generator_iters):

            # Discriminator update: multiple updates per generator iteration
            for p in self.D.parameters():
                p.requires_grad = True

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Clamp weights to enforce Lipschitz continuity (WGAN-CP)
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                (images, condition, y, only_y, caption, lengths) = self.data.__next__()
                if (images.size()[0] != self.batch_size):
                    continue

                H_new = Variable(H.to(self.device))
                A1_new = Variable(A1.to(self.device))
                A2_new = Variable(A2.to(self.device))
                V_new = Variable(V.to(self.device))

                c_output2 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), self.device)
                images = self.get_torch_variable(images)

                # Train discriminator on real images
                d_loss_real = self.D(images).mean(0).view(1)
                d_loss_real.backward(one)

                # Train discriminator on fake images generated by the generator
                z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
                fake_images = self.G(z, c_output2.detach(), only_y)
                d_loss_fake = self.D(fake_images).mean(0).view(1)
                d_loss_fake.backward(mone)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

                print(f'Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake.data}, loss_real: {d_loss_real.data}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad()

            # Train generator
            z = self.get_torch_variable(torch.randn(len(only_y), 9, 4, 4, 4))
            fake_images = self.G(z, c_output2.detach(), only_y)
            g_loss = self.D(fake_images).mean().mean(0).view(1)
            g_loss.backward(one)
            g_cost = -g_loss
            self.g_optimizer.step()

            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.data}')

            recon_batch = self.G(z.detach(), c_output2.detach(), only_y)
            if g_iter >= caption_start:  # Start by autoencoder optimization
                recon_batch = Variable(recon_batch.to(self.device))
                captions = Variable(caption.to(self.device))
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                self.decoder.zero_grad()
                self.encoder.zero_grad()
                features = self.encoder(recon_batch)

                outputs = self.decoder(features, captions, lengths)
                cap_loss = criterion(outputs, targets)
                cap_loss.backward()

                print("cap_loss:", cap_loss.data.item(), g_iter)
                caption_optimizer.step()

            if g_iter % SAVE_PER_TIMES == 0:
                self.save_model()
                time_elapsed = t.time() - self.t_begin
                print(f"Generator iter: {g_iter}, Time {time_elapsed}")

                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_cost.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data
                }
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value.mean().cpu(), g_iter + 1)

        self.t_end = t.time()
        print(f'Time of training: {self.t_end - self.t_begin}')
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path, H, A1, A2, V, Atom_count, probab=True):
        """
        Evaluates the WGAN-CP model.

        Args:
            test_loader: DataLoader for evaluation.
            D_model_path: Path to the discriminator model weights.
            G_model_path: Path to the generator model weights.
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

    def load_model(self, D_model_filename='./discriminator.pkl', G_model_filename='./generator.pkl',
                  gnn_model_filename='./gnn_interface_model.pkl', encoder_file='./encoder.pkl', decoder_file='./decoder.pkl'):
        """
        Loads the generator, discriminator, GNN interface, encoder, and decoder models from files.
        
        Args:
            D_model_filename: Path to the discriminator model file.
            G_model_filename: Path to the generator model file.
            gnn_model_filename: Path to the GNN interface model file.
            encoder_file: Path to the encoder model file.
            decoder_file: Path to the decoder model file.

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

    def get_infinite_batches(self, data_loader):
        """
        Returns an infinite loop of the DataLoader.

        Args:
            data_loader: PyTorch DataLoader.

        Yields:
            Next batch of data.
        """
        while True:
            for i, data in tqdm(enumerate(data_loader)):
                yield data
