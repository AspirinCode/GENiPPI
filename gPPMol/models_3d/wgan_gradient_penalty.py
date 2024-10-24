import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')
sys.path.append('/home/jmwang/WorkSpace/GENiPPI/gPPMol/models_3d')  # Custom path for loading models
import os
from itertools import chain
from torchvision import utils
from tensorboard_logger import Logger  # Logger for tensorboard visualization
from tqdm import tqdm  # Progress bar for loops
import numpy as np

SAVE_PER_TIMES = 1000  # Interval for saving model

# Define Generator model
class Generator(torch.nn.Module):
    def __init__(self, channels, is_cuda):
        super().__init__()
        self.use_cuda = is_cuda  # Store if CUDA is being used
        
        # Generator architecture using ConvTranspose3d layers
        # The input to the generator is a latent vector of size 100 (here it's reduced to 10 in_channels)
        self.main_module = nn.Sequential(
            # First layer: ConvTranspose3d (upsample) from 10 to 1024 channels
            nn.ConvTranspose3d(in_channels=10, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=1024),  # Normalize output to stabilize training
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


# Define Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # Discriminator architecture using Conv3d layers
        self.main_module = nn.Sequential(
            # First layer: Conv3d from 'channels' to 256 with InstanceNorm and LeakyReLU
            nn.Conv3d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),  # Use instance normalization
            nn.LeakyReLU(0.2, inplace=True),  # Activation function

            # Second layer: Conv3d from 256 to 512 with InstanceNorm and LeakyReLU
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer: Conv3d from 512 to 1024 with InstanceNorm and LeakyReLU
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.output = nn.Sequential(
            # Final layer: Conv3d to reduce to a single scalar value (not a probability since we use WGAN-GP)
            nn.Conv3d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        # Pass input through the main module and return the output
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Extract features from the main module and flatten the output for further use (e.g., GAN features)
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4 * 4)  # Flatten output


# Define WGAN with Gradient Penalty training class
class WGAN_GP(object):
    def __init__(self, channels, is_cuda, generator_iters, gnn_interface_model, encoder, decoder, device, batch_size=64):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(channels, is_cuda)  # Initialize Generator
        self.D = Discriminator(channels)  # Initialize Discriminator
        self.gnn_interface_model = gnn_interface_model  # GNN interface model (custom)
        self.encoder = encoder  # Encoder model for captions
        self.decoder = decoder  # Decoder model for captions
        self.device = device  # Device (CPU/GPU)
        self.C = channels  # Number of channels

        # Set up CUDA if available
        self.check_cuda(is_cuda)

        # WGAN-GP hyperparameters and optimizer setup
        self.learning_rate = 1e-4  # Learning rate for both generator and discriminator
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = batch_size  # Batch size

        # RMSProp optimizers for both generator and discriminator
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()) + list(self.gnn_interface_model.parameters()), 
                                               lr=self.learning_rate)

        # Logger for TensorBoard
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = generator_iters  # Number of iterations to run generator
        self.critic_iter = 50  # Number of iterations to update critic (discriminator) before generator
        self.lambda_term = 10  # Lambda for gradient penalty (WGAN-GP specific)

    # Utility function to return torch Variables (with CUDA support if available)
    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    # Check and enable CUDA if available
    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    # Training function
    def train(self, train_loader, H, A1, A2, V, Atom_count):
        self.t_begin = t.time()  # Start time of training

        # Infinite data generator
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)  # Label for real images
        mone = one * -1  # Label for fake images

        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        # Loss function for captions (cross-entropy)
        criterion = nn.CrossEntropyLoss()  # Ignore padding index in captions if necessary
        caption_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

        caption_start = 10000  # Start training captions after certain iterations

        for g_iter in range(self.generator_iters):
            # Update discriminator (critic) multiple times per generator update
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0  # Wasserstein distance for WGAN-GP

            # Train the discriminator multiple times (critic iterations)
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Get a batch of training data
                (images, condition, y, only_y, caption, lengths) = self.data.__next__()

                # Ensure full batch size for training
                if (images.size()[0] != self.batch_size):
                    continue

                # GNN model interface for conditioning
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

                # Compute and apply gradient penalty (WGAN-GP specific)
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                # Update discriminator loss
                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

                print(f'Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Update generator after discriminator
            for p in self.D.parameters():
                p.requires_grad = False  # Avoid computation for discriminator gradients

            self.G.zero_grad()

            # Generate fake images and update generator
            z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
            fake_images = self.G(z, c_output2.detach(), only_y)
            g_loss = self.D(fake_images).mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()

            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')

            # Autoencoder training (for captions)
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

            # Save model at specific intervals
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()

                time = t.time() - self.t_begin
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

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
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the final model after training
        self.save_model()

    # Function to evaluate the trained model (optional)
    def evaluate(self, test_loader, D_model_path, G_model_path, H, A1, A2, V, Atom_count, probab=True):
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

    # Gradient penalty for WGAN-GP
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)

        # Interpolation of real and fake images
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = Variable(interpolated, requires_grad=True)

        # Compute discriminator output
        prob_interpolated = self.D(interpolated)

        # Compute gradients of interpolated images
        gradients = autograd.grad(
            outputs=prob_interpolated, 
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(prob_interpolated.size()),
            create_graph=True, 
            retain_graph=True
        )[0]

        # Gradient penalty term
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    # Function to save model checkpoints
    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        torch.save(self.gnn_interface_model.state_dict(), './gnn_interface_model.pkl')
        torch.save(self.encoder.state_dict(), './encoder.pkl')
        torch.save(self.decoder.state_dict(), './decoder.pkl')
        print('Models saved to ./generator.pkl & ./discriminator.pkl & ./gnn_interface_model.pkl & ./encoder.pkl & ./decoder.pkl')

    # Function to load models from checkpoints
    def load_model(self, D_model_filename='./discriminator.pkl', G_model_filename='./generator.pkl',
                  gnn_model_filename='./gnn_interface_model.pkl', encoder_file='./encoder.pkl', decoder_file='./decoder.pkl'):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))

        self.gnn_interface_model.load_state_dict(torch.load(gnn_model_filename))
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))

    # Infinite batch loader
    def get_infinite_batches(self, data_loader):
        while True:
            for i, data in tqdm(enumerate(data_loader)):
                yield data
