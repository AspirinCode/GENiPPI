import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')  # Switching backend to non-interactive mode for saving plots
sys.path.append('/home/jmwang/WorkSpace/GENiPPI/gPPMol/models_3d')  # Custom path for additional models
import os
from torchvision import utils
from tensorboard_logger import Logger  # Logger for TensorBoard visualization
from tqdm import tqdm  # Progress bar for loops
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SAVE_PER_TIMES = 1000  # Interval for saving models during training

# Generator Model Definition
class Generator(torch.nn.Module):
    def __init__(self, channels, is_cuda):
        super().__init__()
        self.use_cuda = is_cuda  # Check if CUDA is enabled

        # Define the generator architecture (ConvTranspose3d for upsampling)
        self.main_module = nn.Sequential(
            # First layer: Upsample from latent vector (in_channels=10) to 1024 channels
            nn.ConvTranspose3d(in_channels=10, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=1024),  # Normalize to stabilize training
            nn.ReLU(True),  # Activation function

            # Second layer: Upsample from 1024 to 512 channels
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(True),

            # Third layer: Upsample from 512 to 256 channels
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(True),

            # Final layer: Output the generated image with specified number of channels
            nn.ConvTranspose3d(in_channels=256, out_channels=channels, kernel_size=8, stride=1, padding=0)
        )

        # Output activation: Tanh to normalize pixel values between -1 and 1
        self.output = nn.Tanh()

    def forward(self, x, c2, only_y):
        # Reshape c2 condition to match input dimensions
        c2 = c2.view(c2.shape[0], 4, 4, 4)

        # Generate a condition tensor (c2) and a negative tensor (zeros) for conditioning
        c2_condition_f = []
        c2_negative_f = np.zeros(c2.shape, dtype=np.float32)
        c2_negative_f = torch.Tensor(c2_negative_f)

        if self.use_cuda:
            c2_negative_f = Variable(c2_negative_f.cuda())
            c2 = Variable(c2.cuda())
        else:
            c2_negative_f = Variable(c2_negative_f)
            c2 = Variable(c2)

        # Conditional input logic: Append negative or positive condition based on 'only_y'
        for i in range(len(only_y)):
            if only_y[i] == 0:
                c2_condition_f.append(c2_negative_f)
            elif only_y[i] == 1:
                c2_condition_f.append(c2)

        # Stack condition along the batch dimension and concatenate with input x
        c2_condition = torch.stack(c2_condition_f, 0)

        if c2_condition.shape[0] != x.shape[0]:
            print("batch_size error", c2_condition.shape, x.shape)

        # Concatenate condition and latent vector before passing through the network
        try:
            cat_h3 = torch.cat([x, c2_condition], dim=1)
        except:
            print(c2_condition.shape, x.shape, len(only_y))

        # Pass concatenated input through generator network
        cat_h3 = self.main_module(cat_h3)
        return self.output(cat_h3)  # Output the generated image


# Discriminator Model Definition
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Define the discriminator architecture (Conv3d for downsampling)
        self.main_module = nn.Sequential(
            # First layer: Downsample input image from 'channels' to 256
            nn.Conv3d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),  # Normalize to stabilize training
            nn.LeakyReLU(0.2, inplace=True),  # Activation function

            # Second layer: Downsample from 256 to 512 channels
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer: Downsample from 512 to 1024 channels
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final output layer: Single value prediction (no sigmoid since WGAN doesn't use probabilities)
        self.output = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        # Pass input through the discriminator network
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Extract features for additional usage (e.g., GAN feature matching)
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4 * 4)  # Flatten the output


# WGAN Model with Clipping (WGAN-CP)
class WGAN_CP(object):
    def __init__(self, channels, is_cuda, generator_iters, gnn_interface_model, encoder, decoder, device, batch_size=64):
        print("WGAN_CP init model.")
        self.G = Generator(channels, is_cuda)  # Initialize Generator
        self.D = Discriminator(channels)  # Initialize Discriminator
        self.gnn_interface_model = gnn_interface_model  # GNN interface model (custom)
        self.encoder = encoder  # Encoder for captions (optional)
        self.decoder = decoder  # Decoder for captions (optional)
        self.device = device  # Device (CPU or GPU)
        self.C = channels  # Number of image channels

        # Check if CUDA is enabled and move models to GPU if available
        self.check_cuda(is_cuda)

        # WGAN hyperparameters
        self.learning_rate = 0.00005  # Learning rate for RMSprop optimizers
        self.batch_size = batch_size  # Batch size for training
        self.weight_cliping_limit = 0.01  # Clipping threshold for weights in WGAN-CP

        # Optimizers for discriminator and generator (using RMSprop)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()) + list(self.gnn_interface_model.parameters()), 
                                               lr=self.learning_rate)

        # Logger for TensorBoard visualization
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        # Iterations for generator and critic (discriminator)
        self.generator_iters = generator_iters
        self.critic_iter = 30  # Number of discriminator updates per generator update

    # Utility function to convert tensor to a torch Variable (with CUDA support if enabled)
    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    # Check if CUDA is enabled and move models to GPU if available
    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    # Training function for WGAN-CP
    def train(self, train_loader, H, A1, A2, V, Atom_count):
        self.t_begin = t.time()  # Start time of training

        # Infinite data loader to loop through training data
        self.data = self.get_infinite_batches(train_loader)

        # Labels for real and fake images
        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        # Caption optimizer for text-based tasks (optional)
        criterion = nn.CrossEntropyLoss()
        caption_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

        caption_start = 1  # Start caption training after first generator iteration

        for g_iter in range(self.generator_iters):

            # Update the discriminator multiple times before generator update
            for p in self.D.parameters():
                p.requires_grad = True

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Clamp the discriminator weights to enforce Lipschitz continuity (WGAN-CP)
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                # Get a batch of training data
                (images, condition, y, only_y, caption, lengths) = self.data.__next__()
                if (images.size()[0] != self.batch_size):
                    continue

                # Prepare GNN input
                H_new = Variable(H.to(self.device))
                A1_new = Variable(A1.to(self.device))
                A2_new = Variable(A2.to(self.device))
                V_new = Variable(V.to(self.device))

                # GNN output
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

                # Update discriminator
                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

                print(f'Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake.data}, loss_real: {d_loss_real.data}')

            # Update generator after discriminator updates
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad()

            # Generate fake images and update generator
            z = self.get_torch_variable(torch.randn(len(only_y), 9, 4, 4, 4))
            fake_images = self.G(z, c_output2.detach(), only_y)
            g_loss = self.D(fake_images).mean().mean(0).view(1)
            g_loss.backward(one)
            g_cost = -g_loss
            self.g_optimizer.step()

            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.data}')

            # Optional: Train captions autoencoder
            recon_batch = self.G(z.detach(), c_output2.detach(), only_y)
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

                print("cap_loss:", cap_loss.data.item(), g_iter)
                caption_optimizer.step()

            # Save models and log information every 1000 iterations
            if g_iter % SAVE_PER_TIMES == 0:
                self.save_model()

                # Calculate time elapsed and log
                time = t.time() - self.t_begin
                print(f"Generator iter: {g_iter}, Time {time}")

                # TensorBoard logging for scalar values
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

        # Save the final trained models
        self.save_model()

    # Evaluation function (for generating captions)
    def evaluate(self, test_loader, D_model_path, G_model_path, H, A1, A2, V, Atom_count, probab=True):
        self.load_model(D_model_path, G_model_path)

        H_new = Variable(H.to(self.device))
        A1_new = Variable(A1.to(self.device))
        A2_new = Variable(A2.to(self.device))
        V_new = Variable(V.to(self.device))

        # GNN output
        c_output2 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), self.device)

        z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
        only_y = [1 for i in range(self.batch_size)]

        # Generate images using the trained generator
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

    # Save the models to files
    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        torch.save(self.gnn_interface_model.state_dict(), './gnn_interface_model.pkl')
        torch.save(self.encoder.state_dict(), './encoder.pkl')
        torch.save(self.decoder.state_dict(), './decoder.pkl')

        print('Models saved to ./generator.pkl & ./discriminator.pkl & ./gnn_interface_model.pkl & ./encoder.pkl & ./decoder.pkl')

    # Load the models from files
    def load_model(self, D_model_filename='./discriminator.pkl', G_model_filename='./generator.pkl',
                  gnn_model_filename='./gnn_interface_model.pkl', encoder_file='./encoder.pkl', decoder_file='./decoder.pkl'):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))

        self.gnn_interface_model.load_state_dict(torch.load(gnn_model_filename))
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))

    # Infinite batch loader (to loop through dataset indefinitely)
    def get_infinite_batches(self, data_loader):
        while True:
            for i, data in tqdm(enumerate(data_loader)):
                yield data
