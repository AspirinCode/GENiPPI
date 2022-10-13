import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')
sys.path.append('/home/jmwang/WorkSpace/GENiPPI/gPPMol/models_3d')
import os
from itertools import chain
from torchvision import utils
from tensorboard_logger import Logger
from tqdm import tqdm
import numpy as np

SAVE_PER_TIMES = 1000

class Generator(torch.nn.Module):
    def __init__(self, channels,is_cuda):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.use_cuda = is_cuda
        self.main_module = nn.Sequential(
            # Z latent vector 100
            ###Dout =(Din−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            #(4-1)*1-2*0+1*(4-1)+0+1=7
            nn.ConvTranspose3d(in_channels=10, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            #(7-1)*2-2*1+1*(4-1)+0+1=14
            nn.ConvTranspose3d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            #(14-1)*2-2*1+1*(4-1)+0+1=28
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            #(28-1)*1-2*0+1*(8-1)+0+1=35
            nn.ConvTranspose3d(in_channels=256, out_channels=channels, kernel_size=8, stride=1, padding=0))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x,c2,only_y):
        c2 = c2.view(c2.shape[0],4,4,4)
        c2_condition_f = []
        c2_negative_f = np.zeros(c2.shape, dtype=np.float32)
        c2_negative_f = torch.Tensor(c2_negative_f)

        if self.use_cuda:
            c2_negative_f = Variable(c2_negative_f.cuda())
        else:
            c2_negative_f = Variable(c2_negative_f)

        for i in range(len(only_y)):
            if only_y[i] == 0:
                c2_condition_f.append(c2_negative_f)
            elif only_y[i] == 1:
                c2_condition_f.append(c2)

        c2_condition = torch.stack(c2_condition_f, 0) #16 1 4 4 4
        x = torch.cat([x, c2_condition], dim=1) #16 10 4 4 4 

        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv3d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv3d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4*4)


class WGAN_GP(object):
    def __init__(self, channels,is_cuda,generator_iters,gnn_interface_model,encoder,decoder,device,batch_size=64):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(channels,is_cuda)
        self.D = Discriminator(channels)
        self.gnn_interface_model = gnn_interface_model
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.C = channels

        # Check if cuda is available
        self.check_cuda(is_cuda)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = batch_size

        # WGAN_gradient penalty uses ADAM
        #self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        #self.g_optimizer = optim.Adam(list(self.G.parameters()) + list(self.gnn_interface_model.parameters()), 
        #    lr=self.learning_rate, betas=(self.b1, self.b2))

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(list(self.G.parameters()) + list(self.gnn_interface_model.parameters()), 
            lr=self.learning_rate)


        # Set the logger
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = generator_iters
        self.critic_iter = 50
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

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


    def train(self, train_loader,H, A1, A2, V, Atom_count):
        self.t_begin = t.time()

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)        

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        # Caption optimizer
        criterion = nn.CrossEntropyLoss() #ignore_index = 0
        caption_params = list(self.decoder.parameters()) + list(self.encoder.parameters())
        caption_optimizer = torch.optim.Adam(caption_params, lr=0.001)

        caption_start = 10000

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            c_output2 = None
            only_y = None

            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                (images, condition, y, only_y, caption, lengths) = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    #print(images.size()[0],self.batch_size)
                    continue

                #print("only_y:",only_y)

                H_new  = Variable(H.to(self.device))
                A1_new = Variable(A1.to(self.device))
                A2_new = Variable(A2.to(self.device))
                V_new  = Variable(V.to(self.device))
                c_output2 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), self.device)
                #print("c_output2:",c_output2.shape)
                
                images = self.get_torch_variable(images)
                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))

                #print(z.shape,c_output2.shape,len(only_y))

                fake_images = self.G(z,c_output2.detach(),only_y)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                #print(images.shape,fake_images.shape) #torch.Size([8, 9, 35, 35, 35]) torch.Size([8, 9, 35, 35, 35])

                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4, 4, 4))
            fake_images = self.G(z,c_output2.detach(),only_y)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')

            #==========END=============================    
            recon_batch = self.G(z.detach(),c_output2.detach(),only_y)   

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

                print("cap_loss:", cap_loss.data.item(),i)
                caption_optimizer.step()

            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()

                # Testing
                time = t.time() - self.t_begin
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
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

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path,H, A1, A2, V, Atom_count,probab=True):
        self.load_model(D_model_path, G_model_path)
        
        H_new  = Variable(H.to(self.device))
        A1_new = Variable(A1.to(self.device))
        A2_new = Variable(A2.to(self.device))
        V_new  = Variable(V.to(self.device))

        c_output2 = self.gnn_interface_model.train_model((H_new, A1_new, A2_new, V_new, Atom_count), self.device)
        z = self.get_torch_variable(torch.randn(self.batch_size, 9, 4,4,4))
        only_y = [1 for i in range(self.batch_size)]
        recon_batch = self.G(z,c_output2,only_y)
        features = self.encoder(recon_batch)
        if probab:
            captions1,captions2 = self.decoder.sample_prob(features)
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
        return captions1,captions2


    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')

        torch.save(self.gnn_interface_model.state_dict(), './gnn_interface_model.pkl')
        torch.save(self.encoder.state_dict(), './encoder.pkl')
        torch.save(self.decoder.state_dict(), './decoder.pkl')

        print('Models save to ./generator.pkl & ./discriminator.pkl &./gnn_interface_model.pkl &./encoder.pkl & ./decoder.pkl')

    def load_model(self, D_model_filename= './discriminator.pkl', G_model_filename='./generator.pkl',
        gnn_model_filename='./gnn_interface_model.pkl',encoder_file = './encoder.pkl',decoder_file = './decoder.pkl'
        ):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))

        self.gnn_interface_model.load_state_dict(torch.load(gnn_model_filename))
        self.encoder.load_state_dict(torch.load(encoder_file))
        self.decoder.load_state_dict(torch.load(decoder_file))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, data  in tqdm(enumerate(data_loader)):
                yield data


 