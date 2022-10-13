# Copyright (C) 2019 Computational Science Lab, UPF <http://www.compscience.org/>
# Copying and distribution is allowed under AGPLv3 license

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
# cGAN #
########################
class discriminator(nn.Module):
    def __init__(self, nc=9, ngf=128, ndf=128, latent_variable_size=512, use_cuda=False):
        super(discriminator, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        
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

        self.fc1 = nn.Linear(512 * 125, latent_variable_size)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        #print('h5:',h5.shape) #h5: torch.Size([4, 512, 5, 5, 5])
        h5 = h5.view(-1, 512 * 125)
        h6 = self.fc1(h5)
        h7 = self.fc2(h6)
        h8 = self.fc3(h7)
        #print('h8.shape', h8.shape)
        return self.sigmoid(h8)
    
class generator(nn.Module):
    def __init__(self, nc=9, ngf=128, ndf=10, latent_variable_size=512, use_cuda=False):
        super(generator, self).__init__()
        self.use_cuda = use_cuda
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
     
        # up2 12 -> 24
        self.d5 = nn.ConvTranspose3d(ndf, 32, 3, 1, padding=1, output_padding=0) #(35-1)*1-2*1+1*(3-1)+0+1=35
        ##Dout =(Din−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.bn9 = nn.BatchNorm3d(32, 1.e-3)

        # Output layer
        self.d6 = nn.Conv3d(32, nc, 3, 1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.output = nn.Tanh()
    def forward(self, x,c2,only_y): #x:16 9 35 35 35,c2: 35 35 35

        c2 = c2.view(c2.shape[0],35,35,35)
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

        c2_condition = torch.stack(c2_condition_f, 0) #16 1 35 35 35

        #print("c2_condition:",c2_condition.shape)

        cat_h3 = torch.cat([x, c2_condition], dim=1) #16 10 35 35 35

        h5 = self.leakyrelu(self.bn9(self.d5(cat_h3)))
        return self.output(self.d6(h5))


class EncoderCNN(nn.Module):
    def __init__(self, in_layers):
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                # Duplicate number of layers every alternating layer.
                out_layers *= 2
                layers.append(self.pool)
        layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers): #(512,1024,29,1)
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)#(29,512)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) #(512,1024,1)
        self.linear = nn.Linear(hidden_size, vocab_size)#(1024,29)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(105):  #62
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            a = outputs.max(1)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled1_ids, sampled2_ids = [], []
        inputs = features.unsqueeze(1)
        for i in range(105):  # 62  maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
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
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled1_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        inputs = features.unsqueeze(1)
        for i in range(128):  # 62  maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
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
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled2_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return sampled1_ids, sampled2_ids



class GNN_Model(nn.Module):
    def __init__(self, params):
        super(GNN_Model, self).__init__()
        n_graph_layer = params['n_graph_layer']
        d_graph_layer = params['d_graph_layer']
        n_FC_layer = params['n_FC_layer']
        d_FC_layer = params['d_FC_layer']
        N_atom_features = params['N_atom_features']
        final_dimension = params['final_dimension']
        self.dropout_rate = params['dropout_rate']


        self.layers1 = [d_graph_layer for i in range(n_graph_layer +1)]
        self.gconv1 = nn.ModuleList \
            ([GAT_gate(self.layers1[i], self.layers1[ i +1]) for i in range(len(self.layers1 ) -1)])

        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i== 0 else
                                 nn.Linear(d_FC_layer, final_dimension) if i == n_FC_layer - 1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])

        self.mu = nn.Parameter(torch.Tensor([params['initial_mu']]).float())
        self.dev = nn.Parameter(torch.Tensor([params['initial_dev']]).float())
        self.embede = nn.Linear(2 *N_atom_features , d_graph_layer, bias=False)
        self.params=params



    def fully_connected(self, c_hs):
        regularization = torch.empty(len(self.FC) * 1 - 1, device=c_hs.device)

        for k in range(len(self.FC)):
            # c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs
    def Formulate_Adj2(self,c_adjs2,c_valid,atom_list,device):
        study_distance = c_adjs2.clone().detach().to(device)  # only focused on where there exist atoms, ignore the area filled with 0
        study_distance = torch.exp(-torch.pow(study_distance - self.mu.expand_as(study_distance), 2) / self.dev)
        filled_value = torch.Tensor([0]).expand_as(study_distance).to(device)
        for batch_idx in range(len(c_adjs2)):
            num_atoms = int(atom_list[batch_idx])
            count_receptor = len(c_valid[batch_idx].nonzero())
            c_adjs2[batch_idx,:count_receptor,count_receptor:num_atoms]=torch.where(c_adjs2[batch_idx,:count_receptor,count_receptor:num_atoms]<=10,study_distance[batch_idx,:count_receptor,count_receptor:num_atoms],filled_value[batch_idx,:count_receptor,count_receptor:num_atoms])
            c_adjs2[batch_idx,count_receptor:num_atoms,:count_receptor]=c_adjs2[batch_idx,:count_receptor,count_receptor:num_atoms].t()
        return c_adjs2

    def get_attention_weight(self,data):
        c_hs, c_adjs1, c_adjs2 = data
        atten1,c_hs1 = self.gconv1[0](c_hs, c_adjs1,request_attention=True)  # filled 0 part will not effect other parts
        atten2,c_hs2 = self.gconv1[0](c_hs, c_adjs2,request_attention=True)
        return atten1,atten2
    def embede_graph(self, data):
        """

        :param data:
        :return: c_hs:batch_size*max_atoms
        """
        c_hs, c_adjs1, c_adjs2= data
        regularization = torch.empty(len(self.gconv1), device=c_hs.device)

        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)#filled 0 part will not effect other parts
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
        #c_hs = c_hs.sum(1)
        return c_hs
    def Get_Prediction(self,c_hs,atom_list):
        prediction=[]
        for batch_idx in range(len(atom_list)):
            num_atoms = int(atom_list[batch_idx])
            tmp_pred=c_hs[batch_idx,:num_atoms]
            tmp_pred=tmp_pred.sum(0)#sum all the used atoms
            #if self.params['debug']:
            #    print("pred feature size",tmp_pred.size())
            prediction.append(tmp_pred)
        prediction = torch.stack(prediction, 0)
        return prediction
    def train_model(self,data,device):
        #get data
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2=self.Formulate_Adj2(c_adjs2,c_valid,num_atoms,device)
        #then do the gate
        c_hs=self.embede_graph((c_hs,c_adjs1,c_adjs2))
        #if self.params['debug']:
        #    print("embedding size",c_hs.size())
        #sum based on the atoms
        c_hs=self.Get_Prediction(c_hs,num_atoms)
        c_hs = self.fully_connected(c_hs)
        #c_hs = c_hs.view(-1)
        return c_hs
    def test_model(self, data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms,device)
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
    def test_model_final(self,data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms, device)
        attention1, attention2 = self.get_attention_weight((c_hs, c_adjs1, c_adjs2))
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs,attention1,attention2
    def eval_model_attention(self,data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms, device)
        attention1,attention2 = self.get_attention_weight((c_hs, c_adjs1, c_adjs2))
        return attention1,attention2
    def feature_extraction(self,c_hs):
        for k in range(len(self.FC)):
                # c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=False)
                c_hs = F.relu(c_hs)

            return c_hs
    def model_gnn_feature(self, data,device):
        c_hs, c_adjs1, c_adjs2, c_valid, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, c_valid, num_atoms,device)
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        #c_hs = self.fully_connected(c_hs)
        #c_hs = c_hs.view(-1)
        c_hs=self.feature_extraction(c_hs)
        return c_hs




class GAT_gate(nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        # self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature * 2, 1)#default bias=True
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj,request_attention=False):
        h = self.W(x)#x'=W*x_in
        batch_size = h.size()[0]
        N = h.size()[1]#num_atoms
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h, self.A), h))#A is E in the paper,
        #This function provides a way of computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.
        e = e + e.permute((0, 2, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        output_attention=attention
        attention = attention * adj#final attention a_ij
        h_prime = F.relu(torch.einsum('aij,ajk->aik', (attention, h)))#x'' in the paper

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))##calculate z_i
        retval = coeff * x + (1 - coeff) * h_prime#final output,linear combination
        if request_attention:
            return output_attention,retval
        else:
            return retval
    def forward_single(self, x, adj):
        h = self.W(x)#x'=W*x_in
        #batch_size = h.size()[0]
        #N = h.size()[1]#num_atoms
        e = torch.einsum('jl,kl->jk', (torch.matmul(h, self.A), h))#A is E in the paper,
        #This function provides a way of computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.
        e = e + e.permute((0, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, h)
        attention = attention * adj#final attention a_ij
        h_prime = F.relu(torch.einsum('ij,jk->ik', (attention, h)))#x'' in the paper

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(1, 1, x.size(-1))##calculate z_i
        retval = coeff * x + (1 - coeff) * h_prime#final output,linear combination
        return retval