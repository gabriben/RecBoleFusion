r"""
RecFusion
################################################

"""

import torch
import numpy as np
import scipy.sparse as sp

from recbole.utils import InputType, ModelType
from recbole.model.abstract_recommender import GeneralRecommender
#import pdb 



########################


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# import ranking_metrics as metric

import pdb


# device = "cuda" if torch.cuda.is_available() else "cpu"


# Auxiliary functions, classes such as distributions


PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7


def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - \
        0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_min_exp(a, b, epsilon=1e-8):

    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y


def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
        F.logsigmoid((x + 0.5 - mean) / scale),
        F.logsigmoid((x - 0.5 - mean) / scale))

    return logp


def log_integer_probability_standard(x):
    logp = log_min_exp(
        F.logsigmoid(x + 0.5),
        F.logsigmoid(x - 0.5))


def reparameterization(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mu + std * eps


def reparameterization_gaussian_diffusion(x, i, B):
    return torch.sqrt(1. - B) * x + torch.sqrt(B) * torch.randn_like(x)

T = 3  # hyperparater to tune

M = 200  # the number of neurons in scale (s) and translation (t) nets

b = 0.0001  # hyperparater to tune

reduction = "sum"

num_layers = "TODO"

########################

class RecFusion(GeneralRecommender):
    r"""RecFusion

    """
    input_type = InputType.POINTWISE
    # type = ModelType.TRADITIONAL

    # x = None
    # p_dnns = nn.ModuleList()
    # decoder_net = nn.ModuleList()
    # mu_x = None
    # Z = None
    # mus = None
    # log_vars = None
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # recbole code
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        
        ########################

        # self.device = config['device']

        self.beta = torch.FloatTensor([b]).to(self.device)
        
        D = dataset.item_num

        self.p_dnns = nn.ModuleList([nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, M), nn.PReLU(),
                                              nn.Linear(M, 2*D)) for _ in range(T-1)])

        self.decoder_net = nn.Sequential(nn.Linear(D, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, M), nn.PReLU(),
                                    nn.Linear(M, D), nn.Tanh())

        ########################

        # load parameters info
        # reg_weight = config['reg_weight']

        # x = dataset.inter_matrix(form='csr').astype(np.float32)
        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)

        # self.x = torch.FloatTensor(x.toarray()).to(self.device)

        # self.x = self.get_rating_matrix(user)
        # print(self.x)
        # print(self.x.shape)
        # self.interaction_matrix = x

        
    def init_weights(self):
        for layer in self.p_dnns:
            # Xavier Initialization for weights
            size = layer[0].weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer[0].weight.data.normal_(0.0, 0.0001)

            # Normal Initialization for Biases
            layer[0].bias.data.normal_(0.0, 0.0001)
            
        for layer in self.decoder_net:
            # Xavier Initialization for weights
            if str(layer) == "Linear":
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0/(fan_in + fan_out))
                layer.weight.data.normal_(0.0, 0.0001)

                # Normal Initialization for Biases
                layer.bias.data.normal_(0.0, 0.0001)

    # recbole code from multvae
    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device) \
            .repeat_interleave(self.history_item_id.shape[1], dim=0)
        rating_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        rating_matrix.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        return rating_matrix
                
    def forward(self, x):        
        
        # x = self.interaction_matrix

        # =====
        # forward difussion
        self.Z = [reparameterization_gaussian_diffusion(x, 0, self.beta)]

        for i in range(1, T):
            self.Z.append(reparameterization_gaussian_diffusion(self.Z[-1], i, self.beta))
        
        # =====
        # backward diffusion
        self.mus = []
        self.log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1):

            h = self.p_dnns[i](self.Z[i+1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)
            self.mus.append(mu_i)
            self.log_vars.append(log_var_i)

        # pdb.set_trace()
            
        self.mu_x = self.decoder_net(self.Z[0]) # TODO: try Z[-1]

        # print("mu_x " + self.mu_x)
            
        # self.item_similarity = B
        # self.interaction_matrix = Z
        # self.other_parameter_name = ['interaction_matrix', 'item_similarity']
        # self.mu_x = mu_x

        # return self
        return self.mu_x

    def calculate_loss(self, interaction):

        # x = self.interaction_matrix

        user = interaction[self.USER_ID]
        x = self.get_rating_matrix(user)     

        self.init_weights()
        # self = self.forward()
        self.forward(x)
        
        # mu_x = self.mu_x

        # =====ELBO
        # RE

        # Normal RE
        RE = log_standard_normal(x - self.mu_x).sum(-1)

        # KL
        KL = (log_normal_diag(self.Z[-1], torch.sqrt(1. - self.beta) * self.Z[-1],
                              torch.log(self.beta)) - log_standard_normal(self.Z[-1])).sum(-1)

        for i in range(len(self.mus)):
            KL_i = (log_normal_diag(self.Z[i], torch.sqrt(1. - self.beta) * self.Z[i], torch.log(
                self.beta)) - log_normal_diag(self.Z[i], self.mus[i], self.log_vars[i])).sum(-1)

            KL = KL + KL_i    

        # Final ELBO
        anneal = 1
        if reduction == 'sum':
            loss = -(RE - anneal * KL).sum()
        else:
            loss = -(RE - anneal * KL).mean()

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        # pdb.set_trace()
        # print(self.x[user, item].shape)

        x = self.get_rating_matrix(user)

        # print(rating_matrix)
        # print(rating_matrix.shape)        

        scores = self.forward(x) * -1

        # print(scores)
        # print(scores.shape)

        # pdb.set_trace()
        
        return scores[[torch.arange(len(item)).to(self.device), item]]        

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        x = self.get_rating_matrix(user)

        scores = self.forward(x) * -1
        
        return scores.view(-1)

    
        # return self.forward(self.x[user, item])

        # return self.mu_x[user, item]

        # return torch.from_numpy(
        #     (self.interaction_matrix[user, :]
        # ))

    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID].cpu().numpy()

    #     r = self.interaction_matrix[user, :]
    #     R = torch.FloatTensor(r.toarray())
    #     return torch.from_numpy(R.flatten().numpy())

# [🐛BUG] running run_recbole.py with `ml-1m` and `NeuMF` errors out
    
# **Describe the bug**
# A clear and concise description of what the bug is.

# **To Reproduce**
# Run this in colab: https://colab.research.google.com/drive/1163UQKKcmY2uiWxmHsmm_sZ6Uw5x9x_S?usp=sharing    
