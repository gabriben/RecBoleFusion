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

## maybe not-so-simple Unet model from here: https://github.com/usuyama/pytorch-unet


########################

from recbole.model.general_recommender import ddpm

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

# T = 3  # hyperparater to tune

# M = 200  # the number of neurons in scale (s) and translation (t) nets

# b = 0.0001  # hyperparater to tune

# reduction = "avg"

# num_layers = "TODO"

# total_anneal_steps = 200000

# anneal_cap = 0.2


########################

class RecFusion(GeneralRecommender):
    r"""RecFusion

    """
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # recbole code
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]

        self.T, self.M, self.b = config["T"], config["M"], config["b"]
        self.reduction = config["reduction"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.anneal_cap = config["anneal_cap"]
        self.xavier_initialization = config["xavier_initialization"]
        self.x_to_negpos = config["x_to_negpos"]
        self.decode_from_noisiest = config["decode_from_noisiest"]

        self.p_dnns_depth = config["p_dnns_depth"]
        self.decoder_net_depth = config["decoder_net_depth"]
        self.b_start = config["b_start"]
        self.b_end = config["b_end"]
        self.schedule_type = config["schedule_type"]

        self.update = 0        
        
        ########################

        betas = self.get_beta_schedule(self.schedule_type)
        self.betas = torch.FloatTensor(betas).to(self.device)
        
        D = dataset.item_num
        M = self.M

        # self.p_dnns = nn.ModuleList([nn.Sequential(
        #     *[nn.Linear(D, M), nn.PReLU()] +
        #     [nn.Linear(M, M), nn.PReLU()] * self.p_dnns_depth + [nn.Linear(M, 2*D)])
        #                              for _ in range(self.T-1)])

        # self.decoder_net = nn.Sequential(
        #     *[nn.Linear(D, M), nn.PReLU()] +
        #     [nn.Linear(M, M), nn.PReLU()] * self.decoder_net_depth + [nn.Linear(M, D), nn.Tanh()])

        self.unet = ddpm.OriginalUnet(dim = 2, channels = 1, resnet_block_groups=1, dim_mults=(1, 2))
        
        ########################

        if self.xavier_initialization:
            self.init_weights()
        
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

    # https://github.com/InFoCusp/diffusion_models/blob/main/Diffusion_models.ipynb
    def get_beta_schedule(self, schedule_type):
      if schedule_type == 'quadratic':
        betas = np.linspace(self.b_start ** 0.5, self.b_end ** 0.5, self.T, dtype=np.float32) ** 2
      elif schedule_type == 'linear':
        betas = np.linspace(self.b_start, self.b_end, self.T, dtype=np.float32)
      return betas    
                
    def forward(self, x):

        T = self.T

        if self.x_to_negpos:
            x = (x - 0.5 ) * 2
            
        # =====
        # forward difussion
        self.Z = [reparameterization_gaussian_diffusion(x, 0, self.betas[0])]

        for i in range(1, T):
            self.Z.append(reparameterization_gaussian_diffusion(self.Z[-1], i, self.betas[i]))
        
        # =====
        # backward diffusion

        mu_s = []
        
        for t in range(T - 1):
            # t = torch.FloatTensor([t]).to(self.device)
            # pdb.set_trace()
            t = torch.tensor([t], dtype=torch.int32).to(self.device)
            mu_t = self.unet.forward(self.Z[t+1][None, None, :, :], t+1)
            mu_s.append(mu_t)

        return mu_s            
        
        # self.mus = []
        # self.log_vars = []

        # for i in range(len(self.p_dnns) - 1, -1, -1):

        #     h = self.p_dnns[i](self.Z[i+1])
        #     mu_i, log_var_i = torch.chunk(h, 2, dim=1)

        #     ##################################################
        #     if not self.x_to_negpos: # maybe not needed to reparametrize if the original data was already -1, 1
        #         mu_i = reparameterization(mu_i, log_var_i)
        #     ##################################################
            
        #     self.mus.append(mu_i)
        #     self.log_vars.append(log_var_i)

        # if self.decode_from_noisiest:
        #     mu_x = self.decoder_net(self.Z[-1])
        # else:
        #     mu_x = self.decoder_net(self.Z[0])

        # return mu_x

    def calculate_loss(self, interaction):
        
        user = interaction[self.USER_ID]

        x = self.get_rating_matrix(user)[:, : -1]

        # mu_x = self.forward(x)

        Z = self.forward(x)
        
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap        

        ########################
            
        # =====ELBO
        # RE

        # Normal RE
        # RE = log_standard_normal(x - mu_x).sum(-1)
        RE = log_standard_normal(x - Z[0]).sum(-1)

        # KL
        KL = (log_normal_diag(self.Z[-1], torch.sqrt(1. - self.betas[0]) * self.Z[-1],
                              torch.log(self.betas[0])) - log_standard_normal(self.Z[-1])).sum(-1)

        for i in range(1, len(Z) - 1):
            KL_i = (log_normal_diag(self.Z[i], torch.sqrt(1. - self.betas[i]) * self.Z[i],
                              torch.log(self.betas[i])) - log_standard_normal(self.Z[i])).sum(-1)
        
        # for i in range(len(self.mus)):
        #     KL_i = (log_normal_diag(self.Z[i], torch.sqrt(1. - self.betas[i]) * self.Z[i], torch.log(
        #         self.betas[i])) - log_normal_diag(self.Z[i], self.mus[i], self.log_vars[i])).sum(-1)

            KL = KL + KL_i

        # Final ELBO

        if self.reduction == 'sum':
            loss = -(RE - anneal * KL).sum()
        else:
            loss = -(RE - anneal * KL).mean()

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        x = self.get_rating_matrix(user.unique())[:, : -1]

        scores = self.forward(x)

        return scores
