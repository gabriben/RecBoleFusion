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

# https://github.com/usuyama/pytorch-unet


import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out



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

        self.p_dnns = nn.ModuleList([nn.Sequential(
            *[nn.Linear(D, M), nn.PReLU()] +
            [nn.Linear(M, M), nn.PReLU()] * self.p_dnns_depth + [nn.Linear(M, 2*D)])
                                     for _ in range(self.T-1)])

        self.decoder_net = nn.Sequential(
            *[nn.Linear(D, M), nn.PReLU()] +
            [nn.Linear(M, M), nn.PReLU()] * self.decoder_net_depth + [nn.Linear(M, D), nn.Tanh()])

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
        self.mus = []
        self.log_vars = []

        for i in range(len(self.p_dnns) - 1, -1, -1):

            h = self.p_dnns[i](self.Z[i+1])
            mu_i, log_var_i = torch.chunk(h, 2, dim=1)

            ##################################################
            if not self.x_to_negpos: # maybe not needed to reparametrize if the original data was already -1, 1
                mu_i = reparameterization(mu_i, log_var_i)
            ##################################################
            
            self.mus.append(mu_i)
            self.log_vars.append(log_var_i)

        if self.decode_from_noisiest:
            mu_x = self.decoder_net(self.Z[-1])
        else:
            mu_x = self.decoder_net(self.Z[0])

        return mu_x

    def calculate_loss(self, interaction):
        
        user = interaction[self.USER_ID]

        x = self.get_rating_matrix(user)

        mu_x = self.forward(x)
        
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap        

        ########################
            
        # =====ELBO
        # RE

        # Normal RE
        RE = log_standard_normal(x - mu_x).sum(-1)

        # KL
        KL = (log_normal_diag(self.Z[-1], torch.sqrt(1. - self.betas[0]) * self.Z[-1],
                              torch.log(self.betas[0])) - log_standard_normal(self.Z[-1])).sum(-1)

        for i in range(len(self.mus)):
            KL_i = (log_normal_diag(self.Z[i], torch.sqrt(1. - self.betas[i]) * self.Z[i], torch.log(
                self.betas[i])) - log_normal_diag(self.Z[i], self.mus[i], self.log_vars[i])).sum(-1)

            KL = KL + KL_i

        # Final ELBO

        if self.reduction == 'sum':
            loss = -(RE - anneal * KL).sum()
        else:
            loss = -(RE - anneal * KL).mean()

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        x = self.get_rating_matrix(user.unique())

        scores = self.forward(x)

        return scores
