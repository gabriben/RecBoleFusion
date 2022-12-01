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


########################




class RecFusion(GeneralRecommender):
    r"""RecFusion

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        reg_weight = config['reg_weight']

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))



        X = dataset.inter_matrix(form='csr').astype(np.float32)
        # just directly calculate the entire score matrix in init
        # (can't be done incrementally)
        x = torch.FloatTensor(X.toarray())
        print(x)
        print(x.shape)        


        # =====
        # forward difussion
        Z = [self.reparameterization_gaussian_diffusion(x, 0)]

        for i in range(1, self.T):
            Z.append(self.reparameterization_gaussian_diffusion(Z[-1], i))
        

        # self.item_similarity = B
        self.interaction_matrix = Z
        self.other_parameter_name = ['interaction_matrix', 'item_similarity']

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy(
            (self.interaction_matrix[user, :].multiply(self.item_similarity[:, item].T)).sum(axis=1).getA1()
        )

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.interaction_matrix[user, :] @ self.item_similarity
        return torch.from_numpy(r.flatten())
