import os
import pandas as pd
import numpy as np
import torch 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

from cpu_test import KeyGen
from cpu_test import Enc, Dec

import copy
import collections


# def encrypt_weights_params(public_key,params):
#     prec = 32
#     bound = 2 ** 3
#     l = 2**6
#     params_list = list()
    
#     params_c = copy.deepcopy(value).view(-1).float()
#     params_padded = torch.zeros((l))
#     params_padded[:len(params_c)] = params_c
    
#     params_padded_long = ((params_padded+bound)*(2**prec)).long().numpy()
#     #params_list = [((params + bound) * 2 ** prec).long().cuda() for params in params_list]
#     #params_list = [((params + bound) * 2 ** prec).long().numpy() for params in params_padded]
    
#     #encrypted_params = [Enc(public_key, params) for params in params_list]
#     encrypted_params = Enc(pk,params_padded_long)
    
#     return encrypted_params

def encrypt_weights_params(public_key,params):
    
    return params



# def decrypt_weights_param(secret_key, params, bound_multi):
#     m= Dec(secret_key, params)
#     out = (torch.tensor(m).float()/(2**prec))-bound_multi*bound
#     out = out/bound_multi
#     return out

def decrypt_weights_param(secret_key, params, bound_multi):

    return params