# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:10:10 2022

"""

import pandas as pd

from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import math 
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
from typing import Dict
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm 


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_data(cid, run_type='train'):
    if run_type=='train':
        df = pd.read_csv(cid_path_list[cid])
    else:
        df = pd.read_csv(cid_path_list_test[cid])
    return df

def shuffle(X: np.ndarray, y: np.ndarray):
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
class LogisticRegression(nn.Module):
    def __init__(self, param_count=5):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(param_count,1)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model."""
    list_params=model.parameters()
    params=[tens_param.detach() for tens_param in list_params]
    return params



def set_model_params(model, params):
    """Sets the parameters of a sklean LogisticRegression model."""


    for i, layer_weights in enumerate(model.parameters()):
        layer_weights.data.sub_(layer_weights.data)
        layer_weights.data.add_(torch.tensor(params[i]))
        
    return model


def set_initial_params(model):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
        
    for layer_weights in model.parameters():
        layer_weights.data.sub_(layer_weights.data)

        
def loss_classifier(predictions,labels):
    
    loss = nn.BCEWithLogitsLoss()

    labels = labels.float()
    y_pred = predictions.float().view(-1)
    
    return loss(y_pred,labels.view(-1))
    

def difference_models_norm_2(tensor_1, tensor_2):
    """Return the norm 2 difference between the two model parameters
    """
#     tensor_1=list(model_1)
#     tensor_2=list(model_2)
   
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    return norm

def zero_grad(optimizer):
    """Clears the gradients of all optimized :class:`Variable` s."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
                
    return optimizer 


def train_step(model, global_parameters, mu, optimizer, lr, X_train, y_train, loss_f, max_grad_norm, epsilon):
    """
    Function where forecasting model is trained on the training batch sample and 
    federated learning techniques are applied on the gradients of the NN. 
    """
    
    predictions= model(X_train)

    loss=loss_f(predictions,y_train)
    loss+=mu/2*difference_models_norm_2(get_model_parameters(model),global_parameters) #can divide by total batches (#samples)
    loss.backward()

    #DP-SGD
    for param in model.parameters():
        per_sample_grad = param.grad.detach().clone()
        clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
        param.accumulated_grads.append(per_sample_grad)


    for param in model.parameters():
        param.grad = torch.stack(param.accumulated_grads, dim=0)[0]
    
    for i, param in enumerate(model.parameters()):
        param_ = param - lr * param.grad
        
        nm = (math.sqrt(2*(math.log(1.25*X_train.size()[0]))))/epsilon
        noise_vector = torch.Tensor([nm]*X_train.size()[1])
        
        param_ = param_ + torch.normal(mean=0.0, std = torch.Tensor(noise_vector))

        param.data.sub_(param.data)
        param.data.add_(param_[0])

            
   
    #set the model parameters and check params before & after 
    print("Training Loss: ",get_loss_metric(model, X_train, y_train))
    print("Training Accuracy: ",get_accuracy_metric(model, X_train, y_train))
    
    return model, loss

def get_accuracy_metric(model, x, y):
    """Compute the accuracy of `model` on `dataset`"""
    with torch.no_grad():
        correct=0
        predictions= model(x)
        _,predicted=predictions.max(1,keepdim=True)

        correct+=torch.sum(predicted.view(-1,1)==y.view(-1, 1)).item()
        accuracy = 100*correct/len(x)

    return accuracy


def get_loss_metric(model, X_train, y_train):
    predictions = model(X_train)
    loss = loss_classifier(predictions,y_train)
                           
    return loss


def model_fit(model, global_parameters, mu, X_train, y_train, 
        epsilon, max_grad_norm, lr,batch_size,epochs):
   
    loss_f=loss_classifier
    all_per_sample_gradients = []
    
    dataset = TensorDataset(X_train, y_train)
    
    for epoch in range(int(epochs)):
        for batch in DataLoader(dataset, batch_size=batch_size):
            optimizer=optim.SGD(model.parameters(),lr=lr)
            
            for param in model.parameters():
                param.accumulated_grads = []
                
            X_train, y_train = batch
#             model, local_loss = train_step(model, global_parameters, mu, optimizer,lr,
#                                            X_train, y_train, loss_f, max_grad_norm, epsilon)
            
            #TRAINING STEP 
            predictions= model(X_train)

            loss=loss_f(predictions,y_train)
            loss+=mu/2*difference_models_norm_2(get_model_parameters(model),global_parameters) #can divide by total batches (#samples)
            loss.backward()

            #DP-SGD - Clipping Gradients
            for param in model.parameters():
                per_sample_grad = param.grad.detach().clone()
                clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
                param.accumulated_grads.append(per_sample_grad)


        for param in model.parameters():
            param.grad = torch.stack(param.accumulated_grads, dim=0)[0]

        for i, param in enumerate(model.parameters()):
            param_ = param - lr * param.grad

            nm = (math.sqrt(2*(math.log(1.25*X_train.size()[0]))))/epsilon
            noise_vector = torch.Tensor([nm]*X_train.size()[1])

            param_ = param_ + torch.normal(mean=0.0, std = torch.Tensor(noise_vector))

            param.data.sub_(param.data)
            param.data.add_(param_[0])
            
#         param.grad = 0  # Reset for next iteration       
        zero_grad(optimizer)  # p.grad is cumulative so we'd better reset it
        
        print(f"Epoch No. {epoch} Training Loss: ",get_loss_metric(model, X_train, y_train))
        print(f"Epoch No. {epoch}Training Accuracy: ",get_accuracy_metric(model, X_train, y_train))        
              
    return model 

def prepare_developer_true_label(dict_of_loc_absoulte):
    diseas_outcome_training = pd.read_csv(dict_of_loc_absoulte['disease_outcome_training'])
    diseas_outcome_training['state_int'] = (diseas_outcome_training['state']=='I').astype(int)

    diseas_outcome_training = diseas_outcome_training[diseas_outcome_training['day']>=50]

    true_label = diseas_outcome_training.groupby("pid")['state_int'].sum().reset_index()

    true_label['state'] = (true_label["state_int"]>=1).astype(int)

    true_label.rename(columns ={'state':'true_label'}, inplace=True)
    return true_label
        
def utils_test_model(model, x, y, threshold):
    predictions = model(x)
    predictions_np = predictions.detach().numpy()
    final_predictions = (predictions_np >= threshold).astype(int)
    average_precision = average_precision_score(y, predictions_np)
    loss = get_loss_metric(model, x, y)
    accuracy = get_accuracy_metric(model, x, y)
    
#     y_score = clf.predict_proba(X_std)[:, 1]
#     average_precision = average_precision_score( y, y_score)

#     pred_pats = (pd.Series(y_score) >= threshold).astype(int)
    
#     prediction = pd.DataFrame(data = {'pid':final_data.pid.values,
#                            'covid_predicted':pred_pats.values,
#                           'score':y_score,
#                           'covid_ground_truth':y})

#     precision,recall = precision_score(y,pred_pats),recall_score(y,pred_pats)
#     log_loss = log_loss(y, y_score)
    
    
    return predictions_np, loss, accuracy, average_precision

def fed_metrics_weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
   