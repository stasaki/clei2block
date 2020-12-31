import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
from torch.utils import data
import copy
import math
import sys
import subprocess

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_x, data_y, data_x_mat):
        'Initialization'
        self.data_x = data_x
        self.data_y = data_y
        self.data_x_mat = data_x_mat
    def __len__(self):
        'Denotes the total number of samples'
        return self.data_x.shape[1]
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data_x[:,index]
        Y = self.data_y[:,index]
        X_mat = [self.data_x_mat[i][:,index] for i in range(len(self.data_x_mat))]
        return X, X_mat, Y
    
# =============================== Q(z|X) ======================================
class Q(nn.Module):
    def __init__(self):
        super(Q,self).__init__()
        self.fc1 = nn.Linear(X_dim,h_dim,bias=False)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim,bias=False)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.fc_mu = nn.Linear(h_dim,Z_dim)
        self.fc_var = nn.Linear(h_dim,Z_dim)
    def forward(self, X):
        h = self.fc1(X)
        h = self.bn1(h)
        h = nn.LeakyReLU()(h)
        h = nn.Dropout(p=0.5)(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = nn.LeakyReLU()(h)
        h = nn.Dropout(p=0.5)(h)
        z_mu = self.fc_mu(h)
        z_var = self.fc_var(h)
        return z_mu, z_var
    
# =============================== P(X|z) ======================================
class P(nn.Module):
    def __init__(self):
        super(P,self).__init__()
        self.fc1 = nn.Linear(Z_dim,h_dim,bias=False)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.fc2 = nn.Linear(h_dim,h_dim,bias=False)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.fc3 = nn.Linear(h_dim,y_dim,bias=False)
    def forward(self, z):
        h = self.fc1(z)
        h = self.bn1(h)
        h = nn.LeakyReLU()(h)
        h = nn.Dropout(p=0.5)(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = nn.LeakyReLU()(h)
        h3 = nn.Dropout(p=0.5)(h)
        X = self.fc3(h3)
        return X, h3
    
def sample_z(mu, log_var, n_sample):
    eps = Variable(torch.randn(n_sample, Z_dim)).cuda()
    return mu + torch.exp(log_var / 2) * eps

class VAE(nn.Module):
    def __init__(self,n_mat,init_method):
        super(VAE,self).__init__()
        self.Q = Q()
        self.P = P()
        self.weight=nn.ParameterList([nn.Parameter(init_param([1,y_dim],init_method),requires_grad=True) for i in range(n_mat)])
        if use_train_scale == "TRUE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=True)
        elif use_train_scale == "FALSE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=False)
    def forward(self, X, X_mat):
        z_mu, z_var = self.Q(X)
        z = sample_z(z_mu, z_var, X.shape[0])
        X_sample, h = self.P(z)
        for i in range(n_mat):
            X_sample = X_mat[i]*self.weight[i]+X_sample
        X_sample = self.bn(X_sample)
        return X_sample, z, z_mu, z_var, h
    
def vae_loss(y_true, y_pred, mu, log_var, alpha):
    return torch.mean(recon_loss(y_true, y_pred) + alpha * kl_loss(mu, log_var))

def kl_loss(mu, log_var):
    return 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1. - log_var,1)
    
def recon_loss(y_true, y_pred):
    return 0.5 * torch.sum((y_true - y_pred)**2,1)
    
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

def init_param(size,init_method):
    if init_method=="kaiming_uniform":
        y = Variable(nn.init.kaiming_uniform_(torch.empty(size), a=math.sqrt(5)), requires_grad=True)
    elif init_method=="kaiming_normal":
        y = Variable(nn.init.kaiming_normal_(torch.empty(size)), requires_grad=True)
    elif init_method=="normal":
        y = Variable(nn.init.normal_(torch.empty(size)), requires_grad=True)
    elif init_method=="xavier_normal":
        y = Variable(nn.init.xavier_normal_(torch.empty(size)), requires_grad=True)
    elif init_method=="kaiming_uniform":
        y = Variable(nn.init.kaiming_uniform_(torch.empty(size)), requires_grad=True)
    elif init_method=="uniform":
        y = Variable(nn.init.uniform_(torch.empty(size),a=-1.0, b=1.0), requires_grad=True)
    return y

def scale_sub(train_x_iter):
    train_mean = np.zeros(train_x_iter.shape[0])
    train_std = np.zeros(train_x_iter.shape[0])
    for i in range(train_x_iter.shape[0]):
        train_mean[i] = np.nanmean(train_x_iter[i,:])
        train_x_iter[i,:] = train_x_iter[i,:] - train_mean[i]
        train_std[i] = np.nanstd(train_x_iter[i,:],ddof=0)
        train_std[train_std==0]=1
        train_x_iter[i,:] = train_x_iter[i,:] / train_std[i]
    return train_x_iter, [train_mean, train_std]
    
def scaling(test_x, test_y, test_x_mat):
    test_x, test_x_stat = scale_sub(test_x)
    test_y, test_y_stat = scale_sub(test_y)
    test_x_mat_stat = [None] * len(test_x_mat)
    for j in range(len(test_x_mat)):
        test_x_mat[j], test_x_mat_stat[j] = scale_sub(test_x_mat[j])
    return test_x, test_y, test_x_mat, test_x_stat, test_y_stat, test_x_mat_stat
    
def scale_prefix_sub(test_x,train_x_iter_stat):
    for i in range(test_x.shape[0]):
        test_x[i,:] = test_x[i,:] - train_x_iter_stat[0][i]
        test_x[i,:] = test_x[i,:] / train_x_iter_stat[1][i]
    return test_x
    
def scale_prefix(test_x, test_y, test_x_mat, train_x_iter_stat, train_y_iter_stat, train_x_mat_iter_stat):
    test_x = scale_prefix_sub(test_x, train_x_iter_stat)
    test_y = scale_prefix_sub(test_y, train_y_iter_stat)
    for j in range(len(test_x_mat)):
        test_x_mat[j] = scale_prefix_sub(test_x_mat[j],train_x_mat_iter_stat[j])
    return test_x, test_y, test_x_mat


argv = sys.argv
argc = len(argv)
if (argc < 7):
    print ('Usage: python')
    quit()
inloc = argv[1]
outloc = argv[2]
Z_dim = int(argv[3])
h_dim = int(argv[4])
targetloc = argv[5]
vae_input = argv[6]
lr_input = argv[7:]
lr_vae = 0.001
lr_add = 0.01
alpha = 0.00005  
mb_size = 400
patience = 30
max_iter = 10000
use_train_scale = "FALSE"
n_mat=len(lr_input)
init_method = "normal"

# =============================== DATA LOADING ====================================
data_split="train"
train_x = np.genfromtxt(inloc+"/"+data_split+"_vae_x_"+vae_input+".txt.gz")
train_y = np.genfromtxt(inloc+"/"+data_split+"_y.txt.gz")
train_x_mat = [np.genfromtxt(inloc+"/"+data_split+"_lr_x_"+v+".txt.gz") for v in lr_input]

data_split="test"
test_x = np.genfromtxt(inloc+"/"+data_split+"_vae_x_"+vae_input+".txt.gz")
test_y = np.genfromtxt(inloc+"/"+data_split+"_y.txt.gz")
test_x_mat = [np.genfromtxt(inloc+"/"+data_split+"_lr_x_"+v+".txt.gz") for v in lr_input]


if targetloc!="none":
    target_x = np.genfromtxt(targetloc+"/vae_x_"+vae_input+".txt.gz")
    target_y = np.genfromtxt(targetloc+"/y.txt.gz")
    target_x_mat = [np.genfromtxt(targetloc+"/lr_x_"+v+".txt.gz") for v in lr_input]


X_dim = train_x.shape[0]
y_dim = train_y.shape[0]

# =============================== TRAINING ====================================
def run(outloc):
    if not os.path.exists(outloc):
        os.makedirs(outloc)
    
    vae = VAE(n_mat, init_method).float()
    vae.cuda()
    
    # create your optimizer
    optimizer = optim.Adam([
        {'params': vae.P.parameters()},
        {'params': vae.Q.parameters()},
        {'params': vae.weight.parameters(), 'lr': lr_add}],
        lr=lr_vae)
        
    # Parameters
    params = {'batch_size': mb_size,
              'shuffle': True,
              'num_workers': 0,
              'pin_memory':True}
              
    # initial loss
    best_loss=float("inf")
    # count epoch showing no improvement
    not_improve=0
    for it in range(max_iter):
        #print(it)
        # training
        train = Dataset(train_x_iter,
                        train_y_iter,
                        train_x_mat_iter)
        training_generator = data.DataLoader(train, **params)
        vae.train()
        train_loss=0
        train_loss_r=0
        train_loss_k=0 
        for i,(X, X_mat, Y) in enumerate(training_generator):
            #print(i)
            X = X.float().cuda()
            Y = Y.float().cuda()
            for i_mat in range(n_mat):
                X_mat[i_mat][torch.isnan(X_mat[i_mat])]=0
                X_mat[i_mat] = X_mat[i_mat].float().cuda()
                
            # Model computation
            optimizer.zero_grad()   # zero the gradient buffers
            X_sample, z, z_mu, z_var, h = vae(X, X_mat)

            X_sample[torch.isnan(Y)]=0
            Y[torch.isnan(Y)]=0
            
            # Loss
            loss_r = recon_loss(Y, X_sample)
            loss_k = kl_loss(z_mu, z_var)
            loss = torch.mean(loss_r + alpha * loss_k) 
            loss.backward()
            optimizer.step()    # Does the update
            train_loss=train_loss+loss.detach().cpu().numpy()
            train_loss_r=train_loss_r+torch.mean(loss_r).detach().cpu().numpy()
            train_loss_k=train_loss_k+torch.mean(loss_k).detach().cpu().numpy()
        if it % 10 == 0:
            # save training loss
            with open(outloc+"/train_loss.txt", "a") as f:
                print(str(train_loss/(i+1))+","+str(train_loss_r/(i+1))+","+str(train_loss_k/(i+1)),file=f) 

            # validation loss
            valid = Dataset(valid_x_iter,
                            valid_y_iter,
                            valid_x_mat_iter)
            valid_generator = data.DataLoader(valid, **params)
            valid_loss=0
            valid_loss_r=0
            valid_loss_k=0 
            with torch.no_grad():
                vae.eval()
                for i,(X, X_mat, Y) in enumerate(valid_generator):
                    X = X.float().cuda()
                    Y = Y.float().cuda()
                    for i_mat in range(n_mat):
                        X_mat[i_mat][torch.isnan(X_mat[i_mat])]=0
                        X_mat[i_mat] = X_mat[i_mat].float().cuda()

                    X_sample, z, z_mu, z_var, h = vae(X, X_mat)
                    X_sample[torch.isnan(Y)]=0
                    Y[torch.isnan(Y)]=0
                    # Loss
                    loss_r = recon_loss(Y, X_sample)
                    loss_k = kl_loss(z_mu, z_var)
                    loss = torch.mean(loss_r + alpha * loss_k)
                    valid_loss=valid_loss+loss.detach().cpu().numpy()
                    valid_loss_r=valid_loss_r+torch.mean(loss_r).detach().cpu().numpy()
                    valid_loss_k=valid_loss_k+torch.mean(loss_k).detach().cpu().numpy()
                with open(outloc+"/valid_loss.txt", "a") as f:
                    print(str(valid_loss/(i+1))+","+str(valid_loss_r/(i+1))+","+str(valid_loss_k/(i+1)),file=f) 

            # early stopping
            if valid_loss<best_loss:
                best_loss=valid_loss
                not_improve=0
                torch.save(vae.state_dict(), outloc+'/trained_model.pt')
            else:
                not_improve=not_improve+1
            if not_improve>=patience: 
                break
    del vae, X, X_mat, Y, X_sample, z, z_mu, z_var, loss, valid_loss, h
    torch.cuda.empty_cache()
    return best_loss
    

best_loss = []
np.random.seed(1234)
for it_val in range(5):
    val_index = np.random.permutation(list(range(train_x.shape[1])))[0:int(train_x.shape[1]/10)]
    
    # subset training data
    train_x_iter = np.delete(train_x, val_index, axis=1)
    train_y_iter = np.delete(train_y, val_index, axis=1)
    train_x_mat_iter = [np.delete(train_x_mat[i], val_index, axis=1) for i in range(len(train_x_mat))]
    if use_train_scale == "FALSE":
        train_x_iter, train_y_iter, train_x_mat_iter, _, _, _ = scaling(train_x_iter, train_y_iter, train_x_mat_iter)
    elif use_train_scale == "TRUE":
        train_x_iter, train_y_iter, train_x_mat_iter, train_x_iter_stat, train_y_iter_stat, train_x_mat_iter_stat = scaling(train_x_iter, train_y_iter, train_x_mat_iter)
            
    # subset validation data
    valid_x_iter = train_x[:,val_index]
    valid_y_iter = train_y[:,val_index]
    valid_x_mat_iter = [train_x_mat[i][:,val_index] for i in range(len(train_x_mat))]
    if use_train_scale == "FALSE":
        valid_x_iter, valid_y_iter, valid_x_mat_iter, _, _, _ = scaling(valid_x_iter, valid_y_iter, valid_x_mat_iter)
    elif use_train_scale == "TRUE":
        valid_x_iter, valid_y_iter, valid_x_mat_iter = scale_prefix(valid_x_iter, valid_y_iter, valid_x_mat_iter, train_x_iter_stat, train_y_iter_stat, train_x_mat_iter_stat)
    
    # scaling testing data
    test_x_iter = copy.deepcopy(test_x)  
    test_y_iter = copy.deepcopy(test_y)  
    test_x_mat_iter = copy.deepcopy(test_x_mat)  
    if use_train_scale == "FALSE":
        test_x_iter, test_y_iter, test_x_mat_iter, _, _, _ = scaling(test_x_iter, test_y_iter, test_x_mat_iter)
    elif use_train_scale == "TRUE":
        test_x_iter, test_y_iter, test_x_mat_iter = scale_prefix(test_x_iter, test_y_iter, test_x_mat_iter, train_x_iter_stat, train_y_iter_stat, train_x_mat_iter_stat)
    
    # train model
    best_loss.append(run(outloc+str(it_val)+"/"))
    
    # save validation index
    with open(outloc+str(it_val)+"/val_index.txt",'w') as f:
        np.savetxt(f, val_index, delimiter='\t',fmt='%i')
    
    if use_train_scale == "TRUE":
        # save scaling stats
        np.save(outloc+str(it_val)+"/train_x_iter_stat.mean.npy", train_x_iter_stat[0])
        np.save(outloc+str(it_val)+"/train_x_iter_stat.std.npy", train_x_iter_stat[1])
        np.save(outloc+str(it_val)+"/train_y_iter_stat.mean.npy", train_y_iter_stat[0])
        np.save(outloc+str(it_val)+"/train_y_iter_stat.std.npy", train_y_iter_stat[1])
        for i in range(n_mat):
            np.save(outloc+str(it_val)+"/train_x_mat_iter_stat"+str(i)+".mean.npy", train_x_mat_iter_stat[i][0])
            np.save(outloc+str(it_val)+"/train_x_mat_iter_stat"+str(i)+".std.npy", train_x_mat_iter_stat[i][1])
    
    # prediction 
    vae = VAE(n_mat, init_method).float()
    vae.cuda()
    vae.load_state_dict(torch.load(outloc+str(it_val)+"/"+'/trained_model.pt'))
    
    params = {'batch_size': mb_size,
          'shuffle': False,
          'num_workers': 0,
          'pin_memory':True}
          
    # predict testing data
    with torch.no_grad():
        train = Dataset(test_x_iter,test_y_iter,test_x_mat_iter)
        training_generator = data.DataLoader(train, 
                                             **params)
        vae.eval()
        for i,(X, X_mat, Y) in enumerate(training_generator):
            X = X.float().cuda()
            for i_mat in range(n_mat):
                X_mat[i_mat][torch.isnan(X_mat[i_mat])]=0
                X_mat[i_mat] = X_mat[i_mat].float().cuda()
            X_sample, z, z_mu, z_var, h = vae(X, X_mat)
            with open(outloc+str(it_val)+"/test_prediction.txt",'ab') as f:
                np.savetxt(f, X_sample.detach().cpu().numpy(), delimiter='\t') 
    subprocess.run(['gzip', outloc+str(it_val)+"/test_prediction.txt"])
    
    # predict target data
    if targetloc!="none":
        # scaling target data
        target_x_iter = copy.deepcopy(target_x)  
        target_y_iter = copy.deepcopy(target_y)  
        target_x_mat_iter = copy.deepcopy(target_x_mat)  
        if use_train_scale == "FALSE":
            target_x_iter, target_y_iter, target_x_mat_iter, _, _, _ = scaling(target_x_iter, target_y_iter, target_x_mat_iter)
        elif use_train_scale == "TRUE":
            target_x_iter, target_y_iter, target_x_mat_iter = scale_prefix(target_x_iter, target_y_iter, target_x_mat_iter,  train_x_iter_stat, train_y_iter_stat, train_x_mat_iter_stat)
            
        with torch.no_grad():
            train = Dataset(target_x_iter,target_y_iter,target_x_mat_iter)
            training_generator = data.DataLoader(train, 
                                                 **params)
            h_all = torch.tensor([])
            X_sample_all = torch.tensor([])
            vae.eval()
            for i,(X, X_mat, Y) in enumerate(training_generator):
                X = X.float().cuda()
                for i_mat in range(n_mat):
                    X_mat[i_mat][torch.isnan(X_mat[i_mat])]=0
                    X_mat[i_mat] = X_mat[i_mat].float().cuda()
                X_sample, z, z_mu, z_var, h = vae(X, X_mat)
                h_all = torch.cat((h_all,h.cpu()),0)
        for i_mat in range(n_mat):
            target_x_mat_iter[i_mat][np.isnan(target_x_mat_iter[i_mat])]=0
            target_x_mat_iter[i_mat] = torch.tensor(target_x_mat_iter[i_mat]).float().transpose(0,1)
        X_sample = vae.P.fc3(h_all.cuda())
        X_sample = X_sample.cpu()
        for i in range(n_mat):
            X_sample = target_x_mat_iter[i]*vae.weight[i].cpu() + X_sample
        X_sample = vae.bn(X_sample.cuda()).cpu()
        with open(outloc+str(it_val)+"/target_prediction.txt",'ab') as f:
            np.savetxt(f, X_sample.detach().cpu().numpy(), delimiter='\t') 
        subprocess.run(['gzip', outloc+str(it_val)+"/target_prediction.txt"])
