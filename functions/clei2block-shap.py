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
import sys
import pandas as pd
import shap
import subprocess
import math

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_x, data_x_mat):
        'Initialization'
        self.data_x = data_x
        self.data_x_mat = data_x_mat
    def __len__(self):
        'Denotes the total number of samples'
        return self.data_x.shape[1]
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.data_x[:,index]
        X_mat = [self.data_x_mat[i][:,index] for i in range(len(self.data_x_mat))]
        return X, X_mat
    
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
        return X_sample
    
class VAE_3(nn.Module):
    def __init__(self,n_mat,init_method):
        super(VAE_3,self).__init__()
        self.Q = Q()
        self.P = P()
        self.weight=nn.ParameterList([nn.Parameter(init_param([1,y_dim],init_method),requires_grad=True) for i in range(n_mat)])
        if use_train_scale == "TRUE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=True)
        elif use_train_scale == "FALSE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=False)
    def forward(self, X, X_mat1, X_mat2, X_mat3):
        z_mu, z_var = self.Q(X)
        z = sample_z(z_mu, z_var, X.shape[0])
        X_sample, h = self.P(z)
        X_sample = X_mat1*self.weight[0]+X_sample
        X_sample = X_mat2*self.weight[1]+X_sample
        X_sample = X_mat3*self.weight[2]+X_sample
        X_sample = self.bn(X_sample)
        return X_sample
        
class VAE_2(nn.Module):
    def __init__(self,n_mat,init_method):
        super(VAE_2,self).__init__()
        self.Q = Q()
        self.P = P()
        self.weight=nn.ParameterList([nn.Parameter(init_param([1,y_dim],init_method),requires_grad=True) for i in range(n_mat)])
        if use_train_scale == "TRUE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=True)
        elif use_train_scale == "FALSE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=False)
    def forward(self, X, X_mat1, X_mat2):
        z_mu, z_var = self.Q(X)
        z = sample_z(z_mu, z_var, X.shape[0])
        X_sample, h = self.P(z)
        X_sample = X_mat1*self.weight[0]+X_sample
        X_sample = X_mat2*self.weight[1]+X_sample
        X_sample = self.bn(X_sample)
        return X_sample
        
class VAE_1(nn.Module):
    def __init__(self,n_mat,init_method):
        super(VAE_1,self).__init__()
        self.Q = Q()
        self.P = P()
        self.weight=nn.ParameterList([nn.Parameter(init_param([1,y_dim],init_method),requires_grad=True) for i in range(n_mat)])
        if use_train_scale == "TRUE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=True)
        elif use_train_scale == "FALSE":
            self.bn = nn.BatchNorm1d(y_dim, affine=False,track_running_stats=False)
    def forward(self, X, X_mat1):
        z_mu, z_var = self.Q(X)
        z = sample_z(z_mu, z_var, X.shape[0])
        X_sample, h = self.P(z)
        X_sample = X_mat1*self.weight[0]+X_sample
        X_sample = self.bn(X_sample)
        return X_sample
    
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
    
def scaling(test_x, test_x_mat):
    test_x, test_x_stat = scale_sub(test_x)
    test_x_mat_stat = [None] * len(test_x_mat)
    for j in range(len(test_x_mat)):
        test_x_mat[j], test_x_mat_stat[j] = scale_sub(test_x_mat[j])
    return test_x, test_x_mat, test_x_stat, test_x_mat_stat
    
def scale_prefix_sub(test_x,train_x_iter_stat):
    for i in range(test_x.shape[0]):
        test_x[i,:] = test_x[i,:] - train_x_iter_stat[0][i]
        test_x[i,:] = test_x[i,:] / train_x_iter_stat[1][i]
    return test_x
    
def scale_prefix(test_x, test_x_mat, train_x_iter_stat, train_x_mat_iter_stat):
    test_x = scale_prefix_sub(test_x, train_x_iter_stat)
    for j in range(len(test_x_mat)):
        test_x_mat[j] = scale_prefix_sub(test_x_mat[j],train_x_mat_iter_stat[j])
    return test_x, test_x_mat

argv = sys.argv
argc = len(argv)
if (argc < 5):
    print ('Usage: python')
    quit()
inloc = argv[1]
model_loc = argv[2]
outloc = argv[3]

vae_input = argv[4]
#out_index = [int(x) for x in argv[6].split(',')]
lr_input = argv[5:]
mb_size = 400
n_mat=len(lr_input)
init_method = "normal"
use_train_scale = "FALSE"
n_gene_shap = 100
# =============================== DATA LOADING ====================================
data_split="test"
test_x = np.genfromtxt(inloc+"/"+data_split+"_vae_x_"+vae_input+".txt.gz")
test_y = np.genfromtxt(inloc+"/"+data_split+"_y.txt.gz")
test_x_mat = [np.genfromtxt(inloc+"/"+data_split+"_lr_x_"+v+".txt.gz") for v in lr_input]

# scaling target data
test_x_iter = copy.deepcopy(test_x)  
test_x_mat_iter = copy.deepcopy(test_x_mat)  
if use_train_scale == "FALSE":
    test_x_iter, test_x_mat_iter, _, _, = scaling(test_x_iter, test_x_mat_iter)
elif use_train_scale == "TRUE":
    test_x_iter, test_x_mat_iter = scale_prefix(test_x_iter, test_x_mat_iter,  test_x_iter_stat, test_x_mat_iter_stat)


params = {'batch_size': mb_size,
      'shuffle': False,
      'num_workers': 0,
      'pin_memory':True}

train = Dataset(test_x_iter,test_x_mat_iter)
training_generator = data.DataLoader(train, 
                                     **params)
X, X_mat = next(iter(training_generator))

X_isnan = torch.isnan(X)
X[X_isnan]=0
#print(i)
X = X.float().cuda()
X_mat_isnan=[]
for i_mat in range(n_mat):
    X_mat_isnan.append(torch.isnan(X_mat[i_mat]))
    X_mat[i_mat][X_mat_isnan[i_mat]]=0
    X_mat[i_mat] = X_mat[i_mat].float().cuda()

# =============================== SHAP ====================================
# create model
vae_state = torch.load(model_loc)
X_dim = vae_state['Q.fc1.weight'].shape[1]
h_dim = vae_state['Q.fc1.weight'].shape[0]
Z_dim = vae_state['Q.fc_mu.weight'].shape[0]
y_dim = 1 #vae_state['P.fc3.weight'].shape[0]
if n_mat==3:
    vae = VAE_3(n_mat, init_method).float()
elif n_mat==2:
    vae = VAE_2(n_mat, init_method).float()
elif n_mat==1:
    vae = VAE_1(n_mat, init_method).float()
vae.cuda()

# run Grandient Explainer
shap_store = []
index = np.random.permutation(vae_state['weight.0'].shape[1])[0:np.min([n_gene_shap,vae_state['weight.0'].shape[1]])]

#shap_raw_store = {}
#for j in out_index:
#    shap_raw_store[j]=[]

for i in index:
    #print(i)
    vae_state_temp = copy.deepcopy(vae_state)
    vae_state_temp['weight.0']=vae_state_temp['weight.0'][:,i:(i+1)]
    vae_state_temp['weight.1']=vae_state_temp['weight.1'][:,i:(i+1)]
    vae_state_temp['weight.2']=vae_state_temp['weight.2'][:,i:(i+1)]
    vae_state_temp['P.fc3.weight']=vae_state_temp['P.fc3.weight'][i:(i+1),:]
    vae.load_state_dict(vae_state_temp)
    
    e = shap.GradientExplainer(vae,[X, *[x[:,i:(i+1)] for x in X_mat]])
    shap_values = e.shap_values([X,*[x[:,i:(i+1)] for x in X_mat]], nsamples=200)
    shap_values[0][X_isnan] =  float("NaN")
    for i_mat in range(n_mat):
        shap_values[i_mat+1][X_mat_isnan[i_mat][:,i:(i+1)]] =  float("NaN")
    #for j in out_index:
    #    shap_raw_store[j].append(shap_values[0][:,j])
    shap_store.append(np.concatenate([np.nanmean(np.abs(x),axis=0) for x in shap_values]))

# export
shap_store = pd.DataFrame(np.concatenate([index[:, np.newaxis],np.concatenate([x[:, np.newaxis] for x in shap_store],axis=1).T],axis=1))
shap_store.columns=['gene_index'] + ["X"+i for i in (np.arange(X.shape[1])+1).astype(str)]+lr_input
shap_store.to_csv(outloc+'/shap.txt',sep="\t",index=False)
subprocess.run(['gzip', outloc+"/shap.txt"])

#for j in out_index:
#    shap_raw_store[j] = pd.DataFrame(np.concatenate([index[:, np.newaxis],np.concatenate([x[:, np.newaxis] for x in shap_raw_store[j]],axis=1).T],axis=1))
#    shap_raw_store[j].columns =['gene_index'] + ["sample"+i for i in (np.arange(X.shape[0])+1).astype(str)] 
#    shap_raw_store[j].to_csv(outloc+'/shap_raw_X'+str(j)+'.txt',sep="\t",index=False)
#    subprocess.run(['gzip', outloc+'/shap_raw_X'+str(j)+'.txt'])
