import math
import numpy as np
import torch
import secrets

def privatize_vector(x,eps,k,mech,device,p=None,gamma=None,sigma=None,fast=False,W=None,use_GPU = True):
    n = len(x)
    if mech == 'CompressLDP':
      k = 1
      if use_GPU == True:
          return CompressLDP_Compress(x.reshape(1,len(x)),eps,k,device).squeeze()
      else:
          return DJW_no_gpu(x.reshape(1,len(x)),eps,k).squeeze()


def CompressLDP_Compress(X,eps,k,device):
    X_t = X.T
    Y_t = DJW_gpu_compress(X_t,eps,device)
    return Y_t.T

def generate_random_vector(d, given_seed):
    rng = np.random.default_rng(seed=given_seed)
    # generate d elements, each with 32 bits
    random_sequence = rng.standard_normal(d, dtype=np.float32)
    return random_sequence

def DJW_gpu_compress(X, eps, device):
  (d, n) = X.shape
   # Using {d \choose d/2} approximation
  B = (math.exp(eps)+1)/(math.exp(eps)-1)*np.sqrt(math.pi*d/2)
  pi_a = math.exp(eps)/(1+math.exp(eps))
  X_perturb = X.clone().to(device)
  for i in range(n):

    # client-side gradient compression
    seed = secrets.randbits(128)
    v = generate_random_vector(d, seed)
    v = torch.from_numpy(v).to(device)
    v = v/torch.linalg.norm(v, 2) # v uniform over l_2 unit ball
    if torch.sum(v * X[:, i]) < 0:
        v = -v
    T = 2*np.random.binomial(1,pi_a)-1

    # server-side gradient decompression
    u = generate_random_vector(d, seed)
    u = torch.tensor(v).to(device)
    u = u/torch.linalg.norm(v, 2) # v uniform over l_2 unit ball
    X_perturb[:, i] = T*u

  return B*X_perturb

def DJW_no_gpu(X,eps,k):
    X_t = X.T
    Y_t = DJW_nogpu(X_t,eps,k)
    return Y_t.T

def DJW_nogpu(X,eps,k):
    X_hat = np.zeros(X.shape)
    for i in range(k):
      X_perturb = DJW(X,1.0*eps/k)
      X_hat = X_hat + 1.0*X_perturb/k
    return X_hat

def DJW(X, eps):

  (d, n) = X.shape
   # Using {d \choose d/2} approximation
  B = (math.exp(eps)+1)/(math.exp(eps)-1)*np.sqrt(math.pi*d/2)
  pi_a = math.exp(eps)/(1+math.exp(eps))
  X_perturb = X.copy()
    
  for i in range(n):
    # only handle when X[:, i] is a unit vector
    v = np.random.normal(0, 1, size = d)
    v = v/np.linalg.norm(v, 2) # v uniform over l_2 unit ball
    if np.sum(v * X[:, i]) < 0:
        v = -v
    T = 2*np.random.binomial(1, pi_a)-1
    X_perturb[:, i] = T*v
  
  return B*X_perturb