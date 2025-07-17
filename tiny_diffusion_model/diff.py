import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import pairwise
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from types import SimpleNamespace
from typing import Optional, Union, Tuple
from accelerate import Accelerator
from tqdm import tqdm



#We also need to define the noise schedules, which are given by:

class Schedule:
  def __init__(self,sigmas:torch.FloatTensor):
    self.sigmas = sigmas
  def __getitem__(self,i) -> torch.FloatTensor:
    return self.sigmas[i]
  def __len__(self)-> int:
    return len(self.sigmas)
  def sample_batch(self, x0:torch.FloatTensor) -> torch.FloatTensor:
    return self[torch.randint(len(self),(x0.shape[0],))].to(x0)
  def sample_sigmas(self, steps: int) -> torch.FloatTensor:
    ''' Thsis functions is called during sampling for a given number of specified sampling steps '''
    indices = list((len(self)*(1-np.arange(0,steps)/steps)).round().astype(np.int64)-1)

    return self[indices + [0]]

#Following the tutorial, we are going to use the LogLinear schedule:

class ScheduleLogLinear(Schedule):
  def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float = 10):
    super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max),N))


def generate_train_sample(x0: Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]],
                          schedule: Schedule, conditional: bool=False):
    cond = x0[1] if conditional else None
    x0   = x0[0] if conditional else x0
    sigma = schedule.sample_batch(x0)
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    eps = torch.randn_like(x0)
    return x0, sigma, eps, cond


#Including a way to visualize the batch data
def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')

#Including a way to compute the moving avarage    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def training_loop(loader: DataLoader,
                  model : nn.Module,
                  schedule : Schedule,
                  accelerator : Optional[Accelerator] = None,
                  epochs : int = 10000,
                  lr : float = 1e-3,
                  conditional : bool = False):
  
  accelerator = accelerator or Accelerator()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

  for _ in (pbar := tqdm(range(epochs))):
    for x0 in loader:
      model.train()
      optimizer.zero_grad()
      
      x0,sigma,eps,cond = generate_train_sample(x0, schedule, conditional)
      
      loss = model.get_loss(x0,sigma,eps,cond=cond)

      yield SimpleNamespace(**locals())

      accelerator.backward(loss)
      optimizer.step()


@torch.no_grad()
def samples(model : nn.Module,
            
            sigmas : torch.FloatTensor,

            gam   : float = 1.,
            
            mu   : float = 0.,

            cfg_scale : int = 0.,

            batchsize : int = 1,

            xt : Optional[torch.FloatTensor] = None,

            cond : Optional[torch.FloatTensor] = None,
            
            accelerator: Optional[Accelerator] =  None):
  
  model.eval()

  accelerator = accelerator or Accelerator()

  xt = model.rand_input(batchsize).to(accelerator.device)*sigmas[0] if xt is None else xt

  if cond is not None:

    assert cond.shape[0] == xt.shape[0], 'cond must have same shape as x!'

    cond = cond.to(xt.device)

  eps = None

  for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):

    eps_prev, eps = eps, model.predict_eps_cfg(xt,sig.to(xt), cond, cfg_scale)

    eps_av = eps*gam + eps_prev * (1-gam) if i>0 else eps

    sig_p = (sig_prev/sig**mu)**(1/(1-mu))

    eta = (sig_prev**2 - sig_p**2).sqrt()

    xt = xt - (sig - sig_p)*eps_av+eta*model.rand_input(xt.shape[0]).to(xt)

    yield xt


