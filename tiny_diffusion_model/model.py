
import torch
from torch import nn
from itertools import pairwise


class ModelMixin:
    def rand_input(self, batchsize):
        assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
        return torch.randn((batchsize,) + self.input_dims)

    # Currently predicts eps, override following methods to predict, for example, x0
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(eps, self(x0 + sigma * eps, sigma, cond=cond))

    def predict_eps(self, x, sigma, cond=None):
        return self(x, sigma, cond=cond)

    def predict_eps_cfg(self, x, sigma, cond, cfg_scale):
        if cond is None or cfg_scale == 0:
            return self.predict_eps(x, sigma, cond=cond)
        assert sigma.shape == tuple(), 'CFG sampling only supports singleton sigma!'
        uncond = torch.full_like(cond, self.cond_embed.null_cond) # (B,)
        eps_cond, eps_uncond = self.predict_eps(                  # (B,), (B,)
            torch.cat([x, x]), sigma, torch.cat([cond, uncond])   # (2B,)
        ).chunk(2)
        return eps_cond + cfg_scale * (eps_cond - eps_uncond)

def get_sigma_embeds(batches, sigma, scaling_factor=0.5, log_scale=True):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    if log_scale:
        sigma = torch.log(sigma)
    s = sigma.unsqueeze(1) * scaling_factor
    return torch.cat([torch.sin(s), torch.cos(s)], dim=1)


def get_sigma_embeds(batches, sigma, scaling_factor = 0.5, log_scale=True):
  if sigma.shape == torch.Size([]):
    sigma = sigma.unsqueeze(0).repeat(batches)
  else:
    assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
  if log_scale:
    sigma = torch.log(sigma)

  s = sigma.unsqueeze(1)*scaling_factor
  return torch.cat([torch.sin(s),torch.cos(s)], dim=1)

class TimeInputMLP(nn.Module, ModelMixin):

    " This is the simple architecture we are using"


    sigma_dim = 2
    def __init__(self, dim=2, output_dim=None, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + self.sigma_dim,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim or dim))

        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma, cond=None):
        #x     shape:b x dim
        #sigma shape:b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) #shape: b x 2
        nn_input = torch.cat([x, sigma_embeds], dim=1)               #shape: b x (dim + 2)
        return self.net(nn_input)