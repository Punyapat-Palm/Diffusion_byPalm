import torch
import torch.nn as nn
import math
import numpy
from modules import *

class DiffusionModel(nn.Module):
    def __init__(self, in_size, t_range, img_depth, device):
        super().__init__()
        self.automatic_optimization = True
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size
        self.device = device
        self.unet = Unet(dim = 64, dim_mults = (1, 2, 4, 8), channels=img_depth)
        self.betas = self.beta(torch.arange(t_range, device=device))
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        return self.unet(x, t)

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return self.alpha_bars[t]

    def get_loss(self, batch):
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        epsilons = torch.randn(batch.shape, device=self.device)
        a_hat = self.alpha_bars[ts].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        noise_imgs = (a_hat.sqrt() * batch) + ((1 - a_hat).sqrt() * epsilons)
        e_hat = self.forward(noise_imgs, ts)
        loss = nn.functional.mse_loss(
            e_hat.view(-1, self.in_size), epsilons.view(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t):
        with torch.no_grad():
            t = torch.tensor([t], device=x.device)
            e_hat = self.forward(x, t.repeat(x.shape[0]))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            z = torch.randn_like(x, device=x.device) if t.item() > 1 else 0
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x
