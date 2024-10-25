import torch
import torch.nn as nn
import math
import numpy as np
from modules import *

class DiffusionModel(nn.Module):
    def __init__(self, in_size, t_range, img_depth, device):
        super().__init__()
        self.automatic_optimization = True
        self.t_range = t_range
        self.in_size = in_size
        self.device = device
        self.unet = Unet(dim = 64, dim_mults = (1, 2, 4, 8), channels=img_depth)
        self.betas = self.calculate_betas(self.t_range)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        return self.unet(x, t)
    
    def calculate_betas(self, t_range):
        betas = []
        for i in range(t_range):
            t1 = i / t_range
            t2 = (i + 1) / t_range
            alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
            beta = min(1 - alpha_bar_t2 / alpha_bar_t1, 0.999)
            betas.append(beta)
            
        return torch.tensor(betas, device=self.device)

    def beta(self, t):
        return self.betas[t]

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