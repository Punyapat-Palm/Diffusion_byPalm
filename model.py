import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
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

    def forward(self, x, t):
        return self.unet(x, t)

    def beta(self, t):
        # Just a simple linear interpolation between beta_small and beta_large based on t
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        # Product of alphas from 0 to t
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i].item())  # Ensure this is an item() call
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0).to(self.device)
        e_hat = self.forward(noise_imgs, ts)
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, model, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            t = torch.tensor([t], device=x.device)  # Ensure t is on the same device as x
            e_hat = self.forward(x, t.repeat(x.shape[0]))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            z = torch.randn_like(x, device=x.device) if t.item() > 1 else 0  # Ensure z is on the same device
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x