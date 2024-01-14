import numpy as np

import torch
import torch.autograd as autograd


class GMMDist(object):
    def __init__(self):
        self.mix_probs = torch.tensor([0.8, 0.2])
        self.means = torch.stack([5 * torch.ones(2), -torch.ones(2) * 5], dim=0)
        self.sigma = 1
        self.std = torch.stack([torch.ones(2) * self.sigma for i in range(len(self.mix_probs))], dim=0)

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log(
                2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp

    def score(self, x):
        x = x.detach()
        x.requires_grad_(True)
        y = self.log_prob(x).sum()
        return autograd.grad(y, x)[0]
    

class GMMDistAnneal(object):
    def __init__(self):
        self.mix_probs = torch.tensor([0.8, 0.2])
        self.means = torch.stack([5 * torch.ones(2), -torch.ones(2) * 5], dim=0)

    def sample(self, n, sigma=1):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        return torch.randn_like(means) * sigma + means

    def log_prob(self, samples, sigma=1):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * sigma ** 2) - 0.5 * np.log(
                2 * np.pi * sigma ** 2 + 1e-6)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp

    def score(self, x, sigma=1):
        x = x.detach()
        x.requires_grad_(True)
        y = self.log_prob(x, sigma).sum()
        return autograd.grad(y, x)[0]