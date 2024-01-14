import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim

from lib.langevin_daymics import anneal_langevin_dynamics
from lib.gmm import GMMDistAnneal
from lib.scorenet import ModelAnneal
from lib.loss import anneal_sliced_score_estimation_vr

def train(out_folder):
    gmm_dist = GMMDistAnneal()
    model = ModelAnneal()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    sigmas = torch.from_numpy(np.exp(np.linspace(np.log(20), 0., 10))).float()

    for step in range(10000):
        samples = gmm_dist.sample((128, ))
        sigmas_idx = torch.randint(0, 10, size=(128, ))
       
        loss = anneal_sliced_score_estimation_vr(model, samples, sigmas_idx, sigmas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step: {step}, loss: {loss.item()}")

    torch.save(model.state_dict(), osp.join(out_folder, "scorenet_anneal.pth"))

def test(out_folder):
    n_samples = 2000
    gmm_dist_anneal = GMMDistAnneal()
    gt_samples = gmm_dist_anneal.sample((n_samples, ))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].scatter(gt_samples[:, 0], gt_samples[:, 1], s=1)
    axs[0].set_xlim(-8, 8)
    axs[0].set_ylim(-8, 8)
    axs[0].set_title("Samples from GMM")

    model = ModelAnneal()
    model.load_state_dict(torch.load(osp.join(out_folder, "scorenet_anneal.pth")))

    right_bound = 8
    left_bound = -8

    sigmas = torch.from_numpy(np.exp(np.linspace(np.log(20), 0., 10))).float()
    samples = torch.rand(n_samples, 2) * (right_bound - left_bound) + left_bound
    samples = anneal_langevin_dynamics(model, samples, sigmas, save_gif=False).detach().numpy()
    
    axs[1].scatter(samples[:, 0], samples[:, 1], s=1, c="b")
    axs[1].set_title('Anneal Langevin dynamics model')
    axs[1].set_xlim(left_bound, right_bound)
    axs[1].set_ylim(left_bound, right_bound)

    plt.savefig(osp.join(out_folder, 'scorenet_anneal.jpg'))

if __name__ == "__main__":
    out_folder = "outputs"
    os.makedirs(out_folder, exist_ok=True)

    train(out_folder)
    test(out_folder)