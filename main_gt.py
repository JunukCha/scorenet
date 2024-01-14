import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch

from lib.gmm import GMMDist, GMMDistAnneal
from lib.langevin_daymics import langevin_dynamics, anneal_langevin_dynamics

def run_ld(out_folder):
    '''
    Run langevin dynamics
    '''

    n_samples = 2000
    gmm_dist = GMMDist()
    
    samples = gmm_dist.sample((n_samples, ))

    fig, axs = plt.subplots(1, 4, figsize=(20, 5)) 

    axs[0].scatter(samples[:, 0], samples[:, 1], s=1)
    axs[0].set_xlim(-8, 8)
    axs[0].set_ylim(-8, 8)
    axs[0].set_title("Samples from GMM")

    x1, x2 = np.mgrid[-8:8:0.8, -8:8:0.8]
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    mesh = torch.stack([x1, x2], dim=2)
    scores = gmm_dist.score(mesh)
    
    axs[1].quiver(mesh[:, :, 0], mesh[:, :, 1], scores[:, :, 0], scores[:, :, 1], width=0.005)
    axs[1].set_title('Data scores')
    axs[1].axis('square')

    uniform_pts = np.random.uniform(-8, 8, size=(n_samples, 2))
    uniform_pts = torch.from_numpy(uniform_pts)
    ld_results = langevin_dynamics(gmm_dist.score, uniform_pts, save_gif=True)
    
    axs[2].scatter(ld_results[-1][:, 0], ld_results[-1][:, 1], s=1, c="b")
    axs[2].set_xlim(-8, 8)
    axs[2].set_ylim(-8, 8)
    axs[2].set_title("LD results")

    axs[3].set_title('LD resutls anim.')
    ln, = axs[3].plot([], [], 'bo', markersize=1)

    def init():
        axs[3].set_xlim(-8, 8)
        axs[3].set_ylim(-8, 8)
        return ln,

    def update(frame):
        xdata = ld_results[frame][:, 0]
        ydata = ld_results[frame][:, 1]
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, interval=50)

    plt.tight_layout()
    # plt.show()
    ani.save(osp.join(out_folder, 'langevin_dynamics.gif'))

def run_ald(out_folder):
    '''
    Run anneal langevin dynamics
    '''
    n_samples = 2000
    gmm_dist_anneal = GMMDistAnneal()
    
    sigmas = torch.from_numpy(np.exp(np.linspace(np.log(20), 0., 10))).float()

    fig, axs = plt.subplots(3, 3, figsize=(15, 15)) 

    samples = gmm_dist_anneal.sample((n_samples, ))
    axs[0, 0].scatter(samples[:, 0], samples[:, 1], s=1)
    axs[0, 0].set_xlim(-8, 8)
    axs[0, 0].set_ylim(-8, 8)
    axs[0, 0].set_title("Samples from GMM")

    uniform_pts = np.random.uniform(-8, 8, size=(n_samples, 2))
    uniform_pts = torch.from_numpy(uniform_pts)
    ld_results = anneal_langevin_dynamics(gmm_dist_anneal.score, uniform_pts, sigmas, save_gif=True, gt=True)
    
    axs[0, 1].scatter(ld_results[-1][:, 0], ld_results[-1][:, 1], s=1, c="b")
    axs[0, 1].set_xlim(-8, 8)
    axs[0, 1].set_ylim(-8, 8)
    axs[0, 1].set_title("ALD results")

    axs[0, 2].set_title('ALD resutls anim.')
    ln, = axs[0, 2].plot([], [], 'bo', markersize=1)

    def init():
        axs[0, 2].set_xlim(-8, 8)
        axs[0, 2].set_ylim(-8, 8)
        return ln,

    def update(frame):
        xdata = ld_results[frame][:, 0]
        ydata = ld_results[frame][:, 1]
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, interval=50)

    x1, x2 = np.mgrid[-8:8:0.8, -8:8:0.8]
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    mesh = torch.stack([x1, x2], dim=2)
    for i, sigma_idx in enumerate([0, 2, 4, 5, 6, 9]):
        row, col = (i + 3) // 3, (i + 3) % 3
        scores = gmm_dist_anneal.score(mesh, sigma=sigmas[sigma_idx])
        axs[row, col].quiver(mesh[:, :, 0], mesh[:, :, 1], scores[:, :, 0], scores[:, :, 1], width=0.005)
        axs[row, col].set_title(f'Data scores | sigma: {sigmas[sigma_idx]:.3f} | idx: {sigma_idx}')
        axs[row, col].axis('square')

    plt.tight_layout()
    # plt.show()
    ani.save(osp.join(out_folder, 'anneal_langevin_dynamics.gif'))

def run_comp(out_folder):
    '''
    Run langevin dynamics
    '''

    n_samples = 2000
    gmm_dist = GMMDist()
    
    samples = gmm_dist.sample((n_samples, ))

    fig, axs = plt.subplots(1, 3, figsize=(20, 5)) 

    axs[0].scatter(samples[:, 0], samples[:, 1], s=1)
    axs[0].set_xlim(-8, 8)
    axs[0].set_ylim(-8, 8)
    axs[0].set_title("Samples from GMM")
    
    uniform_pts = np.random.uniform(-8, 8, size=(n_samples, 2))
    uniform_pts = torch.from_numpy(uniform_pts)
    ld_results = langevin_dynamics(gmm_dist.score, uniform_pts, save_gif=True)
    
    axs[1].scatter(ld_results[-1][:, 0], ld_results[-1][:, 1], s=1, c="b")
    axs[1].set_xlim(-8, 8)
    axs[1].set_ylim(-8, 8)
    axs[1].set_title("LD results")

    gmm_dist_anneal = GMMDistAnneal()
    
    sigmas = torch.from_numpy(np.exp(np.linspace(np.log(20), 0., 10))).float()

    ld_results = anneal_langevin_dynamics(gmm_dist_anneal.score, uniform_pts, sigmas, save_gif=True, gt=True)
    
    axs[2].scatter(ld_results[-1][:, 0], ld_results[-1][:, 1], s=1, c="b")
    axs[2].set_xlim(-8, 8)
    axs[2].set_ylim(-8, 8)
    axs[2].set_title("ALD results")

    plt.tight_layout()
    plt.savefig(osp.join(out_folder, 'comparision_anneal_baseline.jpg'))
    

if __name__ == "__main__":
    out_folder = "outputs"
    os.makedirs(out_folder, exist_ok=True)

    run_ld(out_folder)
    run_ald(out_folder)
    run_comp(out_folder)