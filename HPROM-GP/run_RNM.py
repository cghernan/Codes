"""
Run the autoencoder PROM, and compare it to the HDM at an out-of-sample
point
"""

import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import pdb
import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import GPy

from hypernet2D import (load_or_compute_snaps, make_2D_grid, inviscid_burgers_implicit2D_LSPG,
                      inviscid_burgers_rnm2D, plot_snaps, POD, rnm, rnm_grad, GP_load)

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=16)

def compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, (ax1, ax2) = plt.subplots(2, 1)
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(grid_x, grid_y, snaps, inds_to_plot,
               label=labels[i],
               fig_ax=(fig, ax1, ax2),
               color=colors[i],
               linewidth=linewidths[i])

  return fig, ax1, ax2

#def main(mu1=5.2, mu2=0.027):
#def main(mu1=4.75, mu2=0.02):
#def main(mu1=4.875, mu2=0.0225):
#def main(mu1=5.5, mu2=0.0225):
#def main(mu1=4.56, mu2=0.019):
def main(mu1=5.19, mu2=0.026):

    t = time.time()
    snap_folder = 'param_snaps'

    mu_rom = [mu1, mu2]

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))

    nc = 10
    nt = 150

    basis = np.load('basis.npy')
    ref = numpy.zeros_like(basis[:, 0]).squeeze()
    basis1 = basis[:, :nc]
    basis2 = basis[:, nc:nt]
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
    man_snaps, man_times, qbar = inviscid_burgers_rnm2D(grid_x, grid_y, w0, dt, num_steps, mu_rom, rnm, rnm_grad, ref, basis1, basis2)
    man_its, man_jac, man_res, man_ls = man_times

    snaps_to_plot = [hdm_snaps, man_snaps]
    labels = ['HDM','PROM-GP']
    colors = ['black', 'green']
    linewidths = [2, 1]
    inds_to_plot = range(0, 501, 100)
    fig, ax1, ax2 = compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths)
    print('Solution relative error: {:3.2f}%'.format(100 * np.linalg.norm(hdm_snaps - man_snaps) / np.linalg.norm(hdm_snaps)))
    print('rnm_its: {:3.2f}, rnm_jac: {:3.2f}, rnm_res: {:3.2f}, rnm_ls: {:3.2f}'.format(man_its, man_jac, man_res, man_ls))
    mse = sum([np.linalg.norm(hdm_snaps[:, c] - man_snaps[:, c])
               for c in range(hdm_snaps.shape[1])]) / sum(np.linalg.norm(hdm_snaps[:, c])
                                                          for c in range(hdm_snaps.shape[1]))
    print('MSE: {:3.2f}%'.format(100*mse))

    ax1.legend(), ax2.legend()
    plt.tight_layout()
    print('gp-prom-sca_gpytorch.png'.format(mu_rom[0], mu_rom[1], nc, nt - nc))
    plt.savefig('gp-prom-sca_gpytorch.png'.format(mu_rom[0], mu_rom[1], nc, nt - nc),dpi=300)

    print('Time: {}'.format(time.time() - t))
    print('qbar relative error: {:3.2f}%'.format(100 * np.linalg.norm(basis2.T @ hdm_snaps - qbar) / np.linalg.norm(basis2.T @ hdm_snaps)))

if __name__ == "__main__":
    main()
