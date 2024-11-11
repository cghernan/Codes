"""
Run the autoencoder PROM, and compare it to the HDM at an out-of-sample
point
"""

import os
import time
from lsqnonneg import lsqnonneg
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import pdb
import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import GPy

from hypernet2D import (load_or_compute_snaps, make_2D_grid,
                        plot_snaps, POD, inviscid_burgers_rnm2D_ecsw,
                        inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
                        compute_ECSW_training_matrix_2D_rnm)
from config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU
import GPy

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, (ax1, ax2) = plt.subplots(2, 1)
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(grid_x, grid_y, snaps, inds_to_plot,
               label=labels[i],
               fig_ax=(fig, ax1, ax2),
               color=colors[i],
               linewidth=linewidths[i])

  return fig, ax1, ax2

def main(mu1=4.875, mu2=0.0225,compute_ecsw=True):
#def main(mu1=4.75, mu2=0.02, compute_ecsw=False):

    t1 = time.time()
    snap_folder = 'param_snaps'

    # Query point of HPROM-GP
    mu_rom = [mu1, mu2]

    # Sample point for ECSW
    mu_samples = [[4.25, 0.0225]]

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
    q = np.load('coords.npy')
    q0 = q[0:nc,:]

    # evaluate ROM at mu_rom
    mu_rom_backup = [v for v in mu_rom]

    basis = np.load('basis.npy')
    ref = numpy.zeros_like(basis[:, 0]).squeeze()
    basis1 = basis[:, :nc]
    basis2 = basis[:, nc:nt]
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    #ECSW
    if compute_ecsw:
        snap_sample_factor = 10

        Clist = []
        for imu, mu in enumerate(mu_samples):
          mu_snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
          def decode(x,gpm, gpScalings):
              return basis @ x + basis2 @ rnm(x,gpm, gpScalings)

          def jacob(x,gpm, gpScalings):
              return basis + basis2 @ rnm_grad(x,gpm, gpScalings)

          print('Generating training block for mu = {}'.format(mu))
          Ci = compute_ECSW_training_matrix_2D_rnm(mu_snaps[:, 2*imu+3:num_steps:snap_sample_factor],
                                                   mu_snaps[:, 2*imu+0:num_steps - 3 - 2*imu:snap_sample_factor],
                                                   basis, decode, jacob, inviscid_burgers_res2D,
                                                   inviscid_burgers_exact_jac2D, grid_x, grid_y, dt, mu)
          Clist += [Ci]

        C = np.vstack(Clist)
        idxs = np.zeros((num_cells_y, num_cells_x))

        # Select boundaries
        nn_x = 1
        nn_y = 1
        idxs[nn_y:-nn_y, nn_x:-nn_x] = 1
        C = C[:, (idxs == 1).ravel()]

        # Weighting for boundary
        bc_w = 10

        t1 = time.time()
        weights, _ = nnls(C, C.sum(axis=1), maxiter=99999999)
        print('nnls solve time: {}'.format(time.time() - t1))

        print('nnls solver residual: {}'.format(
          np.linalg.norm(C @ weights - C.sum(axis=1)) / np.linalg.norm(C.sum(axis=1))))

        weights = weights.reshape((num_cells_y - 2*nn_y, num_cells_x - 2*nn_x))
        full_weights = bc_w*np.ones((num_cells_y, num_cells_x))
        full_weights[idxs > 0] = weights.ravel()
        weights = full_weights.ravel()
        np.save('ecsw_weights_gp', weights)
        plt.clf()
        plt.rc('font', size=16)
        plt.spy(weights.reshape((250, 250)))
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
        plt.title('PROM-GP Reduced Mesh')
        plt.tight_layout()
        plt.savefig('prom-gp-reduced-mesh.png', dpi=300)
        plt.show()
    else:
        weights = np.load('ecsw_weights_gp.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))
    #END ECSW


    t0 = time.time()
    ys, man_times = inviscid_burgers_rnm2D_ecsw(grid_x, grid_y, w0, dt, num_steps, mu_rom_backup, rnm, rnm_grad, ref, basis,
                                                       basis2, weights)
    man_its, man_jac, man_res, man_ls = man_times
    print('Elapsed time: {:3.3e}'.format(time.time() - t0))
    inds_to_plot = range(0, 501, 100)

    def decode(x,gpm, gpScalings):
        return basis @ x + basis2 @ rnm(x,gpm, gpScalings)

    def jacob(x,gpm, gpScalings):
        return basis + basis2 @ rnm_grad(x,gpm, gpScalings)

    man_snaps = np.array([decode(ys[:, i],gpm, gpScalings) for i in range(ys.shape[1])]).T
    snaps_to_plot = [hdm_snaps, man_snaps]
    labels = ['HDM', 'HPROM-GP']
    colors = ['black', 'green']
    linewidths = [2, 1]
    fig, ax1, ax2 = compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths)
    print('gp_its: {:3.2f}, gp_jac: {:3.2f}, gp_res: {:3.2f}, gp_ls: {:3.2f}'.format(man_its, man_jac, man_res, man_ls))
    print('Relative error: {:3.2f}%'.format(100*np.linalg.norm(hdm_snaps - man_snaps)/np.linalg.norm(hdm_snaps)))

    ax1.legend(), ax2.legend()
    plt.tight_layout()
    print('Saving as "hprom-gp_{:3.2f}_{:3.2f}_n{}_nbar{}.png"'.format(mu_rom[0], mu_rom[1], nc, nt-nc))
    plt.savefig('hprom-gp_{:3.2f}_{:3.2f}_n{}_nbar{}.png'.format(mu_rom[0], mu_rom[1], nc, nt-nc), dpi=300)
    plt.show()

    mse = sum([np.linalg.norm(hdm_snaps[:, c] - man_snaps[:, c])
               for c in range(hdm_snaps.shape[1])]) / sum(np.linalg.norm(hdm_snaps[:, c])
                                                          for c in range(hdm_snaps.shape[1]))
    print('Time: {}'.format(time.time() - t1))

    print('MSE: {:3.2f}%'.format(100*mse))
    return mse


if __name__ == "__main__":
    main(compute_ecsw=True)
