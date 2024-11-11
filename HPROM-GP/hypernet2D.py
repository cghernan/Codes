"""
Use the Burgers equation to try out some learning-based hyper-reduction approaches
"""

import glob
import math
import time

import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sklearn.cluster as clust
import torch
import functorch
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

import pickle
import pdb

import matplotlib
import GPy

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=16)

def GP_load():

    t = time.time()
    nc = 10
    nt = 150
    q = np.load('coords.npy')
    q0 = q[0:nc,:]
    q1 = q[nc:nt,:]
    #print('Reduced coordinates read time: ', time.time() - t)

    # Read the stored model training data and model data
    with open('gp_21sept.pkl', 'rb') as fb:
         gp = pickle.load(fb)
    ixTrain = np.load('ixTrain_21sept.npy')
    ns = q1.shape[1]
    ixTest = list(set(range(ns)))
    ixTestE = list(set(range(ns)) - set(ixTrain))
    nTrain = len(ixTrain)

    # Scaling factors
    txmin = np.min(q0, axis=1)
    txmax = np.max(q0, axis=1)
    tymin = np.min(q1, axis=1)
    tymax = np.max(q1, axis=1)

    q1Pred = np.zeros_like(q1)

    # Scale test data
    txTest = q0[:, ixTest]
    tyTest = q1[:, ixTest]
    xTest = 2.0 * ((txTest.T - txmin) / (txmax - txmin)) - 1.0
    yTest = 2.0 * ((tyTest.T - tymin) / (tymax - tymin)) - 1.0

    # Scale training data
    txTrain = q0[:, ixTrain]
    tyTrain = q1[:, ixTrain]
    xTrain = 2.0 * ((txTrain.T - txmin) / (txmax - txmin)) - 1.0
    yTrain = 2.0 * ((tyTrain.T - tymin) / (tymax - tymin)) - 1.0

    gpm = []
    gpScalings = []    
    array_list = []
    kernel_list = []

    # Predict
    yPred = gp.predict(xTest, return_std=False)
    print('Total: ', np.linalg.norm(yTest - yPred) / np.linalg.norm(yTest))
    gpScalings.append([txmin, txmax, tymin, tymax])
    
    print('GP load and setup time: ', time.time() - t)
    return (gp, gpScalings, xTrain)


def rnm(ic, gp, gpScalings, xTrain):

    t = time.time()
    nc = ic.shape[0]
    nt = nc + len(gpScalings[0][3])
    ic = np.expand_dims(ic, axis=1)
    q1NewPred_manual = np.zeros((nt-nc,1))
    txTest = ic
    txmin = gpScalings[0][0]
    txmax = gpScalings[0][1]
    tymin = gpScalings[0][2]
    tymax = gpScalings[0][3]
    xTest = 2.0 * ((txTest.T - txmin) / (txmax - txmin)) - 1.0
    # Use the recreated models
    #breakpoint()
    yPred = gp.predict(xTest, return_std=False)
    
    q1NewPred_manual = (((yPred + 1.0) / 2.0) * (tymax - tymin) + tymin).T

    print('rnm time: ', time.time() - t)
    return np.squeeze(q1NewPred_manual)


def rnm_grad(ic, gp, gpScalings, xTrain):
    t = time.time()
    nc = ic.shape[0]
    nt = nc + len(gpScalings[0][3])
    ic = np.expand_dims(ic, axis=1)
    q1NewPredGrad_manual = np.zeros((nt-nc,ic.shape[0]))
    eps = 1e-4
    txmin = gpScalings[0][0]
    txmax = gpScalings[0][1]
    tymin = gpScalings[0][2]
    tymax = gpScalings[0][3]
    txTest = ic
    xTest = 2.0 * ((txTest.T - txmin) / (txmax - txmin)) - 1.0    

    xTestP_mat = np.zeros((nc,nc))
    xTestP_mat = np.repeat(xTest, repeats=10, axis=0)
    xTestP_mat = xTestP_mat + eps*np.identity(nc) 

    diff_vec_manual_in = np.zeros(nc)

    # Use the recreated models

    yPrd_manual = gp.predict(xTest, return_std=False)

    diff_vec_manual = np.zeros(nc)
    yPrd_mat = gp.predict(xTestP_mat, return_std=False)

    diff_vec_manual = np.squeeze(yPrd_mat  - yPrd_manual) / eps
    #breakpoint()
    q1NewPredGrad_manual = ((tymax - tymin)/2.0 * diff_vec_manual).T * (2.0 / (txmax - txmin))

    print('rnm grad time: ', time.time() - t)    
    return q1NewPredGrad_manual


def inviscid_burgers_rnm2D(grid_x, grid_y, w0, dt, num_steps, mu, rnm, rnm_grad, ref, basis, basis2):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG manifold PROM for a parameterized inviscid 1D burgers
    problem with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    # stuff for operators
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()
    shp = (grid_y.size - 1, grid_y.size - 1)
    size = shp[0] * shp[1]
    idx = np.arange(size).reshape(shp).T.flatten()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))
    Jop = sp.bmat([[JDxec, None], [None, JDyec]])

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    y0 = basis.T @ w0.squeeze()

    gp, gpScalings, xTrain= GP_load()
    print(2)
    print(gp)
    #rnm(y0, gp, gpScalings, xTrain)
    #print(1)
    #print(rnm(y0, gp, gpScalings, xTrain, array_list).shape)

    w0 = basis @ y0 + basis2 @ rnm(y0, gp, gpScalings, xTrain)
    nred = y0.shape[0]
    snaps = np.zeros((w0.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    qbar = np.zeros((basis2.shape[1], num_steps + 1))
    snaps[:, 0] = w0
    red_coords[:, 0] = y0.squeeze()
    qbar[:, 0] = rnm(y0, gp, gpScalings, xTrain)
    wp = w0
    yp = y0

    def decode(x, gp, gpScalings, xTrain):
       return basis @ x + basis2 @ rnm(x, gp, gpScalings, xTrain)

    def jacob(x, gp, gpScalings, xTrain):
       return basis + basis2 @ rnm_grad(x, gp, gpScalings, xTrain)

    t0 = time.time()

    print("Running M-ROM of size {} for mu1={}, mu2={}".format(nred, mu[0], mu[1]))
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp.squeeze(), mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_rnm(res, jac, yp, decode, jacob, gp, gpScalings, xTrain)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        w = basis @ y + basis2 @ rnm(y, gp, gpScalings, xTrain)

        red_coords[:, i + 1] = y.squeeze()
        snaps[:, i + 1] = w.squeeze()
        qbar[:, i +1] = rnm(y, gp, gpScalings, xTrain)
        wp = w
        yp = y

    return snaps, (num_its, jac_time, res_time, ls_time), qbar

def inviscid_burgers_rnm2D_ecsw(grid_x, grid_y, w0, dt, num_steps, mu, rnm, rnm_grad, ref, basis, basis2, weights):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG manifold PROM for a parameterized inviscid 1D burgers
    problem with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    # stuff for operators
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]

    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size / 2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()

    sample_weights = np.concatenate((weights, weights))[sample_inds]

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    kernel_list, gpScalings, xTrain, array_list = GP_load()
    y0 = basis.T @ w0.squeeze()
    w0 = basis @ y0 + basis2 @ rnm(y0, kernel_list, gpScalings, xTrain, array_list)
    nred = y0.shape[0]
    snaps = np.zeros((w0.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0
    red_coords[:, 0] = y0 #
    wp = w0
    yp = y0

    idx = np.concatenate((augmented_sample, int(w0.shape[0] / 2) + augmented_sample))
    wp = w0[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]
    def decode(x,kernel_list, gpScalings, xTrain, array_list):
        return V @ x + Vbar @ rnm(x, kernel_list, gpScalings, xTrain, array_list)

    def jacob(x,kernel_list, gpScalings, xTrain, array_list):
        return V + Vbar @ rnm_grad(x,kernel_list, gpScalings, xTrain, array_list)

    t0 = time.time()

    print("Running M-ROM of size {} for mu1={}, mu2={}".format(nred, mu[0], mu[1]))
    lbc = None
    src = None
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]
    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    wall_clock_time = 0.0
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample, lbc, src)

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_sample)

        print(" ... Working on timestep {}".format(i))
        t0 = time.time()
        y, resnorms, times = gauss_newton_rnm_ecsw(res, jac, yp, decode, jacob, kernel_list, sample_inds, augmented_sample, sample_weights)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        w = V @ y + Vbar @ rnm(y, kernel_list, gpScalings, xTrain, array_list)

        red_coords[:, i + 1] = y.squeeze()
        wp = w
        yp = y
        wall_clock_time += (time.time() - t0)

    return red_coords, (num_its, jac_time, res_time, ls_time)

def gauss_newton_rnm(func, jac, y0, decode, jacfwdfunc, gp, gpScalings, xTrain, max_its=20, relnorm_cutoff=1e-5,
                     lookback=10,
                     min_delta=0.1):
    jac_time = 0
    res_time = 0
    ls_time = 0

    y = np.copy(y0)
    w = decode(y, gp, gpScalings, xTrain)
    init_norm = np.linalg.norm(func(w.squeeze()))
    resnorms = []

    for i in range(max_its):
        resnorm = np.linalg.norm(func(w.squeeze()))
        resnorms += [resnorm]
        if resnorm / init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        t0 = time.time()
        #J = jac(w.squeeze().numpy())
        #V = jacfwdfunc(y,gpm).detach()
        J = jac(w.squeeze())
        V = jacfwdfunc(y, gp, gpScalings, xTrain)
    
        jac_time += time.time() - t0
        t0 = time.time()
        f = func(w.squeeze())
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(V)
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)
        ls_time += time.time() - t0
        y += dy
        w = decode(y, gp, gpScalings, xTrain)
    print('{} iterations: {:3.2e} relative norm'.format(i, resnorm / init_norm))
    return y, resnorms, (jac_time, res_time, ls_time)

def make_ddx(grid_x):
    dx = grid_x[1:] - grid_x[:-1]
    return sp.spdiags([-np.ones(grid_x.size - 1) / dx, np.ones(grid_x.size - 1) / dx], [-1, 0],
                      grid_x.size - 1, grid_x.size - 1, 'lil')

def make_mid(grid_x):
    return sp.spdiags([np.ones(grid_x.size - 1)/2, np.ones(grid_x.size - 1)/2], [-1, 0],
                      grid_x.size - 1, grid_x.size - 1, 'lil')

def make_2D_grid(x_low, x_up, y_low, y_up, num_cells_x, num_cells_y):
    """
    Returns two 1d ndarray of cell boundary points between a lower bound and an upper bound
    with the given number of cells in each direction
    """
    grid_x = np.linspace(x_low, x_up, num_cells_x+1)
    grid_y = np.linspace(y_low, y_up, num_cells_y+1)
    return grid_x, grid_y

def get_ops(grid_x, grid_y):
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()
    idx = np.arange((grid_y.size - 1) * (grid_x.size - 1)).reshape(
        (grid_y.size - 1, grid_x.size - 1)).T.ravel()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    Eye = sp.identity(2 * (grid_x.size - 1) * (grid_y.size - 1))
    return Dxec, Dyec, JDxec, JDyec, Eye

def inviscid_burgers_explicit2D(grid_x, grid_y, u0, v0, dt, num_steps, mu):
    """
    """

    snaps = np.zeros((u0.flatten().size + v0.flatten().size, num_steps+1))
    snaps[:, 0] = np.concatenate((u0.flatten(), v0.flatten()))
    up = u0.copy()
    vp = v0.copy()
    dx = grid_x[1:] - grid_x[:-1]
    xc = (grid_x[1:] + grid_x[:-1])/2

    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    b = np.zeros_like(up)
    b[:, 0] = 0.5 * mu[0]**2
    by = np.zeros_like(vp)
    by[0, :] = 0.5 * mu[0]**2
    f = np.zeros(grid_x.size)
    f[0] = 0.5 * mu[0]**2
    for i in range(num_steps):
        Fux = (0.5 * np.square(up)).T
        Fvy = 0.5 * np.square(vp)
        Fuv = 0.5 * up*vp
        FuvT = Fuv.T
        u = up - dt * ((Dxec@Fux).T - b/dx) + dt*0.02*np.exp(mu[1]*xc[None, :]) \
            - dt * Dyec @ Fuv
        v = vp - dt * Dyec@Fvy\
            - dt * (Dxec@FuvT).T
        if i % 10 == 0:
            print('... Working on timesetp {}'.format(i))
        if i % 200 == 0:
            plt.imshow(v)
            plt.colorbar()
            plt.title('i = {}'.format(i))
            plt.show()
            time.sleep(0.2)
        if i in range(499, 5001, 500):
            snaps[:, i + 1] = np.concatenate((u.ravel(), v.ravel()))
        up = u
        vp = v
    return snaps

def inviscid_burgers_implicit2D(grid_x, grid_y, w0, dt, num_steps, mu):
    """
    """

    print("Running HDM for mu1={}".format(mu[0]))
    snaps = np.zeros((w0.size, num_steps+1))
    snaps[:, 0] = w0.ravel().copy()
    wp = w0.ravel()
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()
    idx = np.arange((grid_y.size - 1)*(grid_x.size - 1)).reshape(
        (grid_y.size - 1, grid_x.size - 1)).T.ravel()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    Eye = sp.eye(2*(grid_x.size - 1)*(grid_y.size - 1))

    for i in range(num_steps):

        def res(w):
            # res1 = inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)
            res2 = inviscid_burgers_res2D_alt(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec)
            # print('Diff in norm: {}'.format(np.linalg.norm(res1 - res2)))
            return res2

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        print(" ... Working on timestep {}".format(i))
        w, resnorms = newton_raphson(res, jac, wp, 100, 1e-12)

        # if i % 10 == 0:
        #     plt.imshow(w.reshape(grid_y.size - 1, grid_x.size - 1))
        #     plt.colorbar()
        #     plt.title('i = {}'.format(i))
        #     plt.show()
        #     time.sleep(0.05)

        snaps[:, i+1] = w.ravel()
        wp = w.copy()

    return snaps

def inviscid_burgers_implicit2D_LSPG(grid_x, grid_y, w0, dt, num_steps, mu, basis):
    """
    """

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    npod = basis.shape[1]
    snaps =  np.zeros((w0.size, num_steps+1))
    red_coords = np.zeros((npod, num_steps+1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:, 0] = w0
    red_coords[:, 0] = y0
    wp = w0.copy()
    yp = y0.copy()
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()
    shp = (grid_y.size - 1, grid_y.size - 1)
    size = shp[0]*shp[1]
    idx = np.arange(size).reshape(shp).T.flatten()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    # JDxec_JDyec = (JDxec + JDyec).tocoo()
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))
    # Jop = sp.bmat([[JDxec, 0.5*JDyec], [0.5*JDxec, JDyec]])
    Jop = sp.bmat([[JDxec, None], [None, JDyec]])
    # Eye = sp.eye((grid_x.size - 1)*(grid_y.size - 1))
    print("Running ROM of size {} for mu1={}, mu2={}".format(npod, mu[0], mu[1]))
    for i in range(num_steps):

        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_LSPG(res, jac, basis, yp)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep
        
        w = basis.dot(y)

        red_coords[:, i+1] = y.copy()
        snaps[:, i+1] = w.copy()
        wp = w.copy()
        yp = y.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)

def generate_augmented_mesh(grid_x, grid_y, sample_inds):
    augmented_sample = set(sample_inds)
    shp = (grid_y.size - 1, grid_x.size - 1)
    for i in sample_inds:
        r, c = np.unravel_index(i, shp)
        if r - 1 >= 0:
            # print('adding ({}, {}): {}'.format(r - 1, c, np.ravel_multi_index((r - 1, c), shp)))
            augmented_sample.add(np.ravel_multi_index((r - 1, c), shp))
        # else:
        #     print('({}, {}): out of bounds!'.format(r - 1, c))
        if c - 1 >= 0:
            # print('adding ({}, {}): {}'.format(r, c - 1, np.ravel_multi_index((r, c - 1), shp)))
            augmented_sample.add(np.ravel_multi_index((r, c - 1), shp))
        # else:
        #     print('({}, {}): out of bounds!'.format(r, c - 1))
        if c - 1 >= 0:
            idx = np.ravel_multi_index((r, c - 1), shp)
            if idx not in augmented_sample:
                print('({}, {}): missing a point!'.format(r, c - 1))
    augmented_sample = np.sort(np.array(list(augmented_sample)))
    return augmented_sample

def inviscid_burgers_ecsw_fixed(grid_x, grid_y, weights, w0, dt, num_steps, mu, basis):
    """
    """

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    npod = basis.shape[1]
    snaps = np.zeros((w0.size, num_steps + 1))
    red_coords = np.zeros((npod, num_steps + 1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:, 0] = w0
    red_coords[:, 0] = y0
    wp = w0.copy()
    yp = y0.copy()

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]

    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size/2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()
    print("Running ROM of size {} for mu1={}, mu2={}".format(npod, mu[0], mu[1]))

    weights2 = np.hstack((weights, weights))
    sample_weights = weights2[sample_inds]

    idx = np.concatenate((augmented_sample, int(w0.size/2) + augmented_sample))
    wp = w0[idx]

    basis_red = basis[idx, :]
    for i in range(num_steps):

        def res(w):
            return inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample)

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_sample)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_ECSW_2D(res, jac, basis_red, yp, sample_inds, augmented_sample, sample_weights)
        print('number iter: {}'.format(len(resnorms)))
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        w = basis_red.dot(y)
        # u, v = np.split(w, 2)
        # plt.imshow(u.reshape(250, 250))
        # plt.colorbar()
        # plt.show()

        red_coords[:, i + 1] = y.copy()
        wp = w.copy()
        yp = y.copy()

    return red_coords, (jac_time, res_time, ls_time)

def inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec):
    """
    """

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1])/2
    u_idx = dx.size*dy.size
    up = wp[:u_idx].reshape(dy.size, dx.size)
    u = w[:u_idx].reshape(dy.size, dx.size)
    vp = wp[u_idx:].reshape(dy.size, dx.size)
    v = w[u_idx:].reshape(dy.size, dx.size)

    Fux, Fpux = (0.5 * np.square(u)).T, (0.5 * np.square(up)).T
    Fvy, Fpvy = 0.5 * np.square(v), 0.5 * np.square(vp)
    Fuv, Fpuv = 0.5 * u * v, 0.5 * up * vp
    FuvT, FpuvT = Fuv.T, Fpuv.T
    src = dt * 0.02 * np.exp(mu[1] * xc[None, :])
    ru = u - up + 0.5 * dt * (Dxec@(Fux + Fpux)).T + \
         0.5 * dt * Dyec @ (Fuv + Fpuv) - src
    ru[:, 0] -= 0.5 * dt * mu[0] ** 2 / dx
    rv = v - vp + 0.5 * dt * Dyec @ (Fvy + Fpvy) + \
         0.5 * dt * (Dxec @ (FuvT + FpuvT)).T

    return np.concatenate((ru.ravel(), rv.ravel()))

def inviscid_burgers_res2D_alt(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec):
    """
    """

    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1])/2
    u, v = np.split(w, 2)
    up, vp = np.split(wp, 2)

    Fux, Fpux = (0.5 * np.square(u)), (0.5 * np.square(up))
    Fvy, Fpvy = 0.5 * np.square(v), 0.5 * np.square(vp)
    Fuv, Fpuv = 0.5 * u * v, 0.5 * up * vp
    FuvT, FpuvT = Fuv, Fpuv
    src = dt * 0.02 * np.exp(mu[1] * xc)
    lbc = np.zeros_like(u).reshape((dy.size, dx.size))
    lbc[:, 0] = 0.5 * dt * mu[0] ** 2 / dx
    lbc = lbc.ravel()
    src = np.tile(src, dy.size)
    ru = u - up + 0.5 * dt * JDxec@(Fux + Fpux) + \
        0.5 * dt * JDyec @ (Fuv + Fpuv) - src
    ru -= lbc
    rv = v - vp + 0.5 * dt * JDyec @ (Fvy + Fpvy) + \
         0.5 * dt * (JDxec @ (FuvT + FpuvT))
    return np.concatenate((ru, rv))

def inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample, lbc=None, src=None):
    """
    Assumes either the full state w or w[augmented_sample] as the input...
    """
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]
    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    u, v = np.split(w, 2)
    up, vp = np.split(wp, 2)
    if u.size > augmented_sample.size:
        Fux, Fpux = 0.5 * np.square(u[augmented_sample]), 0.5 * np.square(up[augmented_sample])
        Fvy, Fpvy = 0.5 * np.square(v[augmented_sample]), 0.5 * np.square(vp[augmented_sample])
        Fuv = 0.5 * u[augmented_sample] * v[augmented_sample]
        Fpuv = 0.5 * up[augmented_sample] * vp[augmented_sample]

        u = u[sample_inds]
        v = v[sample_inds]
        up = up[sample_inds]
        vp = vp[sample_inds]
    else:
        Fux, Fpux = 0.5 * np.square(u), 0.5 * np.square(up)
        Fvy, Fpvy = 0.5 * np.square(v), 0.5 * np.square(vp)
        Fuv = 0.5 * u * v
        Fpuv = 0.5 * up * vp

        overlap = np.isin(augmented_sample, sample_inds)
        u = u[overlap]
        v = v[overlap]
        up = up[overlap]
        vp = vp[overlap]

    ru = u - up + 0.5 * dt * JDxec @ (Fux + Fpux) + \
         0.5 * dt * JDyec @ (Fuv + Fpuv) - src
    ru -= lbc
    rv = v - vp + 0.5 * dt * JDyec @ (Fvy + Fpvy) + \
         0.5 * dt * (JDxec @ (Fuv + Fpuv))

    return np.concatenate((ru, rv))

def inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye):
    u, v = np.split(w, 2)
    ud, vd = 0.5*dt*sp.diags(u), 0.5*dt*sp.diags(v)
    ul = JDxec@ud + 0.5*JDyec@vd
    ur = 0.5*JDyec@ud
    ll = 0.5*JDxec@vd
    lr = JDyec@vd + 0.5*JDxec@ud
    return sp.bmat([[ul, ur], [ll, lr]]) + Eye

def inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_inds):
    u, v = np.split(w, 2)
    if u.size > augmented_inds.size:
        ud, vd = 0.5 * dt * sp.diags(u[augmented_inds]), 0.5 * dt * sp.diags(v[augmented_inds])
    else:
        ud, vd = 0.5 * dt * sp.diags(u), 0.5 * dt * sp.diags(v)
    ul = (JDxec@ud + 0.5*JDyec@vd)
    ur = (0.5*JDyec@ud)
    ll = (0.5*JDxec@vd)
    lr = (JDyec@vd + 0.5*JDxec@ud)
    return sp.bmat([[ul, ur], [ll, lr]]) + Eye


def newton_raphson(func, jac, x0, max_its=20, relnorm_cutoff=1e-12):
    x = x0.copy()
    init_norm = np.linalg.norm(func(x0))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(x))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            print('{}: {:3.2e}'.format(i, resnorm/init_norm))
            break
        J = jac(x)
        f = func(x)
        x -= (sp.linalg.spsolve(J, f))
    return x, resnorms


def gauss_newton_LSPG(func, jac, basis, y0, 
                      max_its=20, relnorm_cutoff=1e-5, min_delta=0.1):
    jac_time = 0
    res_time = 0
    ls_time = 0
    y = y0.copy()
    w = basis.dot(y0)
    init_norm = np.linalg.norm(func(w))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w))
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0
        t0 = time.time()
        f = func(w)
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(basis)
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)
        ls_time += time.time() - t0
        y += dy
        w = basis.dot(y)
    print('iteration {}: relative norm {:3.2e}'.format(i, resnorm/init_norm))
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_rnm_ecsw(func, jac, y0, decode, jacfwdfunc, kernel_list, gpScalings,
                          sample_inds, augmented_sample, weight,
                          max_its=20, relnorm_cutoff=1e-5,
                          min_delta=0.1):
    jac_time = 0
    res_time = 0
    ls_time = 0

    y = np.copy(y0)
    w = decode(y, kernel_list, gpScalings)
    weights = np.concatenate((weight, weight))
    init_norm = np.linalg.norm(func(w) * weights)
    resnorm = init_norm
    i = 0
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w) * weights)
        resnorms += [resnorm]
        if resnorm / init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        t0 = time.time()

        J = jac(w.squeeze)
        V = jacfwdfunc(y,kernel_list, gpScalings)

        jac_time += time.time() - t0
        t0 = time.time()
        f = func(w.squeeze())
        fw = f * weights
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(V)
        dw = sp.spdiags(weights, 0, weights.size, weights.size)
        JVw = dw @ JV
        dy, lst_res, rank, sval = np.linalg.lstsq(JVw, -fw, rcond=None)
        ls_time += time.time() - t0
        y += dy
        w = decode(y,kernel_list, gpScalings)
    print('{} iterations: {:3.2e} relative norm'.format(i, resnorm / init_norm))
    return y, resnorms, (jac_time, res_time, ls_time)


def gauss_newton_ECSW_2D(func, jac, basis, y0, sample_inds, augmented_sample, weight,
                      stepsize=1, max_its=20, relnorm_cutoff=1e-5, min_delta=1E-1):
    y = y0.copy()
    w = basis.dot(y0)

    weights = np.concatenate((weight, weight))
    init_norm = np.linalg.norm(func(w) * weights)
    resnorms = []
    jac_time = 0
    res_time = 0
    ls_time = 0

    for i in range(max_its):
        resnorm = np.linalg.norm(func(w) * weights)
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0
        JV = J @ basis
        dw = sp.spdiags(weights, 0, weights.size, weights.size)
        JVw = dw@JV
        t0 = time.time()
        f = func(w)
        res_time += time.time() - t0
        t0 = time.time()
        fw = f * weights
        dy = np.linalg.lstsq(JVw, -fw, rcond=None)[0]
        ls_time += time.time() - t0
        y = y + stepsize*dy

        w = basis.dot(y)

    return y, resnorms, (jac_time, res_time, ls_time)

def POD(snaps):
    u, s, vh = np.linalg.svd(snaps, full_matrices=False)
    return u, s

def podsize(svals, energy_thresh=None, min_size=None, max_size=None):
    """ Returns the number of vectors in a basis that meets the given criteria """

    if (energy_thresh is None) and (min_size is None) and (max_size is None):
        raise RuntimeError('Must specify at least one truncation criteria in podsize()')

    if energy_thresh is not None:
        svals_squared = np.square(svals.copy())
        energies = np.cumsum(svals_squared)
        energies /= np.square(svals).sum()
        numvecs = np.where(energies >= energy_thresh)[0][0]
    else:
        numvecs = min_size

    if min_size is not None and numvecs < min_size:
        numvecs = min_size

    if max_size is not None and numvecs > max_size:
        numvecs = max_size

    return numvecs

def compute_ECSW_training_matrix_2D(snaps, prev_snaps, basis, res, jac, grid_x, grid_y, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """
    n_hdm, n_snaps = snaps.shape
    n_hdm = int(n_hdm / 2)
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        ires = res(snap, grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        Ji = jac(snap, dt, JDxec, JDyec, Eye)
        Wi = Ji.dot(basis)
        for inode in range(n_hdm):
            C[isnap*n_pod:isnap*n_pod+n_pod, inode] = ires[inode]*Wi[inode] + ires[inode+n_hdm]*Wi[inode+n_hdm]

    return C

def compute_ECSW_training_matrix_2D_rnm(snaps, prev_snaps, basis, approx, jacfwdfunc, res, jac, grid_x, grid_y, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """
    n_hdm, n_snaps = snaps.shape
    n_hdm = int(n_hdm / 2)
    n_pod = basis.shape[1]
    print(n_pod)
    print(n_snaps)
    print(n_hdm)
    C = np.zeros((n_pod * n_snaps, n_hdm))
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    for isnap in range(n_snaps):#range(1, n_snaps):
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        y0 = basis.T @ snap
        init_res = np.linalg.norm(approx(y0).squeeze() - snap)
        approx_res = init_res
        num_it = 0
        y = y0
        print('Initial residual: {:3.2e}'.format(init_res / norm(snap)))
        while abs(approx_res / init_res) > 1e-2 and num_it < 10:
            Jf = jacfwdfunc(y)
            JJ = Jf.T @ Jf
            Jr = Jf.T @ (approx(y) - snap)
            dy, _, _, _ = np.linalg.lstsq(JJ.squeeze(), Jr.squeeze(), rcond=None)
            y -= dy
            approx_res = np.linalg.norm(approx(y).squeeze() - snap)
            # print('it: {}, Relative residual of fit: {:3.2e}'.format(num_it, abs(approx_res / init_res)))
            num_it += 1
        final_res = np.linalg.norm(approx(y).squeeze() - snap)
        print('Final residual: {:3.2e}'.format(final_res / norm(snap)))
        ires = res(approx(y).squeeze(), grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        J = jac(approx(y).squeeze(), dt, JDxec, JDyec, Eye)
        V = (jacfwdfunc(y)).squeeze()
        Wi = J.dot(V)
        for inode in range(n_hdm):
            C[(isnap)*n_pod:(isnap)*n_pod+n_pod, inode] = ires[inode]*Wi[inode] + ires[inode+n_hdm]*Wi[inode+n_hdm]

    return C

def compute_error(rom_snaps, hdm_snaps):
    """ Computes the relative error at each timestep """
    sq_hdm = np.sqrt(np.square(rom_snaps).sum(axis=0))
    sq_err = np.sqrt(np.square(rom_snaps - hdm_snaps).sum(axis=0))
    rel_err = sq_err / sq_hdm
    return rel_err, rel_err.mean()

def param_to_snap_fn(mu, snap_folder="param_snaps", suffix='.npy'):
    npar = len(mu)
    snapfn = snap_folder + '/'
    for i in range(npar):
        if i > 0:
            snapfn += '+'
        param_str = 'mu{}_{}'.format(i+1, mu[i])
        snapfn += param_str
    return snapfn + suffix

def get_saved_params(snap_folder="param_snaps"):
    param_fn_set = set(glob.glob(snap_folder+'/*'))
    return param_fn_set

def load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder="param_snaps"):
    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    saved_params = get_saved_params(snap_folder=snap_folder)
    if snap_fn in saved_params:
        print("Loading saved snaps for mu1={}, mu2={}".format(mu[0], mu[1]))
        snaps = np.load(snap_fn)[:, :num_steps+1]
    else:
        snaps = inviscid_burgers_implicit2D(grid_x, grid_y, w0, dt, num_steps, mu)
        np.save(snap_fn, snaps)
    return snaps

def plot_snaps(grid_x, grid_y, snaps, snaps_to_plot, linewidth=2, color='black', linestyle='solid',
               label=None, fig_ax=None):

    if (fig_ax is None):
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        fig, ax1, ax2 = fig_ax


    x = (grid_x[1:] + grid_x[:-1])/2
    y = (grid_y[1:] + grid_y[:-1])/2
    mid_x = int(x.size / 2)
    mid_y = int(y.size / 2)
    is_first_line = True
    for ind in snaps_to_plot:
        if is_first_line:
            label2 = label
            is_first_line = False
        else:
            label2 = None
        snap = snaps[:(y.size*x.size), ind].reshape(y.size, x.size)
        ax1.plot(x, snap[mid_y, :],
                color=color, linestyle=linestyle, linewidth=linewidth, label=label2)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u_x(x,y={:0.1f})$'.format(y[mid_y]))
        #ax1.set_title('$x$-axis $y$-midpoint slice')
        ax1.grid()
        ax2.plot(y, snap[:, mid_x],
                 color=color, linestyle=linestyle, linewidth=linewidth, label=label2)
        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$u_x(x={:0.1f},y)$'.format(x[mid_x]))
        #ax2.set_title('$y$-axis $x$-midpoint slice')
        ax2.grid()
    return fig, ax1, ax2
