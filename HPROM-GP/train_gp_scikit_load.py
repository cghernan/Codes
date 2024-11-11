import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import pickle
import GPy

tot = time.time()
nc = 10
nt = 150
# b = np.load('basis.npy')
q = np.load('coords.npy')
q0 = q[0:nc, :]
q1 = q[nc:nt, :]
# mu = np.array(range(9))/8.0
# u = np.ones(9)
# q0 = np.vstack((np.hstack(np.array(range(501))/500.0 * u[np.newaxis,:].T),
#                np.hstack((np.ones(501)*mu[np.newaxis,:].T)))
#              )
# q1 = q[0:nt,:]
print('Reduced coordinates read time: ', time.time() - tot)

ns = q1.shape[1]
nTrain = 1000
'''
ixTrain = np.random.choice(range(0, ns, 3), size=nTrain, replace=False)
print('training set size: ', ixTrain.shape)
# ixTest = list(set(range(ns)) - set(ixTrain))
ixTest = list(set(range(ns)))
ixTestE = list(set(range(ns)) - set(ixTrain))
fi = open('ixTrain_sept18.npy', 'wb')
np.save(fi, ixTrain)
fi.close()
'''
ixTrain = np.load('ixTrain_21sept.npy')
ixTest = list(set(range(ns)))
ixTestE = list(set(range(ns)) - set(ixTrain))

with open('gp_21sept.pkl', 'rb') as fb:
     gp = pickle.load(fb)

# Scaling factors
txmin = np.min(q0, axis=1)
txmax = np.max(q0, axis=1)
tymin = np.min(q1[:, :])
tymax = np.max(q1[:, :])

q1Pred = np.zeros_like(q1)

# Scaling factors
txmin = np.min(q0, axis=1)
txmax = np.max(q0, axis=1)
tymin = np.min(q1, axis=1)
tymax = np.max(q1, axis=1)
#tymin = np.min(q1[:, :])
#tymax = np.max(q1[:, :])

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

# Train
kernel0 = ConstantKernel(1.0e0, (1e-3, 1e2)) * \
       Matern(0.5 * np.ones(q0.shape[0]), (1e-2, 5.0), nu=1.5)
# RBF(0.5*np.ones(q0.shape[0]), (1e-2, 5.0))
# kernel0 = ConstantKernel(1.0e0, (1e-3, 1e2)) * \
#          RBF(0.5, (1e-2, 5.0))

#gp = GaussianProcessRegressor(kernel=kernel0, alpha=1e-8, n_restarts_optimizer=1)
#gp.fit(xTrain, yTrain)
print(gp.kernel_.get_params())

# Predict
yPred, yStd = gp.predict(xTest, return_std=True)
print('Total norm: ', np.linalg.norm(yTest - yPred) / np.linalg.norm(yTest))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(np.array(range(xTest.shape[0])), yTest[:,1], 'r-', label='Reference')
plt.plot(np.array(range(xTest.shape[0])), yPred[:,1], 'b.', label='GP fit')
plt.legend(loc='upper left')
plt.savefig('p%03d_sk_vec.pdf')
plt.close()

# save
#with open('gp_sci_test.pkl','wb') as f:
#    pickle.dump(gp,f)

q1Pred[:, ixTest] = (((yPred + 1.0) / 2.0) * (tymax - tymin) + tymin).T
print('Total norm rescaled: ', np.linalg.norm(q1 - q1Pred) / np.linalg.norm(q1))

for num_gp in range(nt-nc):
    print('Total norm rescaled GP no %d : ' % num_gp, np.linalg.norm(q1[num_gp,:] - q1Pred[num_gp,:]) / np.linalg.norm(q1[num_gp,:]))


print('Total GP training time: ', time.time() - tot)
breakpoint()
