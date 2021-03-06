# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 23:14:43 2018

Parallel Extrinsic Mean Gradient Descent for Spheres

@author: Michael Zhang
"""

#import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import os
from mpi4py import MPI
#from scipy.io import loadmat
#from scipy.io import savemat
#from scipy import stats
from scipy.optimize import minimize
#import pdb
import time

#np.random.seed(8888)

def pi_logistic_map(x, grad_length):
    log_map = ((2.*np.pi/grad_length) / (1.+  np.exp(-x)))  - (np.pi/grad_length)
    return(log_map)

def exp_map(theta, v):
    norm_v = np.linalg.norm(v)
    if np.allclose(norm_v,0):
        return(theta)
    else:
        return_map = np.cos(norm_v)*theta
        return_map += np.sin(norm_v)*(v/norm_v)
        return(return_map)

class SphereOptimization(object):
    def __init__(self, X, iters=1000, verbose=1):
        self.iters = int(iters)
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.verbose = int(verbose)

        if self.rank == 0:
            self.X = X
            self.V_N = self.X.sum(axis=0)
            self.N,self.D = self.X.shape
            self.X = np.array_split(self.X, self.P)
            self.extrinsic_mean = self.V_N /  np.linalg.norm(self.V_N)
        else:
            self.X = None
            self.N = None
            self.D = None
            self.V_N = None
            self.extrinsic_mean = None
            self.theta = None

        self.N = self.comm.bcast(self.N)
        self.D = self.comm.bcast(self.D)
        self.V_N = self.comm.bcast(self.V_N)
        self.X_local = self.comm.scatter(self.X)
        self.X = None
        self.N_p, _ = self.X_local.shape
        self.extrinsic_mean = self.comm.bcast(self.extrinsic_mean)
        self.V_p = self.X_local.sum(axis=0)
        if self.rank ==0:
            self.theta = np.empty((self.iters,self.D))
#            self.theta[0] = minimize(self.local_loss, x0= self.X_local.mean(axis=0),jac=self.local_loss_grad).x
            self.theta[0] = np.random.uniform(low=-1,size=self.D)
            self.theta[0] /= np.linalg.norm(self.theta[0])
        else:
            self.theta= None
        self.theta = self.comm.bcast(self.theta)

    def local_loss(self, theta):
        return(np.linalg.norm(self.V_p-theta)**2)

    def local_loss_grad(self,theta):
        grad = (2. / float(self.N_p))*( (np.dot(self.V_p, theta)*theta) - self.V_p )
        return(grad)

    def global_loss_grad(self,theta):
        grad = (2. / float(self.N))*( (np.dot(self.V_N, theta)*theta) - self.V_N)
        return(grad)


    def surrogate_grad_zero(self,theta):
        grad = (2. / float(self.N_p))* ((np.dot((self.V_N/float(self.P)),theta)*theta) -  (self.V_N/float(self.P)))
        return(grad)

    def surrogate_grad(self,theta_new, theta_old):
        V_subtract = ((self.V_N/float(self.P)) - self.V_p)
        V_theta_s = np.dot(V_subtract, theta_old)*theta_old
        theta_dot = np.dot(theta_new,theta_old)
        sqrt_term = np.sqrt(1.-(theta_dot**2))
        V_sub_term = V_theta_s - V_subtract

        grad = self.local_loss_grad(theta_new)

        K_1 = (2.*np.arccos(theta_dot)) / (self.N_p*sqrt_term )
        K_2 =(2.*np.arccos(theta_dot)*theta_dot) - (2.*sqrt_term)
        K_2 /= self.N_p*(sqrt_term**3)

        grad +=  K_1* (V_sub_term - np.dot(V_sub_term,theta_new)*theta_new)
        grad += K_2 * (np.dot(V_sub_term,theta_new)*theta_new) * (theta_old - (theta_dot*theta_new))
        assert(np.all(~np.isnan(grad)))
        return(grad)

    def lambda_obj(self, ell, it):
        if it==1:
            theta_grad = self.surrogate_grad_zero(self.theta[it-1])
        elif it > 1:
            theta_grad = self.surrogate_grad(self.theta[it-1],self.theta[it-2])

        ell = pi_logistic_map(ell,np.linalg.norm(theta_grad))

        exp_map_theta = exp_map(self.theta[it-1],ell*theta_grad)
        loss = self.local_loss(exp_map_theta) + (ell*np.dot(theta_grad, (self.global_loss_grad(self.theta[it-1]) - self.local_loss_grad(self.theta[it-1]))))
        return(loss)


    def extrinsic_opt(self):
        start_time = time.time()

        # print initial value
        if self.rank ==0:
            if self.verbose >=2:
                print("%i\t%i\t%.2f\t%s"%(0, 0, 0, self.theta[0]))

        for it in xrange(1,self.iters):
            current_proc = it % int(self.P) # select processor
#            current_proc = 0
            if self.rank == current_proc:
                if it == 1:
                    theta_grad = self.surrogate_grad_zero(self.theta[it-1])
                else:
                    theta_grad = self.surrogate_grad(self.theta[it-1],self.theta[it-2])
                min_lambda =  pi_logistic_map(minimize(self.lambda_obj,x0=-.5,args=(it,)).x,np.linalg.norm(theta_grad))
                self.theta[it] = exp_map(self.theta[it-1], min_lambda*theta_grad)
                self.theta[it] /= np.linalg.norm(self.theta[it]) # ensure norm of current theta is 1
                if self.verbose >=2:
                    print("%i\t%i\t%.2f\t%s"%(it, current_proc, min_lambda, self.theta[it]))
            self.comm.barrier()
            self.theta = self.comm.bcast(self.theta, current_proc)

            # stopping criteria
            if np.allclose(np.dot(self.theta[it],self.theta[it-1])**2,1.):
                self.theta = self.theta[:it]
                self.comm.barrier()
                self.theta = self.comm.bcast(self.theta, current_proc)
                break

        if self.verbose >= 1:
            self.final_theta = self.theta[-1]
            self.RMSE = np.sqrt(np.mean((self.final_theta-self.extrinsic_mean)**2))
            end_time = time.time() - start_time
            X_gather = self.comm.gather(self.X_local)
            if self.rank==0:
                X_gather = np.vstack(X_gather)
                print("%i\t%.2f\t%s" % (self.P, end_time, self.RMSE))
                f,a = plt.subplots(1,1)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    base_seed =88888
    np.random.seed(base_seed)
    trials = 20
    D = 100
    if comm.Get_rank() == 0:
#        X=np.random.vonmises( np.random.normal(size=D), kappa = [2]*D,size=(int(1e4),D))
        X=np.random.normal( np.random.normal(size=D), scale = [2]*D,size=(int(1e6),D))
        X_norm = np.linalg.norm(X,axis=1)
        X= np.vstack([X[i]/x_n for i,x_n in enumerate(X_norm)])
    else:
        X = None
    for t in xrange(trials):
        np.random.seed(base_seed + t)
        sph = SphereOptimization(X,verbose=1)
        sph.extrinsic_opt()
#    sph = SphereOptimization(X)
#    sph.extrinsic_opt()

