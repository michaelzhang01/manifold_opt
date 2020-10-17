# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 02:16:19 2020

Extrinsic Sphere Code

@author: Michael Zhang
"""

import numpy as np
from manifold_opt import ManifoldOpt

class ExtrinsicSphereOpt(ManifoldOpt):
    
    def __init__(self, X, theta_init, lambda_init, iters=500,
                 filename="extrinsic_sphere_opt2.txt"):
        super().__init__(X, theta_init, lambda_init, iters, filename)
        self.extrinsic_mean = self.X.sum(axis=0) /  np.linalg.norm(self.X.sum(axis=0))

    def local_loss_grad(self,theta):
        V_N = self.X.sum(axis=0)
        grad = (2. / float(self.N))*( (np.dot(V_N, theta)*theta) - V_N )
        return(grad)
        
    def local_loss(self, theta):
        return((np.linalg.norm(self.X-theta)**2)/self.N)
        
    def regularize(self,kappa, theta_new, theta_old):
        theta_dot = np.clip(np.dot(theta_new,theta_old),-1+1e-7,1-1e-7)
        arccos_theta = np.arccos(theta_dot)
        sqrt_term = np.sqrt(1.-(theta_dot**2))
        K_1 = arccos_theta / sqrt_term
        logmap = K_1* (theta_new - (theta_old.dot(theta_dot)))
        return((.5*kappa)*(np.linalg.norm(logmap)**2))
        
    def regularize_grad(self,kappa, theta_new, theta_old):
        theta_dot = np.clip(np.dot(theta_new,theta_old),-1+1e-7,1-1e-7)
        arccos_theta = np.arccos(theta_dot)
        sqrt_term = np.sqrt(1.-(theta_dot**2))
        K_1 = arccos_theta / sqrt_term
        logmap = K_1* (theta_new - (theta_old*theta_dot))
        K_2 = (arccos_theta*theta_dot) - sqrt_term
        K_2 /= sqrt_term**3
        return(kappa*np.linalg.norm(logmap)*K_2)

    def exp_map(self, theta, v):
        norm_v = np.linalg.norm(v)
        if np.allclose(norm_v,0):
            return(theta)
        else:
            return_map = np.cos(norm_v)*theta
            return_map += np.sin(norm_v)*(v/norm_v)
            return_map /= np.linalg.norm(return_map)
            return(return_map)

    def log_map(self,theta,theta_bar):
        theta_dot = np.clip(theta.dot(theta_bar),-1+1e-7,1-1e-7)
        return_map = np.arccos(theta_dot)
        return_map /= np.sqrt(1.- theta_dot**2)
        return_map *= theta - theta_bar * theta_dot
        return(return_map)
    
    def print_fun(self, it, iter_time, f):
        theta_mse = np.mean((self.theta[it] - self.extrinsic_mean)**2)
        print("%i\t%f\t%f\t%f\t%f" % (it,self.theta_norm,self.theta_loss,theta_mse,iter_time), file=f)

        
if __name__ == '__main__':
    base_seed = 8888
    np.random.seed(base_seed)
    trials = 1
    D = 100
    N = int(1e4)
    X=np.random.normal( np.random.normal(size=D), scale = [2]*D,size=(N,D))
    X_norm = np.linalg.norm(X,axis=1)
    X= np.vstack([X[i]/x_n for i,x_n in enumerate(X_norm)])

    for t in range(trials):
        np.random.seed(base_seed + t)
        theta = np.ones(D) #np.random.vonmises([0]*self.D,[1]*self.D)  +
        theta /= np.linalg.norm(theta)
        V_N = X.sum(axis=0)
        theta_grad = (2. / float(N))*( (np.dot(V_N, theta)*theta) - V_N )
        theta_norm = np.linalg.norm(theta_grad)
        lambda0 = (-np.pi / theta_norm)
        sph = ExtrinsicSphereOpt(X,theta_init=theta, lambda_init=lambda0)
        sph.catalyst_opt()
        sph = ExtrinsicSphereOpt(X,theta_init=theta, lambda_init=lambda0)
        sph.opt()
        sph = ExtrinsicSphereOpt(X,theta_init=theta, lambda_init=lambda0)
        sph.RAGD_opt()
        sph = ExtrinsicSphereOpt(X,theta_init=theta, lambda_init=lambda0)
        sph.DANE_opt()
