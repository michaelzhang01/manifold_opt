# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:44:49 2018

Parallel Intrinsic Mean Gradient Descent for Spheres

@author: Michael Zhang
"""
import time
import numpy as np
from mpi4py import MPI
from scipy.optimize import minimize
import pdb

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
        return_map /= np.linalg.norm(return_map)
        return(return_map)

class SphereOptimization(object):
    def __init__(self, X, iters=1000, verbose = 1):
        self.iters = int(iters)
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.verbose = int(verbose)
        if self.rank == 0:
            self.X = X
            self.N,self.D = self.X.shape
            self.X = np.array_split(self.X, self.P)
        else:
            self.X = None
            self.N = None
            self.D = None
            self.theta = None

        self.N = self.comm.bcast(self.N)
        self.D = self.comm.bcast(self.D)
        self.X_local = self.comm.scatter(self.X)
        self.X = None
        self.N_p, _ = self.X_local.shape
        if self.rank ==0:
            self.theta = np.empty((self.iters,self.D))
            self.theta[0] = np.random.uniform(low=-1,size=self.D)
            self.theta[0] /= np.linalg.norm(self.theta[0])
            local_grad = self.local_loss_grad(self.theta[0])
        else:
            self.theta= None
            self.grad_diff = None
            local_grad = None

        self.theta = self.comm.bcast(self.theta)
        local_grad = self.comm.bcast(local_grad)
        self.grad_diff = self.global_loss_grad(self.theta[0]) - local_grad
#        self.grad_diff = self.comm.bcast(self.grad_diff)

    def local_loss_grad2(self,theta):
        grad = np.zeros(self.D)
        for i in xrange(self.N_p):
            dot_prod = np.dot(theta,self.X_local[i])
            prod_term = self.X_local[i] -  (theta*dot_prod)
            cos_term = np.arccos(dot_prod)
            sqrt_term = np.sqrt(1. - (dot_prod**2))
            grad += (cos_term/sqrt_term)*prod_term
        return((-2./self.N_p)*grad)

    def local_loss_grad(self,theta):
        x_theta_dot = np.dot(self.X_local, theta)
        x_diff = self.X_local - theta*np.tile(x_theta_dot,self.D).reshape(self.D,self.N_p).T
        cos_sqrt_term  = np.arccos(x_theta_dot)
        cos_sqrt_term  /= np.sqrt(1. - x_theta_dot**2)
        grad = -2.*np.dot( x_diff.T, cos_sqrt_term)/ self.N_p
        return(grad)

    def local_loss(self, theta):
        loss = np.arccos(np.dot(self.X_local,theta))**2
        return(loss.sum())

    def global_loss_grad(self,theta):
        grad = self.comm.allreduce(self.local_loss_grad(theta))
        return(grad / self.P)

    def surrogate_grad_zero(self,theta):
        return(self.local_loss_grad(theta) + self.grad_diff)

    def surrogate_grad(self,theta_new, theta_old):
        theta_dot = np.dot(theta_new,theta_old)
        arccos_theta = np.arccos(theta_dot)
        sqrt_term = np.sqrt(1.-(theta_dot**2))
        grad_diff_theta = np.dot(self.grad_diff, theta_new)
        grad = self.local_loss_grad(theta_new)
        K_1 = arccos_theta / sqrt_term
        K_1_grad = self.grad_diff - (grad_diff_theta*theta_new)
        grad += K_1 * K_1_grad
        K_2 = (arccos_theta*theta_dot) - sqrt_term
        K_2 /= sqrt_term**3
        K_2_grad = grad_diff_theta*(theta_old - (theta_dot*theta_new))
        grad += K_2 * K_2_grad
        assert(np.all(~np.isnan(grad)))
        return(grad)

    def surrogate_loss(self,theta_new, theta_old):
        loss = self.local_loss(theta_new)
        theta_dot = np.dot(theta_new,theta_old)
        arccos_theta = np.arccos(theta_dot)
        sqrt_term = np.sqrt(1.-(theta_dot**2))
#        K_1 = arccos_theta / sqrt_term
        loss += (arccos_theta / sqrt_term)*np.dot(theta_new,self.grad_diff)
        return(loss)

    def surrogate_loss_zero(self, theta):
        return(self.local_loss(theta) - np.dot(theta,self.grad_diff))

    def armijo(self, it,  theta_grad, theta_norm, tau=.5, kappa=1e-4):
        lambda0 = -np.pi / theta_norm
        if it==1:
            current_loss = self.surrogate_loss_zero(self.theta[it-1])
        elif it > 1:
            current_loss = self.surrogate_loss(self.theta[it-1],self.theta[it-2])
        exp_map_theta =exp_map(self.theta[it-1],lambda0*theta_grad)
        new_loss = self.surrogate_loss(exp_map_theta,self.theta[it-1])
        path_derivative = np.dot(theta_grad.T, lambda0*theta_grad)
#        pdb.set_trace()
        while (new_loss - current_loss) > (kappa*path_derivative):
            lambda0 *= tau
            exp_map_theta =exp_map(self.theta[it-1],lambda0*theta_grad)
            new_loss = self.surrogate_loss(exp_map_theta,self.theta[it-1])
            path_derivative = np.dot(theta_grad.T, lambda0*theta_grad)
        return(exp_map_theta,lambda0)


    def lambda_obj(self, ell, it):
        if it==1:
            theta_grad = self.surrogate_grad_zero(self.theta[it-1])
        elif it > 1:
            theta_grad = self.surrogate_grad(self.theta[it-1],self.theta[it-2])
        ell = pi_logistic_map(ell,np.linalg.norm(theta_grad))
        exp_map_theta = exp_map(self.theta[it-1],ell*theta_grad)
        loss = self.local_loss(exp_map_theta) + (ell*np.dot(theta_grad, self.grad_diff)).flatten()
#        pdb.set_trace()
        return(loss)


    def extrinsic_opt(self):
        start_time = time.time()
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
                theta_norm = np.linalg.norm(theta_grad)
#                min_lambda =  pi_logistic_map(minimize(self.lambda_obj,x0=-.5,args=(it,)).x,)
                self.theta[it], min_lambda = self.armijo(it,  theta_grad, theta_norm, tau=.5, kappa=1e-4)
#                self.theta[it] = exp_map(self.theta[it-1], min_lambda*theta_grad)
#                self.theta[it] /= np.linalg.norm(self.theta[it]) # ensure norm of current theta is 1
                local_grad = self.local_loss_grad(self.theta[it])
                if self.verbose >=2:
                    print("%i\t%i\t%.2f\t%s"%(it, current_proc, min_lambda, self.theta[it]))
            else:
                local_grad = None

            self.comm.barrier()
            current_theta = self.comm.bcast(self.theta[it], current_proc)
            self.theta[it] = current_theta
#            local_grad = self.comm.bcast(local_grad,current_proc)
            local_grad = self.local_loss_grad(self.theta[it])
            self.grad_diff = self.global_loss_grad(self.theta[it]) - local_grad

            if np.allclose(np.dot(self.theta[it],self.theta[it-1])**2,1.):
                self.theta = self.theta[:it]
                self.comm.barrier()
                self.theta = self.comm.bcast(self.theta, current_proc)
                break

        if self.verbose >= 1:
            self.final_theta = self.theta[-1]
            self.end_time = time.time() - start_time
            if self.rank==0:
                print("%i\t%.2f\t%s" % (self.P, self.end_time, self.final_theta))
            return(np.hstack((self.P, self.end_time, self.final_theta)))


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    base_seed =8888
    np.random.seed(base_seed)
    trials = 20
    D=100
    results = np.empty((trials,D+2))
    if comm.Get_rank() == 0:
        X=np.random.normal( np.random.normal(size=D), scale = [2]*D,size=(int(1e6),D))
        X_norm = np.linalg.norm(X,axis=1)
        X= np.vstack([X[i]/x_n for i,x_n in enumerate(X_norm)])
    else:
        X = None

    for t in xrange(trials):
        np.random.seed(base_seed + t)
        sph = SphereOptimization(X,verbose=1)
        results[t]=sph.extrinsic_opt()

    if comm.Get_rank() == 0:
        np.savetxt("intrinsic_mean_"+str(comm.Get_size())+".txt",results)
