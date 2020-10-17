# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:15:56 2020

Base class for Manifold Optimization

@author: Michael Zhang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.optimize import minimize_scalar
from math import ceil
import numpy as np
import time
import pdb

class ManifoldOpt(object):
    
    def __init__(self, X, theta_init, lambda_init, iters=1000,
                 filename="default.txt",a1_tol=1e-6):
        self.X = X
        self.N,self.D = self.X.shape
        self.iters = int(iters)
        self.lambda_init = lambda_init
        self.theta = np.empty((self.iters,self.D))
        self.theta[0] = theta_init
        self.theta_grad = self.local_loss_grad(self.theta[0])
        self.theta_norm = np.linalg.norm(self.theta_grad)
        self.theta_loss = self.local_loss(self.theta[0])
        self.filename = filename
        self.a1_tol = a1_tol
        
    def local_loss_grad(self,theta):
        raise NotImplementedError()

    def local_loss(self, theta):
        raise NotImplementedError()        
        
    def regularize(self,kappa, theta_new, theta_old):
        raise NotImplementedError()        

    def regularize_grad(self,kappa, theta_new, theta_old):
        raise NotImplementedError()        

    def exp_map(self, theta, v):
        #exp_v(theta)
        raise NotImplementedError()

    def log_map(self,theta,theta_bar):
        #log_map_(theta_bar)(theta)
        raise NotImplementedError()

    def dist(self, theta_0, theta_1):
        return(np.linalg.norm(theta_0-theta_1))

    def accerlation_grad(self,theta_new, theta_old, kappa):
        return(self.local_loss_grad(theta_new) + self.regularize_grad(kappa, theta_new, theta_old) )

    def acceleration_loss(self,theta_new, theta_old, kappa):
        return(self.local_loss(theta_new) + self.regularize(kappa, theta_new, theta_old))

    def print_fun(self, it, iter_time, f):
        print("%i\t%f\t%f\t%f" % (it,self.theta_norm,self.theta_loss,iter_time),file=f)
        
    def adapt_armijo(self, theta, theta_grad, theta_norm,  old_theta = None,
               tau=.95, eta=1e-5, kappa = .1):
        """Armijo adaptive stepsize algorithm"""
        lambda0 = float(self.lambda_init)
        if old_theta is None:
            current_loss = self.local_loss(theta)
        else:
            current_loss = self.acceleration_loss(theta,old_theta,kappa)
        exp_map_theta = self.exp_map(theta,lambda0*theta_grad)
        new_loss = self.acceleration_loss(exp_map_theta,theta,kappa)
        while (new_loss - current_loss) > (eta*tau):
            lambda0 *= tau
            exp_map_theta = self.exp_map(theta,lambda0*theta_grad)
            new_loss = self.acceleration_loss(exp_map_theta,theta,kappa)
            if np.allclose((lambda0 * theta_grad),0.):
                break
        return(exp_map_theta,lambda0)        

    def opt(self):
        """Gradient Descent Optimization"""
        start_time = time.time()
        f = open("gd_"+self.filename,'w')
#        print("%i\t%f\t%f\t%f" % ( 0,np.linalg.norm(self.theta_grad),self.local_loss(self.theta[0]),0), file=f)        
        iter_time = time.time() - start_time
        self.print_fun(0, iter_time, f)
        for it in range(1,self.iters):
            theta_grad = self.local_loss_grad(self.theta[it-1])
            theta_norm = np.linalg.norm(theta_grad)
            self.theta[it],min_lambda =self.adapt_armijo(self.theta[it-1],  theta_grad, theta_norm)
            self.theta_loss = self.local_loss(self.theta[it])
            self.theta_grad = self.local_loss_grad(self.theta[it])
            self.theta_norm = np.linalg.norm(self.theta_grad)
            iter_time = time.time() - start_time
            self.print_fun(it, iter_time, f)
            
    def catalyst_opt(self, S=10,T=5, kappa=.1, alpha = 1.):
        """
        Catalyst optimization
        S : Budget term for finding proximal point
        T : number of iterations to run adaptation step
        kappa : smoothing term
        """
        start_time = time.time()
        nu = np.copy(self.theta[0])
        kappa_init = float(kappa)
        f = open("catalyst_"+self.filename,'w')
        iter_time = time.time() - start_time
        self.print_fun(0, iter_time, f)
        for it in range(1,self.iters):
            for t in range(T):
                if t == 0:
                    adapt_theta, min_lambda = self.adapt_armijo(self.theta[it-1], self.theta_grad, self.theta_norm)
                    theta_grad = self.accerlation_grad(adapt_theta,self.theta[it-1],kappa)#),self.W,self.UW)
                    theta_norm = np.linalg.norm(theta_grad)
                else:
                    adapt_theta, min_lambda = self.adapt_armijo(adapt_theta, theta_grad, theta_norm,old_theta=self.theta[it-1], kappa=kappa)
                    theta_grad = self.accerlation_grad(adapt_theta,self.theta[it-1],kappa)
                    theta_norm = np.linalg.norm(theta_grad)
                adapt_loss = self.local_loss(adapt_theta)
                condition_1 = theta_norm > kappa*self.dist(self.theta[it-1],adapt_theta) 
                condition_2 = adapt_loss > self.theta_loss
                if condition_1 or condition_2:
                    kappa *= 2
                else:
                    break
            approx_step = self.exp_map(alpha*self.log_map(nu,self.theta[it-1]),self.theta[it-1])
            s_log_k = int(ceil(S*np.log(it+2)))
            kappa = (kappa_init+1)**-.5
            for s in range(s_log_k):
                if s == 0:
                    old_theta = np.copy(self.theta[it-1])
                    a1_theta, min_lambda = self.adapt_armijo(self.theta[it-1], self.theta_grad, self.theta_norm,old_theta=approx_step,tau=.05,  kappa=kappa)
                else:
                    old_theta = np.copy(a1_theta)
                    a1_theta, min_lambda = self.adapt_armijo(a1_theta, theta_grad, theta_norm,old_theta=approx_step,tau=.05,  kappa=kappa)
                theta_grad = self.accerlation_grad(a1_theta,nu,kappa)#self.W,self.UW)
                theta_norm = np.linalg.norm(theta_grad)
                old_loss = self.local_loss(old_theta)
                a1_loss = self.local_loss(a1_theta)
                if np.abs(old_loss-a1_loss) < self.a1_tol:
                    break

            nu = self.exp_map(alpha*self.log_map(a1_theta,self.theta[it-1]),self.theta[it-1]) #self.theta + (inv_retract(self.theta,a1_theta)/self.alpha)
            self.alpha = .5*(np.sqrt(alpha**4 + 4.*alpha**2) - alpha**2)
            if a1_loss < adapt_loss:
                self.theta[it]=a1_theta
            else:
                self.theta[it]=adapt_theta
            self.theta_loss = self.local_loss(self.theta[it])
            self.theta_grad = self.local_loss_grad(self.theta[it])
            self.theta_norm = np.linalg.norm(self.theta_grad)
            self.kappa = float(kappa_init)
            iter_time = time.time() - start_time
            self.print_fun(it, iter_time, f)

    def DANE_opt(self, kappa=1.):
        """
        DANE optimization
        kappa : smoothing term
        """
        start_time = time.time()
        f = open("DANE_"+self.filename,'w')
        iter_time = time.time() - start_time
        self.print_fun(0, iter_time, f)
        for it in range(1,self.iters):
            if it == 1:
                theta_grad = self.local_loss_grad(self.theta[it-1])
            else:
                theta_grad = self.accerlation_grad(self.theta[it-1],self.theta[it-2],kappa)
            theta_norm = np.linalg.norm(theta_grad)
            theta_local, min_lambda =self.adapt_armijo(self.theta[it-1],theta_grad,theta_norm)
            self.theta[it] = theta_local
            self.theta_loss = self.local_loss(self.theta[it])
            self.theta_grad = self.local_loss_grad(self.theta[it])
            self.theta_norm = np.linalg.norm(self.theta_grad)
            iter_time = time.time() - start_time
            self.print_fun(it, iter_time, f)

    def RAGD_opt_constant_step(self,beta = 1., mu= 1.):
        """Constant Step RAGD Nesterov Acceleration of Zhang and Sra (2018)"""

        start_time = time.time()
        nu = np.copy(self.theta[0])
        K_1 = np.sqrt(beta**2 + 4*(1+beta)*mu*-self.lambda_init)
        alpha = .5*(K_1 - beta)
        gamma = np.sqrt( (K_1 - beta) / (K_1 + beta)  )*mu
        gamma_bar = gamma*(1+beta)
        f = open("RAGD_constant_"+self.filename,'w')
        iter_time = time.time() - start_time
        self.print_fun(0, iter_time, f)
        for it in range(1,self.iters):
            y_t = self.exp_map(alpha*gamma / (gamma+alpha*mu) * self.log_map(nu,self.theta[it-1]), self.theta[it-1])
            self.theta[it] = self.exp_map(self.lambda_init*self.local_loss_grad(y_t), y_t )
            nu = self.exp_map(y_t, ((1.-alpha)/gamma_bar)*self.log_map(nu,y_t)  -(alpha / gamma_bar)*self.local_loss_grad(y_t))
            self.theta_loss = self.local_loss(self.theta[it])
            self.theta_grad = self.local_loss_grad(self.theta[it])
            self.theta_norm = np.linalg.norm(self.theta_grad)                
            iter_time = time.time() - start_time
            self.print_fun(it, iter_time, f)


    def RAGD_opt(self,beta = 5., mu= 5., gamma=1.):
        """RAGD Nesterov Acceleration of Zhang and Sra (2018)"""
        
        def alpha_eqn(alpha):
            return(alpha**2-(min_lambda*((1.-alpha)*gamma + (alpha*mu))))    
            
        start_time = time.time()
        nu = np.copy(self.theta[0])        
        min_lambda = np.copy(self.lambda_init)
        f = open("RAGD_"+self.filename,'w')
        iter_time = time.time() - start_time
        self.print_fun(0, iter_time, f)
        for it in range(1,self.iters):
            alpha = minimize_scalar(alpha_eqn, bounds = (0,1), method='bounded').x
            gamma_bar = ((1.-alpha)*gamma) + (alpha*mu)
            y_t = self.exp_map(alpha*gamma / (gamma+alpha*mu) * self.log_map(nu,self.theta[it-1]), self.theta[it-1])            
            y_t_grad = self.local_loss_grad(y_t)
            self.theta[it], min_lambda =self.adapt_armijo(y_t,y_t_grad,self.theta_norm)
            nu = self.exp_map(y_t, ((1.-alpha)/gamma_bar)*self.log_map(nu,y_t)  -(alpha / gamma_bar)*y_t_grad)
            gamma = gamma_bar/(1.+beta)
            self.theta_grad = self.local_loss_grad(self.theta[it])
            self.theta_norm = np.linalg.norm(self.theta_grad)    
            self.theta_loss = self.local_loss(self.theta[it])            
            iter_time = time.time() - start_time
            self.print_fun(it, iter_time, f)


if __name__ == '__main__':
    pass                        