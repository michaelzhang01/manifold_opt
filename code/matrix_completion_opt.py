# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:27:07 2020

Grassmann Manifold Matrix Completion Code

@author: Michael Zhang
"""

import numpy as np
from manifold_opt import ManifoldOpt
from utils import sparse_product

class MatrixCompletion(ManifoldOpt):

    def __init__(self, X, X_star, train_mask, test_idx,
                 theta_init, lambda_init, iters=50,
                 filename="matrix_completion.txt",a1_tol=1e-6,
                 matrix_rank=5, regularization =.01):
        self.X = X
        self.X_star = X_star
        self.train_mask = train_mask
        self.test_idx = test_idx
        self.N,self.D = self.X.shape
        self.iters = int(iters)
        self.matrix_rank = int(matrix_rank)
        self.lambda_init = lambda_init
        self.filename = filename
        self.a1_tol = a1_tol
        self.regularization = float(regularization)
        self.theta = {-2: None, -1: None, 0: theta_init} # needed to play nicely with base class
        self.theta_grad = self.local_loss_grad(self.theta[0])
        self.theta_norm = np.linalg.norm(self.theta_grad)
        self.theta_loss = self.local_loss(self.theta[0])

    def local_loss_grad(self,theta):
        UTU_inv_U_T = np.linalg.solve(np.dot(theta.T,theta),theta.T)
        W = self.X.T.dot(UTU_inv_U_T.T).T
        UW_prod = sparse_product(theta,W,self.train_mask).tocsc()
        loss_grad = (UW_prod - self.X) - (self.regularization*UW_prod)
        loss_grad = loss_grad.dot(W.T)
        return(loss_grad)

    def regularize(self, kappa, theta_new, theta_old):
        VTU_inv_V = np.linalg.solve(theta_new.T.dot(theta_old), theta_new.T)
        square_norm  = np.trace(VTU_inv_V.dot(VTU_inv_V.T)) - self.matrix_rank
        return(.5*kappa*square_norm)

    def regularize_grad(self,kappa, theta_new, theta_old):
        VT_U = theta_new.T.dot(theta_old)
        square_term = np.linalg.solve(VT_U.dot(VT_U.T),theta_new.T).T
        loss_grad = theta_new.T.dot(square_term)
        VTU_inv_loss = np.linalg.solve(VT_U, loss_grad)
        loss_grad =  -2.*theta_old.dot(VTU_inv_loss)
        loss_grad += 2.*square_term
        return(kappa*loss_grad)
        
    def local_loss(self,theta):
        UTU_inv_U_T = np.linalg.solve(np.dot(theta.T,theta),theta.T)
        W = self.X.T.dot(UTU_inv_U_T.T).T
        UW_prod = sparse_product(theta,W,self.train_mask).tocsc()
        loss = np.linalg.norm((UW_prod - self.X)[self.X.nonzero()])**2
        loss += self.regularization*np.linalg.norm(W)**2
        loss -= self.regularization*np.linalg.norm(UW_prod[self.X.nonzero()])**2
        return(.5*loss)

    def exp_map(self, theta, v):
        return(theta + v)

    def log_map(self,theta,theta_bar):
        UT_U =  theta_bar.T.dot(theta)
        UTU_inv_U_T = np.linalg.solve(UT_U,theta.T).T
        return(UTU_inv_U_T - theta_bar)

        
    def test_set_predict(self,it):
        UTU_inv_U_T = np.linalg.solve(np.dot(self.theta[it].T,self.theta[it]),self.theta[it].T)
        W = self.X.T.dot(UTU_inv_U_T.T).T
        UW_test = sparse_product(a=self.theta[it],b=W,c=self.test_idx)
        return(np.sqrt(np.power((UW_test - self.X_star), 2).mean()))

    def print_fun(self, it, iter_time, f):
        theta_rmse = self.test_set_predict(it)
        print("%i\t%f\t%f\t%f\t%f" % (it,self.theta_norm,self.theta_loss,theta_rmse,iter_time),file=f)
        del(self.theta[it-2])
        
if __name__ == '__main__':    
    from scipy.sparse.linalg import svds as sparse_svds    
    from scipy.sparse import load_npz, coo_matrix
    from scipy.io import loadmat
    base_seed = 8888
    np.random.seed(base_seed)
    matrix_rank = 5
    regularization = .01
    X = load_npz("../data/netflix_sparse.npz")
    nnz_idx = X.nonzero()
#    N_p_nnz = X.nnz
    N_p_nnz = 10000000 # truncate to first million obs.
    train_mask = coo_matrix((np.ones(N_p_nnz,dtype=int), (nnz_idx[0][:N_p_nnz], nnz_idx[1][:N_p_nnz])), shape=(X.shape))
    X = X.multiply(train_mask)
    netflix_dict = loadmat("../data/netflix_dict.mat")
    X_star, test_idx, mean_rating = netflix_dict['X_star'][0], netflix_dict['test_idx'], netflix_dict['mean_rating'][0,0]
    test_idx = (test_idx[0], test_idx[1])
    theta, _, _ = sparse_svds(X, k = matrix_rank)

    UTU_inv_U_T = np.linalg.solve(np.dot(theta.T,theta),theta.T)
    W = X.T.dot(UTU_inv_U_T.T).T
    UW_prod = sparse_product(theta,W,train_mask).tocsc()
    loss_grad = (UW_prod - X) - (regularization*UW_prod)
    loss_grad = loss_grad.dot(W.T)
    grad_norm = np.linalg.norm(loss_grad)
    lambda_init = -50000. / grad_norm
    netflix = MatrixCompletion(X, X_star, train_mask, test_idx, theta, 
                               lambda_init, matrix_rank = matrix_rank, 
                               regularization = regularization)
    netflix.catalyst_opt()

    netflix = MatrixCompletion(X, X_star, train_mask, test_idx, theta, 
                               lambda_init, matrix_rank = matrix_rank, 
                               regularization = regularization)
    netflix.opt()
    netflix = MatrixCompletion(X, X_star, train_mask, test_idx, theta, 
                               lambda_init, matrix_rank = matrix_rank, 
                               regularization = regularization)
    netflix.DANE_opt()
    netflix = MatrixCompletion(X, X_star, train_mask, test_idx, theta, 
                               lambda_init, matrix_rank = matrix_rank, 
                               regularization = regularization)
    netflix.RAGD_opt()
