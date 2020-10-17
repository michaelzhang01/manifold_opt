# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:50:46 2018

Manifold optimization for low-rank matrix completion on Netflix data.
Obtain netflix data here: http://i.stanford.edu/hazy/victor/download/
To replicate code, shuffle data and divide data into 64 partitions

@author: Michael Zhang
"""
import time
import os
import numpy as np
from mpi4py import MPI
from scipy import sparse
from scipy.sparse.linalg import svds as sparse_svds
#from scipy.optimize import minimize_scalar
#from numba import jit
#import tarfile
import pdb
import argparse


def sparse_product(a,b, c):
    """ computes only the matrix product values where c_ij = 1"""
    if sparse.isspmatrix_coo(c):
        value = np.sum(a[c.row,:] * b[:,c.col].T,axis=1)
        product_mat = sparse.coo_matrix((value, (c.row,c.col)), shape=c.shape ).tocsc()
    elif type(c) is tuple:
        row, col = c
        assert(len(row) == len(col))
        product_mat = np.sum(a[row,:] * b[:,col].T,axis=1)
    return(product_mat)


def pi_logistic_map(x, grad_length):
    """maps real line to -pi / grad_length <= x <= pi grad_length"""
    log_map = ((2.*np.pi/grad_length) / (1.+  np.exp(-x)))  - (np.pi/grad_length)
    return(log_map)

class MatrixCompletion(object):

    def __init__(self, training_dir, test_file, iters=200, alpha=.05,
                 matrix_rank = 1, regularization =.01,verbose=2,
                 tar_file="netflix_user.tar.gz"):
        self.iters = int(iters)
        self.comm = MPI.COMM_WORLD
        self.P = self.comm.Get_size()
        assert(self.P <= 64)
        self.rank = self.comm.Get_rank()
        self.D = 17770
        self.N = 480189
        self.matrix_rank = int(matrix_rank)
        self.regularization = float(regularization)
        self.verbose=int(verbose)
        self.alpha = float(alpha)
        assert((self.regularization >.0) and (self.regularization <= 1.))
#        assert( (train_prop > 0.) and (train_prop < 1.) )
#        self.train_prop = float(train_prop)
#        training_dir = str(training_dir)
#        observations = int(observations)

        training_dir=os.path.abspath(str(training_dir))
#        test_dir=os.path.abspath(str(test_dir))
        train_file_list= np.random.permutation(list(os.listdir(training_dir)))
        assert(self.P <= len(train_file_list))
        train_file_list =np.array_split(train_file_list, self.P)
        train_file_list = self.comm.scatter(train_file_list)
#        test_file_list= np.random.permutation(list(os.listdir(test_dir)))
#        assert(self.P <= len(test_file_list))
        self.X_local = sparse.coo_matrix((self.N,self.D))
        for file in train_file_list:
#            train_file = os.path.join(training_dir, train_file_list)
            train_file = os.path.join(training_dir, file)
#        print(train_file)
#        test_file = os.path.join(test_dir, test_file_list[self.rank])
#        print(test_file)
#        test_file = os.path.join(training_dir, file_list[self.rank])
            raw_train_data_p = np.loadtxt(train_file, dtype='int')
            raw_train_data_p[:,:2] -= 1
            self.X_local += sparse.coo_matrix((raw_train_data_p[:,2], (raw_train_data_p[:,0], raw_train_data_p[:,1])), shape=(self.N,self.D),dtype=float )

        self.X_local = self.X_local.tocsc()
#        self.X_local = sparse.coo_matrix((raw_train_data_p[:,2], (raw_train_data_p[:,0], raw_train_data_p[:,1])), shape=(self.N,self.D),dtype=float ).tocsc()
#        except:
#            pdb.set_trace()
        del raw_train_data_p
        raw_test_data_p = np.loadtxt(test_file, dtype='int')
        raw_test_data_p[:,:2] -= 1
        nnz_star_p, _ = raw_test_data_p.shape
        self.X_star = np.copy(raw_test_data_p[:,2]).astype(float)
#        self.test_mask = sparse.coo_matrix((np.ones(nnz_star_p,dtype=int), (raw_test_data_p[:,0], raw_test_data_p[:,1])), shape=(self.N, self.D))
        self.test_idx = (raw_test_data_p[:,0], raw_test_data_p[:,1])
        del raw_test_data_p

        self.comm.barrier()
        self.nnz_star = self.X_star.size
        self.nnz_idx = self.X_local.nonzero()
#        self.zero_idx = np.where(self.X_local.A==0)
        self.N_p_nnz = self.X_local.nnz
        self.train_mask = sparse.coo_matrix((np.ones(self.N_p_nnz,dtype=int), (self.nnz_idx[0], self.nnz_idx[1])), shape=(self.N, self.D))
        self.N_nnz = self.comm.allreduce(self.N_p_nnz)
        self.mean_rating = self.comm.allreduce(self.X_local.sum()) / self.N_nnz
        self.X_local[self.nnz_idx] -= self.mean_rating
        self.X_star -= self.mean_rating


        if self.rank == 0:
            self.theta, _, _ = sparse_svds(self.X_local, k = self.matrix_rank)
#            mat_rank = 0
#            while mat_rank != self.matrix_rank:
#                self.theta = np.random.normal(size=(self.N, self.matrix_rank))
#                mat_rank = np.linalg.matrix_rank(self.theta)
            self.theta_T_theta = np.dot(self.theta.T,self.theta)
            UTU_inv_U_T = np.linalg.solve(self.theta_T_theta,self.theta.T)
            self.W = self.X_local.T.dot(UTU_inv_U_T.T).T
            self.UW = sparse_product(self.theta,self.W,self.train_mask)
#            local_grad = self.local_loss_grad(self.theta,theta_T_theta,UTU_inv_U_T)
            local_grad = self.local_loss_grad(self.theta,self.W,self.UW)
#            V, _, _, _=np.linalg.lstsq(self.theta.T,theta_T_theta)

        else:
            self.theta= None
            UTU_inv_U_T = None
#            V = None
            self.theta_T_theta = None
            self.W = None
#            self.UW = None
            local_grad = None

        self.theta = self.comm.bcast(self.theta)
        self.theta_T_theta = self.comm.bcast(self.theta_T_theta)
        UTU_inv_U_T = self.comm.bcast(UTU_inv_U_T)
        self.W = self.comm.bcast(self.W)

        if self.rank !=0:
#            self.W = self.X_local.T.dot(UTU_inv_U_T.T).T
            self.UW = sparse_product(self.theta,self.W,self.train_mask)
            local_grad = self.local_loss_grad(self.theta,self.W,self.UW)

#        self.grad_diff = self.global_loss_grad(self.theta,theta_T_theta,UTU_inv_U_T) - local_grad
        self.grad_diff = local_grad - self.comm.allreduce(local_grad)


#    def local_loss_grad(self,theta, theta_T_theta = None, UTU_inv_U_T = None, W = None, UW_prod = None):
    def local_loss_grad(self,theta, W, UW_prod):
#        if theta_T_theta is None:
#            theta_T_theta = np.dot(theta.T,theta)
#        if UTU_inv_U_T is None:
#            UTU_inv_U_T = np.linalg.solve(theta_T_theta,theta.T)
#        if W is None:
#            W = self.X_local.T.dot(UTU_inv_U_T.T).T
#        if UW_prod is None:
#            UW_prod =sparse_product(theta,W,self.train_mask).tocsc()
#        grad_1 = sparse.csc_matrix.dot((UW_prod - self.X_local),W.T)
#        grad_1 -= self.regularization*UW_prod.dot(W.T)
        grad = (UW_prod - self.X_local) - (self.regularization*UW_prod)
        grad = grad.dot(W.T)
#        grad = (UW_prod - self.X_local)
#        grad += self.regularization*np.linalg.norm(W)**2
#        grad -= self.regularization*UW_prod.dot(W.T)

#        grad[self.zero_idx] *= self.regularization
#        grad = np.dot(W,grad).T
        return(grad )


    def local_loss(self,theta, W = None, UW_prod = None):
        if W is None:
            UTU_inv_U_T = np.linalg.solve(np.dot(theta.T,theta),theta.T)
            W = self.X_local.T.dot(UTU_inv_U_T.T).T
        if UW_prod is None:
            UW_prod =sparse_product(theta,W,self.train_mask).tocsc()
        loss = np.linalg.norm((UW_prod - self.X_local)[self.X_local.nonzero()])**2
        loss += self.regularization*np.linalg.norm(W)**2
        loss -= self.regularization*np.linalg.norm(UW_prod[self.X_local.nonzero()])**2
        return(.5 * loss)

    def surrogate_loss(self,theta, W = None, UW_prod = None):
        return(self.local_loss(theta,W, UW_prod) - np.trace(np.dot(theta.T, self.grad_diff)))

    def surrogate_grad(self,theta, theta_T_theta, W, UW_prod):
        grad = self.local_loss_grad(theta,W,UW_prod)
        grad -= self.grad_diff
        VTV_inv_V_T =np.linalg.solve(theta_T_theta,np.dot(theta.T,self.grad_diff))
        grad += np.dot(theta,VTV_inv_V_T)
        return(grad)

    def surrogate_grad_zero(self,theta, W, UW_prod):
        return(self.local_loss_grad(theta, W, UW_prod) - self.grad_diff)

    def lambda_obj(self, ell, theta_grad, theta_norm):
        ell = pi_logistic_map(ell,theta_norm)
        exp_map_theta = self.theta + (ell*theta_grad)
        loss = self.local_loss(exp_map_theta) - ell*(np.trace(np.dot(theta_grad.T, self.grad_diff)))
        return(loss)

    def lambda_step_size(self, it, theta_norm):
        ell = -np.pi / theta_norm
        ell*= np.exp(-it)**(self.alpha)
#        ell =
        return(ell)

    def armijo(self, theta, theta_grad, theta_norm, tau=.5, kappa=1e-4):
        lambda0 = -np.pi / theta_norm
#        try:
        current_loss = self.surrogate_loss(theta,self.W,self.UW)
#        except:
#            pdb.set_trace()
        retract_theta = theta+(lambda0 * theta_grad)
        new_loss = self.surrogate_loss(retract_theta)
        path_derivative = (np.trace(np.dot(theta_grad.T, lambda0*theta_grad)))
        while (new_loss - current_loss) > (kappa*path_derivative):
            lambda0 *= tau
            retract_theta = theta+(lambda0 * theta_grad)
            new_loss = self.surrogate_loss(retract_theta)
            path_derivative = (np.trace(np.dot(theta_grad.T, lambda0*theta_grad)))
        return(retract_theta,lambda0)


    def optimize(self):
        start_time = time.time()
        start_time = self.comm.bcast(start_time)
        for it in xrange(self.iters):
            current_proc = it % int(self.P) # select processor
            if self.rank == current_proc:
                if it == 0:
                    theta_grad = self.surrogate_grad_zero(self.theta,self.W,self.UW)
                else:
                    theta_grad = self.surrogate_grad(self.theta,self.theta_T_theta,self.W,self.UW)

                theta_norm = np.linalg.norm(theta_grad)
                self.theta, min_lambda = self.armijo(self.theta, theta_grad, theta_norm)
                iter_time = time.time() - start_time
                self.theta_T_theta = np.dot(self.theta.T,self.theta)

                UTU_inv_U_T = np.linalg.solve(self.theta_T_theta,self.theta.T)
                self.W = self.X_local.T.dot(UTU_inv_U_T.T).T
                self.UW =sparse_product(self.theta,self.W,self.train_mask)
                local_grad = self.local_loss_grad(self.theta,self.W,self.UW)
                UW_test =sparse_product(self.theta,self.W,self.test_idx)
                RMSE = np.sqrt(np.power( (UW_test - self.X_star) ,2).mean())
#                RMSE_train = np.power( (self.UW - self.X_local[self.nnz_idx]) ,2).sum()
            else:
#                min_lambda = None
                theta_norm = None
                iter_time = None
                self.theta_T_theta = None
                UTU_inv_U_T = None
                local_grad=None

            self.comm.barrier()
            theta_norm = self.comm.bcast(theta_norm,current_proc)
            self.theta = self.comm.bcast(self.theta, current_proc)
            self.theta_T_theta = self.comm.bcast(self.theta_T_theta,current_proc)
            UTU_inv_U_T = self.comm.bcast(UTU_inv_U_T,current_proc)

            if self.rank != current_proc:
                self.W = self.X_local.T.dot(UTU_inv_U_T.T).T
                self.UW =sparse_product(self.theta,self.W,self.train_mask)
                local_grad = self.local_loss_grad(self.theta,self.W,self.UW)
#                UW_test =sparse_product(self.theta,self.W,self.test_idx)
#                RMSE = np.power( (UW_test - self.X_star) ,2).sum()
#                RMSE_train = np.power( (self.UW - self.X_local[self.nnz_idx]) ,2).sum()



            self.grad_diff = local_grad - self.comm.allreduce(local_grad)
#            RMSE = self.comm.reduce(RMSE,root=current_proc)
#            RMSE_train = self.comm.reduce(RMSE_train,root=current_proc)

            if self.rank == current_proc:
                if self.verbose >=2:
#                    RMSE /= self.nnz_star
#                    RMSE = np.sqrt(RMSE)
#                    RMSE_train /= self.N_nnz
#                    RMSE_train = np.sqrt(RMSE_train)
                    print("%i\t%i\t%.5f\t%f\t%.2f"%(it, current_proc, min_lambda, RMSE, iter_time))

            if np.allclose(theta_norm,0.):
                self.comm.barrier()
                break
#            else:
#                self.grad_diff *= -1.
#                self.grad_diff += local_grad


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    parser = argparse.ArgumentParser(description="Accelerate inference code for Multinomial DP mixture from Zhang and Perez-Cruz (2017)")
    parser.add_argument('--sparse', type=float, default=.01,
                        help='Sparsity regularization parameter')
    parser.add_argument('--rank', type=int, default=10,
                        help='Matrix rank')
    args = parser.parse_args()
    regularization = args.sparse
    matrix_rank = args.rank
    base_seed = 8888
    np.random.seed(base_seed)
    netflix=MatrixCompletion(training_dir="../data/netflix_train",
                             test_file="../data/netflix_probe_set.dat.gz",
                             matrix_rank = matrix_rank,
                             regularization = regularization)
    netflix.optimize()