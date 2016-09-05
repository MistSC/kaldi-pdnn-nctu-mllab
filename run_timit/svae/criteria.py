from __future__ import print_function
from collections import OrderedDict

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

################################################################################################################
################################################################################################################

def GaussianKernel(x, y, n_batch):
    '''
    This layer is presenting the gaussian sampling process of Stochastic Gradient Variational Bayes(SVGB)

    :type x,y: theano.tensor.dmatrix
    :param x,y: a symbolic tensor of shape (n_batch, n_in)       
    
    Here use sigma = 1
    '''
    
    sigma=1
    I = T.ones([n_batch, 1], dtype=theano.config.floatX)
    x_tmp = T.sum(T.sqr(x), axis=1, keepdims=True)
    y_tmp = T.sum(T.sqr(y), axis=1, keepdims=True)
    x2 = T.dot(I, x_tmp.T)
    y2 = T.dot(I, y_tmp.T)
    normTmp = x2 - 2*T.dot(y, x.T) + y2.T
    K = T.exp(-1 * normTmp / sigma)
        
    return K    

def MMD(x, y, n_batch):
    
    Kxx = GaussianKernel(x, x, n_batch) / (n_batch**2)
    Kxy = GaussianKernel(x, y, n_batch) / (n_batch**2)
    Kyy = GaussianKernel(y, y, n_batch) / (n_batch**2)
    
    MMD = Kxx - 2*Kxy + Kyy
    
    return MMD

def MMDEstimator(rng, x, y, n_in, n_batch, D):
    '''
    This layer is presenting the gaussian sampling process of Stochastic Gradient Variational Bayes(SVGB)

    :type x,y: theano.tensor.dmatrix
    :param x,y: a symbolic tensor of shape (n_batch, n_in)       
    
    '''    
    gamma = 1
    
    W_values = np.asarray(
        rng.standard_normal(
            size=(n_in, D)
        ),
        dtype=theano.config.floatX
    )

    W = theano.shared(value=W_values, name='MMDEstimator_W', borrow=True)

    b_values = np.asarray(
        rng.uniform(
            low = 0,
            high = 2 * np.pi,
            size=(D,)
        ),
        dtype=theano.config.floatX
    )
    b = theano.shared(value=b_values, name='MMDEstimator_b', borrow=True)

    psi_x = np.sqrt(2.0/D) * T.cos( np.sqrt(2/gamma) * T.dot(x, W) + b)
    psi_y = np.sqrt(2.0/D) * T.cos( np.sqrt(2/gamma) * T.dot(y, W) + b)
    MMD = T.sum(psi_x, axis=0) / n_batch - T.sum(psi_y, axis=0) / n_batch
        
    return MMD.norm(2)

def LogGaussianPDF(x, mu, log_sigma):
    '''
    :type x, mu, sigma: theano.tensor.dmatrix
    :param x, mu, sigma: a symbolic tensor of shape (n_batch, n_in)         
    '''
    
    return T.sum(-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma) - 0.5 * ((x - mu)**2 * T.exp(-log_sigma)), axis=1)

def PossionPDF(x, log_mean):
    '''
    :type x: theano.tensor.dmatrix
    :param x: a symbolic tensor of shape (n_batch, n_in) 
    '''
    #TODO
    return 0

def KLGaussianStdGaussian(mu, log_sigma):
    '''
    -KL(q||p), note we have included minus term
    '''
    KL = 1/2 * T.sum(1 + log_sigma - T.exp(log_sigma) - T.sqr(mu), axis=1)
    return KL

def KLGaussianGaussian(mu1, log_sigma1, mu2, log_sigma2):
    '''
    -KL(q||p), note we have included minus term
    '''
    KL = 1/2 * T.sum(1 + log_sigma1 - log_sigma2 - T.exp(log_sigma1 - log_sigma2) - (mu1 - mu2)**2 * T.exp(-log_sigma2), axis=1)
    return KL   
        