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

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y        

################################################################################################################
################################################################################################################


def ArrayZeroCheck(array):
    zero_percentage = len(np.where(array == 0)[0]) / len(array)
    return zero_percentage

def ArrayNaNCheck(array):
    nan_flag = np.sum(np.isnan(array)) > 0
    return nan_flag
    

def OutputsNaNCheck(epoch, minibatch_index, n_batches, params_name=None, outputs_name = None, VAE_params=None,
                    VAE_gparams=None, VAE_outputs=None):
            
    if VAE_params is not None:
        for i in range(len(VAE_params)):            
            if ArrayNaNCheck(VAE_params[i]) > 0:
                print('          NaNCheck-Parameters : epoch %i, minibatch %i/%i : %s parameters is nan' % 
                      (
                        epoch,
                        minibatch_index + 1,
                        n_batches,
                        params_name[i].name
                    )
                )    
                
    if VAE_gparams is not None:
        for i in range(len(VAE_gparams)):            
            if ArrayNaNCheck(VAE_gparams[i]) > 0:
                print('          NaNCheck-Gardient : epoch %i, minibatch %i/%i : %s gradient is nan' % 
                      (
                        epoch,
                        minibatch_index + 1,
                        n_batches,
                        params_name[i].name
                    )
                )  
                
    if VAE_outputs is not None:
        for i in range(len(VAE_outputs)):
            if ArrayNaNCheck(VAE_outputs[i]) > 0:
                print('          NaNCheck-Outputs : epoch %i, minibatch %i/%i : %s output is nan' % 
                      (
                        epoch,
                        minibatch_index + 1,
                        n_batches,
                        outputs_name[i]
                    )
                )

    '''
    if VAE_gparams is not None:
        for i in range(len(VAE_gparams)):            
            if ArrayZeroCheck(VAE_gparams[i]) > 0.9:
                print('          ZeroCheck-Gardient : epoch %i, minibatch %i/%i : %s gradient is zero' % 
                      (
                        epoch,
                        minibatch_index + 1,
                        n_batches,
                        params_name[i].name
                    )
                )                
    '''
    
    
    return None
    
    
        