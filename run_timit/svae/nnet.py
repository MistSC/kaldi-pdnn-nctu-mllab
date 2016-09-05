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


'''Layer Definition'''
class HiddenLayer(object):
    def __init__(self, rng, input_source, input_target, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid, name=''):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=name+'_W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=name+'_b', borrow=True)

        self.W = W
        self.b = b

        lin_output_source = T.dot(input_source, self.W) + self.b
        lin_output_target = T.dot(input_target, self.W) + self.b
        
        self.output_source = (
            lin_output_source if activation is None
            else activation(lin_output_source)
        )
        
        self.output_target = (
            lin_output_target if activation is None
            else activation(lin_output_target)
        )
        
        # parameters of the model
        self.params = [self.W, self.b]
        #self.params = [(name+'_W', self.W), (name+'_b', self.b)]

        
class GaussianSampleLayer(object):
    def __init__(self, mu, log_sigma, n_in, batch_size):
        '''
        This layer is presenting the gaussian sampling process of Stochastic Gradient Variational Bayes(SVGB)

        :type mu: theano.tensor.dmatrix
        :param mu: a symbolic tensor of shape (n_batch, n_in), means the sample mean(mu)
        
        :type log_sigma: theano.tensor.dmatrix
        :param log_sigma: a symbolic tensor of shape (n_batch, n_in), means the log-variance(log_sigma). 
                          here using diagonal variance, so a data has a row variance.
        '''
        seed = 42
        '''
        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
        '''
        srng = T.shared_randomstreams.RandomStreams(seed=seed)

        epsilon = srng.normal((batch_size, n_in))        
        #self.mu = mu
        #self.log_sigma = log_sigma
        #epsilon = np.asarray(rng.normal(size=(batch_size,n_in)), dtype=theano.config.floatX)
        self.output = mu + T.exp(0.5*log_sigma) * epsilon

class CatSampleLayer(object):
    def __init__(self, pi, n_in, batch_size):  
        '''
        This layer is presenting the categorical distribution sampling process of Stochastic Gradient Variational Bayes(SVGB)

        :type pi: theano.tensor.dmatrix
        :param pi: a symbolic tensor of shape (n_batch, n_in), means the probability of each category      
        '''
        
        seed = 42
        srng = T.shared_randomstreams.RandomStreams(seed=seed)
        #generate standard Gumbel distribution from uniform distribution
        epsilon = srng.uniform((batch_size, n_in))     
        
        c = 0.01                
        gamma = T.log(pi + c) + epsilon
        
        self.output = T.eq(gamma / T.max(gamma), T.ones((batch_size, n_in)))
        
        
################################################################################################################
################################################################################################################        
        
'''
Data/Parameter Structure Definition
Here NN parameter refer to Weight, bias. NN structure refer to hidden dimension and accroding activation function 
'''

class NN_struct:
    def __init__(self):
        self.layer_dim = []
        self.activation = []
        
'''
Neural Network Block Definition
here refer to complete Neural Network system block with fixed hidden layer number
'''       
class NN_Block_0L:
    def __init__(self, rng, input_source, input_target, struct, name=''):
        if len(struct.layer_dim) != 2:
            print('used wrong NN Block size')            
                        
        #Output Layer
        self.OL = HiddenLayer(
            rng=rng,
            input_source=input_source,
            input_target=input_target,
            n_in=struct.layer_dim[0],
            n_out=struct.layer_dim[1],
            activation=struct.activation[0],
            name=name+'_OL'
        )
        
        self.output_source = self.OL.output_source
        self.output_target = self.OL.output_target
        
        self.params = self.OL.params


class NN_Block_1L:
    def __init__(self, rng, input_source, input_target, struct):
        if len(struct.layer_dim) != 3:
            print('used wrong NN Block size')            
        
        #Hidden Layer
        self.HL_1 = HiddenLayer(
            rng=rng,
            input_source=input_source,
            input_target=input_target,
            n_in=struct.layer_dim[0],
            n_out=struct.layer_dim[1],
            activation=struct.activation[0],
            name=name+'_L1'
        )
                
        #Output Layer
        self.OL = HiddenLayer(
            rng=rng,
            input_source=self.HL_1.output_source,
            input_target=self.HL_1.output_target,
            n_in=struct.layer_dim[1],
            n_out=struct.layer_dim[2],
            activation=struct.activation[1],
            name=name+'_OL'
        )
        
        self.output_source = self.OL.output_source
        self.output_target = self.OL.output_target
        
        self.params = self.HL_1.params + self.OL.params          
        