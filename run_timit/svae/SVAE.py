from __future__ import print_function
from collections import OrderedDict

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

import nnet as nn
import criteria	as er
import util

'''Model Definition/Construct'''

class Supervised_VAE(object):   
    """
    The semi-supervised model Domain-Adversial Variational Autoencoder
    To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
    The domain label s will constuct inside this class
    For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
    Encoder refer to Encoder NN, Decoder refer to Decoder NN    
    """

    def __init__(self, rng, input_x, label_y, batch_size,
                 encoder1_struct, decoder1_struct):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input_source: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
        
        :type input_target: theano.tensor.TensorType
        :param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)        

        :type xxx_struct: class NN_struct
        :param xxx_strucat: define the structure of each NN
        """
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
            rng=rng,
            input=phi_xy_in,
            struct = phi_1_struct,
            name='Encoder1'
        )    
		'''
		
        self.phi_mu = nn.Stacked_NN_0L(
            rng=rng,
            input=phi_xy_in,
            struct = phi_1_struct,
            name='Encoder1_mu'
        )        

        self.phi_sigma = nn.Stacked_NN_0L(
            rng=rng,
            input=phi_xy_in,
            struct = phi_1_struct,
            name='Encoder1_sigma'
        )         
        
        #Sample layer
        self.phi_z = nn.GaussianSampleLayer(
            mu=self.phi_z_mu,
            log_sigma=self.phi_z_sigma,
            n_in = phi_1_struct.layer_dim[-1],
            batch_size = batch_size
        )
       
        EC_z_dim = phi_1_struct.layer_dim[-1]
        self.EC_mu = self.phi_mu.output
        self.EC_log_sigma = self.phi_sigma.output
        self.EC_sigma = T.exp(self.EC_log_sigma)
        self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
        self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
        self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]        
        
        #------------------------------------------------------------------------
        #Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		'''
        theta_1 = nn.NNLayer(
            rng=rng,            
            input_source=input_x,
            struct = theta_1_struct,
            name='Decoder1'
        ) 
		'''
        self.theta_mu = nn.Stacked_NN_0L(
            rng=rng,            
            input=input_x,
            struct = theta_1_struct,
            name='Decoder1_mu'
        )        

        self.theta_sigma = nn.Stacked_NN_0L(
            rng=rng,            
            input=input_x,
            struct = theta_1_struct,
            name='Decoder1_sigma'
        )  
		
		self.DC_mu = self.theta_mu.output
        self.DC_log_sigma = self.theta_sigma.output
        self.DC_sigma = T.exp(self.DC_log_sigma)
		
        self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma]
        self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma"]        
        
		#------------------------------------------------------------------------
        #Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
        self.theta_xz_in = T.concatenate([input_x, phi_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
            rng=rng,            
            input=self.theta_xz_in,
            struct = theta_2_struct,
            name='Predict1'
        ) 
		
        self.theta_pi = nn.Stacked_NN_0L(
            rng=rng,            
            input=self.theta_2.output,
            struct = theta_pi_struct,
            name='Predict1_pi'
        )        
		
		self.y_hat = nn.CatSampleLayer(
            pi=self.theta_pi.output,
            n_in = theta_pi_struct.layer_dim[-1],
            batch_size = batch_size 
        )
		
		PR_y_hat_dim = theta_pi_struct.layer_dim[-1]
		self.PR_theta_2 = self.theta_2.output
        self.PR_pi = self.theta_pi.output
		self.PR_y_hat = self.y_hat.output
		
        self.theta_2_params = self.theta_2.params + self.theta_pi.params + self.y_hat.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_pi, self.PR_y_hat]
        self.theta_2_outputs_name = ["PR_theta_2", "PR_pi", "PR_y_hat"]  
                            
        #------------------------------------------------------------------------
        # Error Function Set                
        # KL(q(z|x,y)||p(z|x)) -----------
        self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_sigma, self.DC_mu, self.DC_sigma)
             
        threshold = 0.0000001                 
        # Cross entropy
        self.CE = T.nnet.categorical_crossentropy(self.PR_y_hat, label_y);
                
        #Cost function
        self.cost = -self.KL - self.CE
                        
        # the parameters of the model
        self.params = self.phi_1_params + self.theta_1_params + self.theta_2_params
        
        # all output of VAE
        self.outputs = self.phi_1_outputs + self.theta_1_outputs + self.theta_2_outputs
        self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name + self.theta_2_outputs_name

    