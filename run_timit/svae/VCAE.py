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

################################################################################################################
################################################################################################################    
    
    
'''Model Definition/Construct'''

class Supervised_VCAE(object):   
    """
    The semi-supervised model Domain-Adversial Variational Autoencoder
    To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
    The domain label s will constuct inside this class
    For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
    Encoder refer to Encoder NN, Decoder refer to Decoder NN    
    """

    def __init__(self, rng, input_source, input_target, label_source, label_target, batch_size,
                 encoder1_struct, encoder2_struct, encoder3_struct, decoder1_struct, decoder2_struct, alpha, beta, D):
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
        #Encoder 1 Neural Network: present q_\phi({z_y}_n | x_n, d_n)
        d_source = T.zeros([batch_size,1], dtype=theano.config.floatX)
        xd_source = T.concatenate([input_source, d_source], axis=1)
        d_target = T.ones([batch_size,1], dtype=theano.config.floatX)
        xd_target = T.concatenate([input_target, d_target], axis=1)
    
        
        self.Encoder1_mu = nn.NN_Block_0L(
            rng=rng,
            input_source=xd_source,
            input_target=xd_target,
            struct = encoder1_struct,
            name='Encoder1_mu'
        )        

        self.Encoder1_sigma = nn.NN_Block_0L(
            rng=rng,
            input_source=xd_source,
            input_target=xd_target,
            struct = encoder1_struct,
            name='Encoder1_sigma'
        )         
        
        #Sample layer
        self.EC_1_GSL_source = nn.GaussianSampleLayer(
            mu=self.Encoder1_mu.output_source,
            log_sigma=self.Encoder1_sigma.output_source,
            n_in = encoder1_struct.layer_dim[-1],
            batch_size = batch_size
        )
        
        self.EC_1_GSL_target = nn.GaussianSampleLayer(
            mu=self.Encoder1_mu.output_target,
            log_sigma=self.Encoder1_sigma.output_target,
            n_in = encoder1_struct.layer_dim[-1],
            batch_size = batch_size            
        )
        
        zy_dim = encoder1_struct.layer_dim[-1]
        self.EC_zy_S_mu = self.Encoder1_mu.output_source
        self.EC_zy_S_log_sigma = self.Encoder1_sigma.output_source
        self.EC_zy_S_sigma = T.exp(self.EC_zy_S_log_sigma)
        self.EC_zy_T_mu = self.Encoder1_mu.output_target
        self.EC_zy_T_log_sigma = self.Encoder1_sigma.output_target
        self.EC_zy_T_sigma = T.exp(self.EC_zy_T_log_sigma)
        
        self.zy_S = self.EC_1_GSL_source.output
        self.zy_T = self.EC_1_GSL_target.output
        
        self.Encoder1_params = self.Encoder1_mu.params + self.Encoder1_sigma.params
        self.Encoder1_outputs = [self.EC_zy_S_mu, self.EC_zy_S_log_sigma, self.zy_S, self.EC_zy_T_mu, self.EC_zy_T_log_sigma, self.zy_T]
        self.Encoder1_outputs_name = ["EC_zy_S_mu", "EC_zy_S_log_sigma", "zy_S", "EC_zy_T_mu", "EC_zy_T_log_sigma", "zy_T"]        
        
        #------------------------------------------------------------------------
        #Encoder 5 Neural Network: present q_\phi(y_n | {z_1}_n)
        self.Encoder5_pi = nn.NN_Block_0L(
            rng=rng,            
            input_source=self.zy_S,
            input_target=self.zy_T,
            struct = encoder5_struct,
            name='Encoder5_pi'
        )        
               
        y_dim = encoder5_struct.layer_dim[-1]
        self.EC_y_S_pi = self.Encoder5_pi.output_source
        self.EC_y_T_pi = self.Encoder5_pi.output_target

        self.Encoder5_params = self.Encoder5_pi.params        
        self.Encoder5_outputs = [self.EC_y_S_pi, self.EC_y_T_pi]
        self.Encoder5_outputs_name = ["EC_y_S_pi", "EC_y_T_pi"]        
        
        #------------------------------------------------------------------------
        #Encoder 3 Neural Network: present q_\phi({a_y}_n | {z_y}_n, y_n)    
        #Input Append        
        zyy_source = T.concatenate([self.zy_S, label_source], axis=1)        
        zyy_target = T.concatenate([self.zy_T, label_target], axis=1)   
        
        self.Encoder3_mu = nn.NN_Block_0L(
            rng=rng,            
            input_source=zyy_source,
            input_target=zyy_target,
            struct = encoder3_struct,
            name='Encoder3_mu'
        )        

        self.Encoder3_sigma = nn.NN_Block_0L(
            rng=rng,            
            input_source=zyy_source,
            input_target=zyy_target,
            struct = encoder3_struct,
            name='Encoder3_sigma'
        )  
        
        #Sample layer
        self.EC_3_GSL_source = nn.GaussianSampleLayer(
            mu=self.Encoder3_mu.output_source,
            log_sigma=self.Encoder3_sigma.output_source,
            n_in = encoder3_struct.layer_dim[-1],
            batch_size = batch_size            
        )
        
        self.EC_3_GSL_target = nn.GaussianSampleLayer(
            mu=self.Encoder3_mu.output_target,
            log_sigma=self.Encoder3_sigma.output_target,
            n_in = encoder3_struct.layer_dim[-1],
            batch_size = batch_size             
        )
        
        ay_dim = encoder3_struct.layer_dim[-1]
        self.EC_ay_S_mu = self.Encoder3_mu.output_source
        self.EC_ay_S_log_sigma = self.Encoder3_sigma.output_source
        self.EC_ay_S_sigma = T.exp(self.EC_ay_S_log_sigma)
        self.EC_ay_T_mu = self.Encoder3_mu.output_target
        self.EC_ay_T_log_sigma = self.Encoder3_sigma.output_target
        self.EC_ay_T_sigma = T.exp(self.EC_ay_T_log_sigma)
        
        self.ay_S = self.EC_3_GSL_source.output;
        self.ay_T = self.EC_3_GSL_target.output;                
       
        self.Encoder3_params = self.Encoder3_mu.params + self.Encoder3_sigma.params        
        self.Encoder3_outputs = [self.EC_ay_S_mu, self.EC_ay_S_log_sigma, self.ay_S, self.EC_ay_T_mu, self.EC_ay_T_log_sigma, self.ay_T]
        self.Encoder3_outputs_name = ["EC_ay_S_mu", "EC_ay_S_log_sigma", "ay_S", "EC_ay_T_mu", "EC_ay_T_log_sigma", "ay_T"]        
        
#------------------------------------------------------------------------
        #Encoder 2 Neural Network: present q_\phi({z_d}_n | x_n, d_n)            
        
        self.Encoder2_mu = nn.NN_Block_0L(
            rng=rng,
            input_source=xd_source,
            input_target=xd_target,
            struct = encoder2_struct,
            name='Encoder2_mu'
        )        

        self.Encoder2_sigma = nn.NN_Block_0L(
            rng=rng,
            input_source=xd_source,
            input_target=xd_target,
            struct = encoder2_struct,
            name='Encoder2_sigma'
        )         
        
        #Sample layer
        self.EC_2_GSL_source = nn.GaussianSampleLayer(
            mu=self.Encoder2_mu.output_source,
            log_sigma=self.Encoder2_sigma.output_source,
            n_in = encoder2_struct.layer_dim[-1],
            batch_size = batch_size
        )
        
        self.EC_2_GSL_target = nn.GaussianSampleLayer(
            mu=self.Encoder2_mu.output_target,
            log_sigma=self.Encoder2_sigma.output_target,
            n_in = encoder2_struct.layer_dim[-1],
            batch_size = batch_size            
        )
        
        zd_dim = encoder2_struct.layer_dim[-1]
        self.EC_zd_S_mu = self.Encoder2_mu.output_source
        self.EC_zd_S_log_sigma = self.Encoder2_sigma.output_source
        self.EC_zd_S_sigma = T.exp(self.EC_zd_S_log_sigma)
        self.EC_zd_T_mu = self.Encoder2_mu.output_target
        self.EC_zd_T_log_sigma = self.Encoder2_sigma.output_target
        self.EC_zd_T_sigma = T.exp(self.EC_zd_T_log_sigma)
        
        self.zd_S = self.EC_2_GSL_source.output
        self.zd_T = self.EC_2_GSL_target.output
        
        self.Encoder2_params = self.Encoder2_mu.params + self.Encoder2_sigma.params
        self.Encoder2_outputs = [self.EC_zd_S_mu, self.EC_zd_S_log_sigma, self.zd_S, self.EC_zd_T_mu, self.EC_zd_T_log_sigma, self.zd_T]
        self.Encoder2_outputs_name = ["EC_zd_S_mu", "EC_zd_S_log_sigma", "zd_S", "EC_zd_T_mu", "EC_zd_T_log_sigma", "zd_T"]        

        #------------------------------------------------------------------------
        #Encoder 4 Neural Network: present q_\phi({a_d}_n | {d_y}_n, d_n)    
        #Input Append        
        zdd_source = T.concatenate([self.zd_S, d_source], axis=1)        
        zdd_target = T.concatenate([self.zd_T, d_target], axis=1)   
        
        self.Encoder4_mu = nn.NN_Block_0L(
            rng=rng,            
            input_source=zdd_source,
            input_target=zdd_target,
            struct = encoder4_struct,
            name='Encoder4_mu'
        )        

        self.Encoder4_sigma = nn.NN_Block_0L(
            rng=rng,            
            input_source=zdd_source,
            input_target=zdd_target,
            struct = encoder4_struct,
            name='Encoder4_sigma'
        )  
        
        #Sample layer
        self.EC_4_GSL_source = nn.GaussianSampleLayer(
            mu=self.Encoder4_mu.output_source,
            log_sigma=self.Encoder4_sigma.output_source,
            n_in = encoder4_struct.layer_dim[-1],
            batch_size = batch_size            
        )
        
        self.EC_4_GSL_target = nn.GaussianSampleLayer(
            mu=self.Encoder4_mu.output_target,
            log_sigma=self.Encoder4_sigma.output_target,
            n_in = encoder4_struct.layer_dim[-1],
            batch_size = batch_size             
        )
        
        ad_dim = encoder4_struct.layer_dim[-1]
        self.EC_ad_S_mu = self.Encoder4_mu.output_source
        self.EC_ad_S_log_sigma = self.Encoder4_sigma.output_source
        self.EC_ad_S_sigma = T.exp(self.EC_ad_S_log_sigma)
        self.EC_ad_T_mu = self.Encoder4_mu.output_target
        self.EC_ad_T_log_sigma = self.Encoder4_sigma.output_target
        self.EC_ad_T_sigma = T.exp(self.EC_ad_T_log_sigma)
        
        self.ay_S = self.EC_4_GSL_source.output;
        self.ay_T = self.EC_4_GSL_target.output;                
       
        self.Encoder4_params = self.Encoder4_mu.params + self.Encoder4_sigma.params        
        self.Encoder4_outputs = [self.EC_ad_S_mu, self.EC_ad_S_log_sigma, self.ad_S, self.EC_ad_T_mu, self.EC_ad_T_log_sigma, self.ad_T]
        self.Encoder4_outputs_name = ["EC_ad_S_mu", "EC_ad_S_log_sigma", "ad_S", "EC_ad_T_mu", "EC_ad_T_log_sigma", "ad_T"]  		

		
        #------------------------------------------------------------------------
        #Decoder 1 Neural Network: present p_\theta(x_n | {z_y}_n, {z_d}_n)
        zyzd_source = T.concatenate([self.zy_S, zd_S], axis=1)
        zyzd_target = T.concatenate([self.zy_T, zd_T], axis=1)         
                
        self.Decoder1_mu = nn.NN_Block_0L(
            rng=rng,            
            input_source=zyzd_source,
            input_target=zyzd_target,
            struct = decoder1_struct,
            name='Decoder1_mu'
        )        

        self.Decoder1_sigma = nn.NN_Block_0L(
            rng=rng,            
            input_source=zyd_source,
            input_target=zyd_target,
            struct = decoder1_struct,
            name='Decoder1_sigma'
        )  
        
        
        x_dim = decoder1_struct.layer_dim[-1]
        self.DC_x_S_mu = self.Decoder1_mu.output_source
        self.DC_x_S_log_sigma = self.Decoder1_sigma.output_source
        self.DC_x_S_sigma = T.exp(self.DC_x_S_log_sigma)
        self.DC_x_T_mu = self.Decoder1_mu.output_target
        self.DC_x_T_log_sigma = self.Decoder1_sigma.output_target
        self.DC_x_T_sigma = T.exp(self.DC_x_T_log_sigma)       
        
        self.Decoder1_params = self.Decoder1_mu.params + self.Decoder1_sigma.params    
        self.Decoder1_outputs = [self.DC_x_S_mu, self.DC_x_S_log_sigma, self.DC_x_T_mu, self.DC_x_T_log_sigma]
        self.Decoder1_outputs_name = ["DC_x_S_mu", "DC_x_S_log_sigma", "DC_x_T_mu", "DC_x_T_log_sigma"]        
        
        #------------------------------------------------------------------------
        #Decoder 2 Neural Network: present p_\theta({z_y}_n | {a_y}_n, y_n)
        ayy_source = T.concatenate([self.ay_S, label_source], axis=1)        
        ayy_target = T.concatenate([self.ay_T, label_target], axis=1)         
        
        self.Decoder2_mu = nn.NN_Block_0L(
            rng=rng,            
            input_source=ayy_source,
            input_target=ayy_target,
            struct = decoder2_struct,
            name='Decoder2_mu'
        )        

        self.Decoder2_sigma = nn.NN_Block_0L(
            rng=rng,            
            input_source=ayy_source,
            input_target=ayy_target,
            struct = decoder2_struct,
            name='Decoder2_sigma'
        )      
        
        self.DC_zy_S_mu = self.Decoder2_mu.output_source
        self.DC_zy_S_log_sigma = self.Decoder2_sigma.output_source
        self.DC_zy_S_sigma = T.exp(self.DC_zy_S_log_sigma)
        self.DC_zy_T_mu = self.Decoder2_mu.output_target
        self.DC_zy_T_log_sigma = self.Decoder2_sigma.output_target
        self.DC_zy_T_sigma = T.exp(self.DC_zy_T_log_sigma)
        
        self.Decoder2_params = self.Decoder2_mu.params + self.Decoder2_sigma.params         
        self.Decoder2_outputs = [self.DC_zy_S_mu, self.DC_zy_S_log_sigma, self.DC_zy_T_mu, self.DC_zy_T_log_sigma]
        self.Decoder2_outputs_name = ["DC_zy_S_mu", "DC_zy_S_log_sigma", "DC_zy_T_mu", "DC_zy_T_log_sigma"]

        #------------------------------------------------------------------------
        #Decoder 3 Neural Network: present p_\theta({z_d}_n | {a_d}_n, d_n)
        add_source = T.concatenate([self.ad_S, d_source], axis=1)        
        add_target = T.concatenate([self.ad_T, d_target], axis=1)         
        
        self.Decoder3_mu = nn.NN_Block_0L(
            rng=rng,            
            input_source=add_source,
            input_target=add_target,
            struct = decoder3_struct,
            name='Decoder3_mu'
        )        

        self.Decoder3_sigma = nn.NN_Block_0L(
            rng=rng,            
            input_source=add_source,
            input_target=add_target,
            struct = decoder3_struct,
            name='Decoder3_sigma'
        )      
        
        self.DC_zd_S_mu = self.Decoder3_mu.output_source
        self.DC_zd_S_log_sigma = self.Decoder3_sigma.output_source
        self.DC_zd_S_sigma = T.exp(self.DC_zd_S_log_sigma)
        self.DC_zd_T_mu = self.Decoder3_mu.output_target
        self.DC_zd_T_log_sigma = self.Decoder3_sigma.output_target
        self.DC_zd_T_sigma = T.exp(self.DC_zd_T_log_sigma)
        
        self.Decoder3_params = self.Decoder3_mu.params + self.Decoder3_sigma.params         
        self.Decoder3_outputs = [self.DC_zd_S_mu, self.DC_zd_S_log_sigma, self.DC_zd_T_mu, self.DC_zd_T_log_sigma]
        self.Decoder3_outputs_name = ["DC_zd_S_mu", "DC_zd_S_log_sigma", "DC_zd_T_mu", "DC_zd_T_log_sigma"]		
		
                            
        #------------------------------------------------------------------------
        # Error Function Set                
        # KL(q(zy)||p(zy)) -----------
        self.KL_zy_source = er.KLGaussianGaussian(self.EC_zy_S_mu, self.EC_zy_S_log_sigma, self.DC_zy_S_mu, self.DC_zy_S_log_sigma)
        self.KL_zy_target = er.KLGaussianGaussian(self.EC_zy_T_mu, self.EC_zy_T_log_sigma, self.DC_zy_T_mu, self.DC_zy_T_log_sigma)        

        # KL(q(zd)||p(zd)) -----------
        self.KL_zd_source = er.KLGaussianGaussian(self.EC_zd_S_mu, self.EC_zd_S_log_sigma, self.DC_zd_S_mu, self.DC_zd_S_log_sigma)
        self.KL_zd_target = er.KLGaussianGaussian(self.EC_zd_T_mu, self.EC_zd_T_log_sigma, self.DC_zd_T_mu, self.DC_zd_T_log_sigma)
        
        # KL(q(ay)||p(ay)) -----------     
        self.KL_ay_source = er.KLGaussianStdGaussian(self.EC_ay_S_mu, self.EC_ay_S_log_sigma)
        self.KL_ay_target = er.KLGaussianStdGaussian(self.EC_ay_T_mu, self.EC_ay_T_log_sigma)

		# KL(q(ad)||p(ad)) -----------     
        self.KL_ad_source = er.KLGaussianStdGaussian(self.EC_ad_S_mu, self.EC_ad_S_log_sigma)
        self.KL_ad_target = er.KLGaussianStdGaussian(self.EC_ad_T_mu, self.EC_ad_T_log_sigma)
		
        threshold = 0.0000001                 
        # Likelihood q(y) only source data-----------
        self.LH_y_source = - T.sum(- label_source * T.log( T.maximum(self.EC_y_S_pi, threshold)), axis=1)
        self.LH_y_target = - T.sum(- label_target * T.log( T.maximum(self.EC_y_T_pi, threshold)), axis=1)
        #self.LH_y_source = T.nnet.nnet.categorical_crossentropy(self.EC_y_S_pi, label_source)
        
        # Likelihood p(x) ----------- if gaussian
        self.LH_x_source = er.LogGaussianPDF(input_source, self.DC_x_S_mu, self.DC_x_S_log_sigma)
        self.LH_x_target = er.LogGaussianPDF(input_target, self.DC_x_T_mu, self.DC_x_T_log_sigma)

        
        # MMD betwween s, x using gaussian kernel-----------
        #self.MMD = MMD(self.zy_S, self.zy_T, batch_size)
        self.MMD = er.MMDEstimator(rng, self.zy_S, self.zy_T, zy_dim, batch_size, D)  
                
        #Cost function
        tmp = self.KL_zy_source + self.KL_zy_target + self.KL_ay_source + self.KL_ay_target \
			+ self.KL_zd_source + self.KL_zd_target + self.KL_ad_source + self.KL_ad_target \
            + self.LH_x_source + self.LH_x_target + self.LH_y_source * alpha + self.LH_y_target * alpha  
        self.cost = -tmp.mean() + self.MMD * beta        
                        
        # the parameters of the model
        self.params = self.Encoder1_params + self.Encoder2_params + self.Encoder3_params \
					+ self.Encoder3_params + self.Encoder5_params \
					+ self.Decoder1_params + self.Decoder2_params + self.Decoder3_params            
        
        # keep track of model input
        self.input_source = input_source
        self.input_target = input_target            
        
        #Predict Label
        self.y_pred_source = T.argmax(self.EC_y_S_pi, axis=1)
        self.y_pred_target = T.argmax(self.EC_y_T_pi, axis=1)
        

    def source_predict_raw(self):
        return self.EC_y_S_pi
    
    def target_predict_raw(self):
        return self.EC_y_T_pi       
        
    def source_predict(self):
        return self.y_pred_source
    
    def target_predict(self):
        return self.y_pred_target 
        
    def source_errors(self, y):
        #Classification Error
        return T.mean(T.neq(self.y_pred_source, T.argmax(y, axis=1)))

    def target_errors(self, y):
        #Classification Error
        return T.mean(T.neq(self.y_pred_target, T.argmax(y, axis=1)))
		
	def output_variance(self):
		EC_zy_S = T.mean(T.sum(self.EC_zy_S_log_sigma, axis=1))
		EC_zy_T = T.mean(T.sum(self.EC_zy_T_log_sigma, axis=1))
		EC_ay_S = T.mean(T.sum(self.EC_ay_S_log_sigma, axis=1))
		EC_ay_T = T.mean(T.sum(self.EC_ay_T_log_sigma, axis=1))
		EC_zd_S = T.mean(T.sum(self.EC_zd_S_log_sigma, axis=1))
		EC_zd_T = T.mean(T.sum(self.EC_zd_T_log_sigma, axis=1))
		EC_ad_S = T.mean(T.sum(self.EC_ad_S_log_sigma, axis=1))
		EC_ad_T = T.mean(T.sum(self.EC_ad_T_log_sigma, axis=1))		
		DC_zy_S = T.mean(T.sum(self.DC_zy_S_log_sigma, axis=1))
		DC_zy_T = T.mean(T.sum(self.DC_zy_T_log_sigma, axis=1))
		DC_zd_S = T.mean(T.sum(self.DC_zd_S_log_sigma, axis=1))
		DC_zd_T = T.mean(T.sum(self.DC_zd_T_log_sigma, axis=1))		
		DC_x_S = T.mean(T.sum(self.DC_x_S_log_sigma, axis=1))
		DC_x_T = T.mean(T.sum(self.DC_x_T_log_sigma, axis=1))
		
		return [EC_zy_S, EC_zy_T, EC_ay_S, EC_ay_T, EC_zd_S, EC_zd_T, EC_ad_S, EC_ad_T, DC_zy_S, DC_zy_T, DC_zd_S, DC_zd_T, DC_x_S, DC_x_T]    
    '''
    def outputs_mean(self):
        result=[]
        for i in range(len(self.outputs)):
            result[i] = T.mean(self.outputs[i])
        
        return result
    

    def cost(self):
        alpha = 1
        beta = 0.01
        tmp = self.KL_zy_source + self.KL_zy_target + self.KL_ay_source + self.KL_ay_target \
            + self.LH_x_source + self.LH_x_target + self.KL_y_target + self.LH_y_source * alpha            
        
        
        return -tmp.mean() + self.MMD * beta
    '''    
    
    
        