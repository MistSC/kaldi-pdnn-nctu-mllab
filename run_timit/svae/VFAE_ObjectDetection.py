%matplotlib inline
import matplotlib.pyplot as plt

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
import VFAE

def object_reconition_test(s):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """    

    if s == 0 :
        print('Semi-Supervised Learning')
    else :
        print('Supervised Learning')    
    
    '''Load Data'''
    source_file = '/home/cwhuang/Dataset/Office_Object/dslr_SURF_L10.npy'
    target_file = '/home/cwhuang/Dataset/Office_Object/webcam_SURF_L10.npy'
    
    source_data = np.load(source_file)
    target_data = np.load(target_file)
                
    train_ftd_source, train_labeld_source = source_data[0]
    valid_ftd_source, valid_labeld_source = source_data[1]
    test_ftd_source, test_labeld_source = source_data[2]
    
    train_ftd_target, train_labeld_target = target_data[0]
    valid_ftd_target, valid_labeld_target = target_data[1]
    test_ftd_target, test_labeld_target = target_data[2]
    
    #Make Source & Target size same by discard part data
    if train_ftd_source.shape[0] > train_ftd_target.shape[0]:
        train_ftd_source = train_ftd_source[0:train_ftd_target.shape[0], :]
        train_labeld_source = train_labeld_source[0:train_labeld_target.shape[0], :]
        valid_ftd_source = valid_ftd_source[0:valid_ftd_target.shape[0], :]
        valid_labeld_source = valid_labeld_source[0:valid_labeld_target.shape[0], :]
        test_ftd_source = test_ftd_source[0:test_ftd_target.shape[0], :]
        test_labeld_source = test_labeld_source[0:test_labeld_target.shape[0], :]            
    elif train_ftd_source.shape[0] < train_ftd_target.shape[0]:
        train_ftd_target = train_ftd_target[0:train_ftd_source.shape[0], :]
        train_labeld_target = train_labeld_target[0:train_labeld_source.shape[0], :]
        valid_ftd_target = valid_ftd_target[0:valid_ftd_source.shape[0], :]
        valid_labeld_target = valid_labeld_target[0:valid_labeld_source.shape[0], :]
        test_ftd_target = test_ftd_target[0:test_ftd_source.shape[0], :]
        test_labeld_target = test_labeld_target[0:test_labeld_source.shape[0], :]                      
    
    
    train_ftd_source, train_labeld_source = util.shared_dataset((train_ftd_source, train_labeld_source))
    valid_ftd_source, valid_labeld_source = util.shared_dataset((valid_ftd_source, valid_labeld_source))
    test_ftd_source, test_labeld_source = util.shared_dataset((test_ftd_source, test_labeld_source))
    
    train_ftd_target, train_labeld_target = util.shared_dataset((train_ftd_target, train_labeld_target))
    valid_ftd_target, valid_labeld_target = util.shared_dataset((valid_ftd_target, valid_labeld_target))
    test_ftd_target, test_labeld_target = util.shared_dataset((test_ftd_target, test_labeld_target))
            
    '''
    n_train_source_batches = train_ftd_source.shape[0] // batch_size
    n_valid_source_batches = train_ftd_source.shape[0] // batch_size
    n_test_source_batches = train_ftd_source.shape[0] // batch_size        

    n_train_target_batches = train_ftd_target.shape[0] // batch_size
    n_valid_target_batches = train_ftd_target.shape[0] // batch_size
    n_test_target_batches = train_ftd_target.shape[0] // batch_size 
    '''    
    
    '''
    print(train_ftd_source.get_value(borrow=True).shape[0])
    print(train_ftd_target.get_value(borrow=True).shape[0])    
    print(valid_ftd_source.get_value(borrow=True).shape[0])
    print(valid_ftd_target.get_value(borrow=True).shape[0])
    print(test_ftd_source.get_value(borrow=True).shape[0])
    print(test_ftd_target.get_value(borrow=True).shape[0])

    print(train_labeld_source.get_value(borrow=True).shape[0])
    print(train_labeld_target.get_value(borrow=True).shape[0])    
    print(valid_labeld_source.get_value(borrow=True).shape[0])
    print(valid_labeld_target.get_value(borrow=True).shape[0])
    print(test_labeld_source.get_value(borrow=True).shape[0])
    print(test_labeld_target.get_value(borrow=True).shape[0])

    print(train_ftd_source.get_value(borrow=True).shape[1])
    print(train_ftd_target.get_value(borrow=True).shape[1])    
    print(valid_ftd_source.get_value(borrow=True).shape[1])
    print(valid_ftd_target.get_value(borrow=True).shape[1])
    print(test_ftd_source.get_value(borrow=True).shape[1])
    print(test_ftd_target.get_value(borrow=True).shape[1])    
    
    print(train_labeld_source.get_value(borrow=True).shape[1])
    print(train_labeld_target.get_value(borrow=True).shape[1])    
    print(valid_labeld_source.get_value(borrow=True).shape[1])
    print(valid_labeld_target.get_value(borrow=True).shape[1])
    print(test_labeld_source.get_value(borrow=True).shape[1])
    print(test_labeld_target.get_value(borrow=True).shape[1])
    '''

    '''Coefficient Initial'''        
    batch_size = 14
    epsilon_std = 0.01
    n_epochs = 50
    learning_rate = 0.0001
    D = 800
    alpha = 10 # Weight of classification error
    beta = 100 # Weight of MMD penalty 
    
    n_train_batches = train_ftd_source.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_ftd_source.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_ftd_source.get_value(borrow=True).shape[0] // batch_size         
    print(
        'number of minibatch at one epoch: train  %i, validation %i, test %i' %
        (n_train_batches, n_valid_batches, n_test_batches)
    )
    
    z_dim = 100 #dimension of latent feature
    a_dim = 50 #dimension of prior of latent feature
    x_dim = train_ftd_source.get_value(borrow=True).shape[1]
    y_dim = train_labeld_target.get_value(borrow=True).shape[1]
    d_dim = 2
    activation = None
    
    encoder1_struct=nn.NN_struct()
    encoder1_struct.layer_dim = [x_dim+d_dim, z_dim]
    encoder1_struct.activation = [activation]
    
    encoder2_struct=nn.NN_struct()
    encoder2_struct.layer_dim = [z_dim+y_dim, a_dim]
    encoder2_struct.activation = [activation]
    
    encoder3_struct=nn.NN_struct()
    encoder3_struct.layer_dim = [z_dim, y_dim]
    encoder3_struct.activation = [T.nnet.softmax]
    
    decoder1_struct=nn.NN_struct()
    decoder1_struct.layer_dim = [z_dim+d_dim, x_dim]
    decoder1_struct.activation = [activation]
    
    decoder2_struct=nn.NN_struct()
    decoder2_struct.layer_dim = [a_dim+y_dim, z_dim]
    decoder2_struct.activation = [activation]        
    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    
    # allocate symbolic variables for the data
    #index_source = T.lscalar()  # index to a [mini]batch
    #index_target = T.lscalar()  # index to a [mini]batch
    index = T.lscalar()  # index to a [mini]batch
    x_source = T.matrix('x_source')  # the data is presented as rasterized images
    y_source = T.matrix('y_source')  # the labels are presented as signal vector 
    x_target = T.matrix('x_target')  # the data is presented as rasterized images
    y_target = T.matrix('y_target')  # the labels are presented as signal vector    
    
    rng = np.random.RandomState(1234)
        
    # construct the DAVAE class
    if s == 0 :
        classifier = VFAE.VFAE(
            rng=rng,
            input_source = x_source,
            input_target = x_target,
            label_source = y_source,
            batch_size = batch_size,
            encoder1_struct = encoder1_struct,
            encoder2_struct = encoder2_struct,
            encoder3_struct = encoder3_struct,
            decoder1_struct = decoder1_struct,
            decoder2_struct = decoder2_struct,
            alpha = alpha,
            beta = beta,
            D = D
        )    
    else :
        classifier = VFAE.Supervised_VFAE(
            rng=rng,
            input_source = x_source,
            input_target = x_target,
            label_source = y_source,
            label_target = y_target,
            batch_size = batch_size,
            encoder1_struct = encoder1_struct,
            encoder2_struct = encoder2_struct,
            encoder3_struct = encoder3_struct,
            decoder1_struct = decoder1_struct,
            decoder2_struct = decoder2_struct,
            alpha = alpha,
            beta = beta,
            D = D
        )    

    
    cost = (classifier.cost)
        
    gparams = [T.grad(cost, param) for param in classifier.params]
                   
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    Output_test_model = theano.function(
        inputs=[index],
        outputs=classifier.params+classifier.outputs+gparams,
        givens={
            x_source: train_ftd_source[index * batch_size : (index + 1) * batch_size, :],
            y_source: train_labeld_source[index * batch_size : (index + 1) * batch_size, :],
            x_target: train_ftd_target[index * batch_size : (index + 1) * batch_size, :]
            #y_target: train_labeld_target[index * batch_size : (index + 1) * batch_size, :]            
        }       
    )     
    
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        givens={
            x_source: test_ftd_source[index * batch_size : (index + 1) * batch_size, :],
            y_source: test_labeld_source[index * batch_size : (index + 1) * batch_size, :],
            x_target: test_ftd_target[index * batch_size : (index + 1) * batch_size, :],
            y_target: test_labeld_target[index * batch_size : (index + 1) * batch_size, :]
        }        
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        givens={
            x_source: valid_ftd_source[index * batch_size : (index + 1) * batch_size, :],
            y_source: valid_labeld_source[index * batch_size : (index + 1) * batch_size, :],
            x_target: valid_ftd_target[index * batch_size : (index + 1) * batch_size, :],
            y_target: valid_labeld_target[index * batch_size : (index + 1) * batch_size, :]
        }        
    )                
    
    validate_bytraindata_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        givens={
            x_source: train_ftd_source[index * batch_size : (index + 1) * batch_size, :],
            y_source: train_labeld_source[index * batch_size : (index + 1) * batch_size, :],
            x_target: train_ftd_target[index * batch_size : (index + 1) * batch_size, :],
            y_target: train_labeld_target[index * batch_size : (index + 1) * batch_size, :]            
        }       
    )     
    
    train_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.source_errors(y_source), classifier.target_errors(y_target), 
                 classifier.source_predict_raw(), classifier.target_predict_raw()],
        updates=updates,
        givens={
            x_source: train_ftd_source[index * batch_size : (index + 1) * batch_size, :],
            y_source: train_labeld_source[index * batch_size : (index + 1) * batch_size, :],
            x_target: train_ftd_target[index * batch_size : (index + 1) * batch_size, :],
            y_target: train_labeld_target[index * batch_size : (index + 1) * batch_size, :]            
        }       
    )                   
    
    ###############
    # TRAIN MODEL #
    ###############
    '''
    Define :
        xx_loss : Cost function value
        xx_score : Classification accuracy rate
        
    '''        
    
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_valid_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    
    validation_frequency = n_train_batches
    
    best_iter = 0
    best_train_loss = np.inf
    best_validation_loss = np.inf  
    test_loss = np.inf
    train_score = 0.
    validation_score = 0.
    test_score = 0.    
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)[0]  
                        
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index                   
                        
            if (iter + 1) % validation_frequency == 0:
                # compute loss on all training set
                train_losses = [validate_bytraindata_model(i)[0] for i in range(n_train_batches)]
                this_train_loss = np.mean(train_losses)
                
                # compute loss on validation set
                validation_losses = [validate_model(i)[0] for i in range(n_valid_batches)]  
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, training loss %f, validation loss %f ' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_train_loss,
                        this_validation_loss
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    train_loss = this_train_loss
                    best_validation_loss = this_validation_loss                    
                    best_iter = iter
                                        
                    #Get Accuracy
                    train_losses = [validate_bytraindata_model(i)[1]for i in range(n_train_batches)]
                    train_score_S = 1 - np.mean(train_losses)
                    train_losses = [validate_bytraindata_model(i)[2]for i in range(n_train_batches)]
                    train_score_T = 1 - np.mean(train_losses)
                    
                    validation_losses = [validate_model(i)[1] for i in range(n_valid_batches)]  
                    validation_score_S = 1 - np.mean(validation_losses)
                    validation_losses = [validate_model(i)[2] for i in range(n_valid_batches)]  
                    validation_score_T = 1 - np.mean(validation_losses)
                    
                    # test it on the test set
                    test_losses = [test_model(i)[1]for i in range(n_test_batches)]
                    test_score_S = 1 - np.mean(test_losses)
                    test_losses = [test_model(i)[2]for i in range(n_test_batches)]
                    test_score_T = 1 - np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test accuracy of '
                           'best model: source domain :%f %%, target domain %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score_S * 100., test_score_T * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation loss of %f '
           'obtained at iteration %i, with train loss %f \n'
           'train accuracy : source domain %f %%, target domain  %f %%\n'
           'validation accuracy : source domain %f %%, target domain  %f %%\n'
           'test accuracy : source domain %f %%, target domain  %f %%') %
          (best_validation_loss, best_iter + 1, train_loss, train_score_S * 100., train_score_T * 100.,
           validation_score_S * 100., validation_score_T * 100., test_score_S * 100., test_score_T * 100.))
    '''
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    '''
    
    #Return Trained Parameter
    '''Model Construct'''
if __name__ == '__main__':    
    object_reconition_test(0)