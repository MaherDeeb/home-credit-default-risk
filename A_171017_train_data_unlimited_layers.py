# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 18:11:14 2017

@author: Maher Deeb
"""
import time
import pandas as pd
import numpy as np
from fit_mode_Tensorflow_mini_batche_unlimited_layers import *
import tensorflow as tf
from tensorflow.python.framework import ops
#from map_features import *
from sklearn.model_selection import train_test_split
import datetime, time

#==============================================================================
# building the model that combines all important function together
#==============================================================================
def model(layers_dims,learning_rate = 0.000001,num_epochs = 1000, minibatch_size = 9000, print_cost = True):
    #262144
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    #with h5py.File('data_preprocessing.h5', 'r') as hf:
        #X_train = hf['X_train'][:]
        #Y_train = hf['y_train'][:]  

    X_train = pd.read_csv('x_train_sr0.csv')
    y_train = pd.read_csv('y_train_sr0.csv',header = None)
    x_test = pd.read_csv('x_cv_sr0.csv')
    y_test = pd.read_csv('y_cv_sr0.csv',header = None)
    
    col_consider = X_train.columns[1:111]
    
    X_train = X_train[col_consider]
    x_test = x_test[col_consider]
    
    
# =============================================================================
#     col_to_delete = list(X_train.columns[std_0==0])
#     
#     for col_i in col_to_delete:
#         
#         X_train = X_train.drop([col_i],axis=1)
#         x_test = x_test.drop([col_i],axis=1)
# =============================================================================
        
# =============================================================================
#     X_train = X_train.drop(['SK_ID_CURR'],axis=1)
#     x_test = x_test.drop(['SK_ID_CURR'],axis=1)
# =============================================================================



    X_train_norm = (X_train - X_train.mean())/(X_train.std())
    mu_train =X_train.mean()
    std_train = X_train.std()
    

    
    X_cv_norm = (x_test - X_train.mean())/(X_train.std())
    
    X_train_norm = X_train.fillna(0)

    X_cv_norm = x_test.fillna(0)                                               
    
    X_train = np.array(X_train_norm)
    X_cv = np.array(X_cv_norm)
    Y_train = np.array(y_train[1])
    y_cv = np.array(y_test[1])
# =============================================================================
#     mu_i=np.mean(X_train,axis=0)
#     std_i=np.std(X_train,axis=0)
#     for i in range(len(X_train[0])):
#         X_train[:,i]=(X_train[:,i]-mu_i[i])/std_i[i]
# =============================================================================
        
    X_train=X_train.T
    #print(type(Y_train))
    Y_train=np.reshape(Y_train,(1,len(Y_train)))
    
    #layers_dims = [np.shape(X_train)[0],10,1] #  4-layer model  
    layers_dims[0] = np.shape(X_train)[0]
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]                            # n_y : output size
    
    # Create Placeholders of shape (n_x, n_y)

    X, Y = create_placeholders(n_x, n_y)


    # Initialize parameters

    parameters = initialize_parameters(layers_dims)

    
    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z3 = forward_propagation(X, parameters,layers_dims)

    y_hat_1=tf.sigmoid(Z3)
    correct_prediction1 = tf.nn.l2_loss(tf.cast(y_hat_1, "float")- Y)
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
    # Cost function: Add cost function to tensorflow graph

    beta=0.00001
    
    #cost = compute_cost(Z3, Y)
    #print(parameters['W1'])
    
    regularizer = tf.nn.l2_loss(parameters['W1'])
    for ii in range(2,len(layers_dims)):
        regularizer += tf.nn.l2_loss(parameters['W'+str(ii)])

    cost = tf.reduce_mean(compute_cost(Z3[0:n_y], Y) + beta * regularizer)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    #learning_rate_final = tf.train.exponential_decay(starter_learning_rate, global_step,
     #                                      500, 0.96, staircase=True)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

    optimizer = tf.train.AdamOptimizer(learning_rate = starter_learning_rate).minimize(cost, global_step=global_step)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            
            start = time.time()
            
            epoch_cost = 0.                       # Defines a cost related to an epoch
            #num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            #for minibatch in minibatches:

                # Select a minibatch
                #(minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                #_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

                
                #epoch_cost += minibatch_cost / num_minibatches

            end = time.time()
            #print(end - start)
            #print (epoch)
            # Print the cost every epoch
# =============================================================================
            if print_cost == True and epoch % 10 == 0:
                
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                
               # with h5py.File('data_preprocessing.h5', 'r') as hf:
                    #X_test = hf['X_test'][:]
                    #Y_test = hf['y_test'][:]
                X_test =X_cv
                Y_test =y_cv
# =============================================================================
#                 for i in range(len(X_test[0])):
#                     X_test[:,i]=(X_test[:,i]-mu_i[i])/std_i[i]
# =============================================================================
                X_test=X_test.T
                Y_test=np.reshape(Y_test,(1,len(Y_test)))
                
                parameter1 = sess.run(parameters)    
                z_test = sess.run(forward_propagation(tf.cast(X_test, "float"), parameter1,layers_dims))
                y_test_NN= sess.run(tf.sigmoid(tf.cast(z_test, "float")))
                y_test_NN=y_test_NN.T
                #pd.DataFrame(y_test_NN).to_csv('y_test_NN.csv', index=False, float_format = '%.5f')   
                
                #current_CV_cost=sess.run(accuracy1, feed_dict={X: X_test, Y: Y_test})
                current_CV_cost= np.sum((Y_test-y_test_NN.T)**2)/len(Y_test[0])
                
                #print(y_test_NN)
                #print(Y_test)
                #wait = input("PRESS ENTER TO CONTINUE.")
                
                del X_test,Y_test
                
                print (' error cv: ', current_CV_cost)
                
                    
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        #y_hat=tf.sigmoid(Z3)
        #y_hat=(y_hat>0.5)
        # Calculate the correct predictions
        #correct_prediction = tf.equal(tf.cast(y_hat, "float"), Y)
        # Calculate accuracy on the test set
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters,layers_dims,col_consider,mu_train,std_train
#==============================================================================
# Run the tesnsorflow model    
#==============================================================================

layers_dims =  [145,50,25,1]

parameters,layers_dims,col_consider,mu_train,std_train = model(layers_dims,num_epochs = 10,minibatch_size = 18900)


# =============================================================================
# def predict_state0(parameters,X_test):
#     
#     with tf.Session() as sess:
#         
#         z_test = sess.run(forward_propagation(tf.cast(X_test, "float"), parameters,layers_dims))
#         y_test_NN= sess.run(tf.sigmoid(tf.cast(z_test, "float")))
#         
#         return y_test_NN
#  
# =============================================================================

 
 
def model_pred(parameters,layers_dims,col_consider,mu_train,std_train):
    #X_train = pd.read_csv('X_train.csv')
    X_sub = pd.read_csv('df_test_decoded.csv',encoding='iso-8859-1')
    X_sub = X_sub[col_consider]
    X_sub_norm = (X_sub-mu_train)/std_train
    X_sub_norm = X_sub_norm.fillna(0)

    X_sub = np.array(X_sub_norm)
    X_sub = X_sub.T
    
    df_id_submit = pd.read_csv('SK_ID_CURR_grouped.csv',encoding='iso-8859-1',header=None)
    with tf.Session() as sess:
        
        z_test = sess.run(forward_propagation(tf.cast(X_sub, "float"), parameters,layers_dims))
        y= sess.run(tf.sigmoid(tf.cast(z_test, "float")))
    
    Y_submit=pd.DataFrame()
    Y_submit = pd.concat([df_id_submit[1], pd.DataFrame(y.T)], axis=1)
    # =============================================================================
    Y_submit.columns=['SK_ID_CURR','TARGET']
    df_best_submit = pd.read_csv('1529870832_submit.csv')
    Y_submit['SK_ID_CURR'] = df_best_submit.SK_ID_CURR
    
    Y_submit.to_csv('{}_submit.csv'.format(str(round(time.mktime((datetime.datetime.now().timetuple()))))),index=False)
    # =============================================================================
    return Y_submit


Y_submit = model_pred(parameters,layers_dims,col_consider,mu_train,std_train)