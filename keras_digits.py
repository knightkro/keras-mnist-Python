
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:25:01 2017

@author: KnightG
"""

# using keras on the MNIST dataset

# Import necessary modules
# Import necessary modules
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.callbacks  import EarlyStopping # checks if the updates are improving the model
from keras.optimizers import SGD # stochastic gradient descent optimiser 
from keras.layers     import Dense #type of connection between nodes (to everything)
from keras.models     import Sequential # type of model
from keras.utils      import to_categorical # allows us to create categorical variables
from keras.models     import load_model # allows us to load saved models

###############################################################################
# functions
def get_sequential_model(input_nodes, first_layer, hidden_layer, output_shape, activation_input, activation_hidden, activation_output):
 """ Define a dense sequential neural network with input_nodes number of input nodes,
 hidden_layer a list of hidden layers, output_shape defines output shape. In addition
 the types of activation need to be defined. """
 model = Sequential() # set up model
 model.add(Dense(first_layer, activation = activation_input, input_shape = input_nodes)) # first layer
 for layer in hidden_layer: # set up further hidden layers
     model.add(Dense(layer, activation = activation_hidden))
 model.add(Dense(output_shape, activation = activation_output)) 
 return model

###############################################################################                 


# Load the MNIST data
df_train = pd.read_csv('train.csv') 
df_test = pd.read_csv('test.csv') 
predictors = df_train.drop(['label'], axis=1).as_matrix()
#predictors_2 = (df_train.drop(['label'], axis=1) >= 128).as_matrix() # change to black and white
target = to_categorical(df_train['label'])

# extract shape of the predictors
n_cols = predictors.shape[1]

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the models: model
model_1 = get_sequential_model((n_cols,), 50, [50,50], 10, 'relu', 'relu', 'softmax')
model_2 = get_sequential_model((n_cols,), 20, [20,20,20], 10, 'relu', 'relu', 'softmax')
# Compile the model
model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])


# Fit the model
model_1_training = model_1.fit(predictors,target, epochs=30, validation_split = 0.2, callbacks=[early_stopping_monitor])
model_2_training = model_2.fit(predictors,target, epochs=30, validation_split = 0.2, callbacks=[early_stopping_monitor])

# Create the plot
plt.plot(model_1_training.history['val_acc'], 'r', model_2_training.history['val_acc'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

"""
# save the model to file
#model_1.save('model_1.h5')
#load model
load_model = load_model('model_1.h5')
#Predict
preds = load_model.predict_classes(df_test.as_matrix())

#save to file
solution=pd.DataFrame({"ImageID":np.arange(1,28001,1),"Label":preds})
solution.to_csv("solution.csv", index=False)
"""