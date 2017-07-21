"""
This script will load test data and predict output against the derived model
It will load the best weights of all KFold iterations and average the output
It will store the labels and the image filenames in a pickle file
"""
import numpy as np
import os
import random
from sklearn.metrics import fbeta_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.callbacks import ModelCheckpoint
import pickle


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras import backend


# this is the sequential model as defined with Keras layers
# we have convolution filters - 32, 64, 128 and 256 followed by maxpool and dropout
# This is followed by Dense layer of 256 and final classification layer of 17 
# each layer  has RELU activation except last sigmoid activation for classification

def define_model(input_shape=(64, 64, 3)):
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))
    
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))        
    
    return model


# This routine loads a batch of test images from pickle file into memory
def load_preprocess_test_batch(batch_id):
    """
     Load the Preprocessed Test data and return them in arrays of features and corresponding filename
    """
    filename = 'test_batch.{}.p'.format(batch_id)
    print filename
    filenames, test_features = pickle.load(open(filename, mode='rb'))
    print 'test_features len=', len(test_features), "filenames len=", len(filenames)
    
    # Return the array of test  features and corresponding filename
    return np.array(filenames), np.array(test_features)


 
# This  routine loads result pickle file into memory, 
# loads the best weights for model and predicts teh results.
def test_results(start, end, num_folds=10):
    
    input_shape=(64, 64, 3)
    
    print '######## test batch Predict ########## start = ', start, ' end = ', end, " ###############"
    
    x_test = []
    filepaths = [] 
    for batch_i in range(start, end):
        filenames, x_test_batch = load_preprocess_test_batch(batch_i)             
        x_test.extend(x_test_batch)
        filepaths.extend(filenames)
        
    x_test = np.array(x_test)
    
    model = define_model(input_shape)
    
    y_pred = []
    for num_fold in range(num_folds):
        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        model.load_weights(kfold_weights_path)
        
        print("Weights loaded for kFold = ", num_fold)
        y_pred.append(model.predict(x_test))
    
    results = sum (y_pred)
    
    results /= float(num_folds)
    
    pickle_filename = 'result.{}.{}.p'.format(start, end)                             
    print 'pickle_filename = ', pickle_filename   
    pickle.dump((filepaths, results), open(pickle_filename, 'wb'))
                

# start of script
test_results (0, 62, num_folds=10)                            