"""
This script will calculate the F1 score and accuracy over the whole training set
This will also calculate the best threshold to maximize F1 score
"""
import numpy as np
import os
import random
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import pickle


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping
from keras import backend



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




def load_preprocess_training_batch(batch_id):
    """
    Load the Preprocessed Training data and return them in arrays of features and labels
    """
    filename = 'train_batch.{}.p'.format(batch_id)
    filepath = os.path.join('./train_pickle', filename)
    print filepath
    features, labels = pickle.load(open(filepath, mode='rb'))
    print 'feature len=', len(features), "labels len=", len(labels)
    
    # Return the training data in arrays of features and labels
    return np.array(features), np.array(labels)




def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')



def find_f2score_threshold(p_valid, y_valid, verbose=True):
    
    print p_valid.shape, y_valid.shape
    best = 0
    best_score = -1
    totry = np.arange(0,1,0.1)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True: 
        print('Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best


def validation_batch(start, end, num_folds = 10):
    
    input_shape=(64, 64, 3)
    
    print '######## valid  Predict ########## start = ', start, ' end = ', end, " ###############"
    
    x_test = []
    y_valid = []
    for batch_i in range(start, end):
        features, labels  = load_preprocess_training_batch(batch_i)             
        x_test.extend(features)
        y_valid.extend(labels)
        
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
    
    return y_valid, results


y_valid, p_valid = validation_batch(0, 41, 10)
find_f2score_threshold(np.array(p_valid), np.array(y_valid))
print 'accuracy = ', accuracy_score(np.array(y_valid), np.array(p_valid > 0.2))
    
