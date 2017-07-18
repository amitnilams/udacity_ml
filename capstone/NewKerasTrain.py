"""
This script will run training over KFold training and validation set
This will loop over various epoch and learning rate
It will checkpoint the best model based on the 'val_loss' criterion
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


def _fbeta_score(model, X_valid, y_valid):
    
    p_valid = model.predict(X_valid)
    
    #print y_valid.shape
    #print y_valid
    
    p_arr = []
    for p in p_valid:
        p_arr.append(map (lambda x: 1.0 if x > 0.2 else 0.0, p))
        
    p_arr = np.array(p_arr)
    
    #print p_arr.shape
    #print p_arr
    
    score = fbeta_score(y_valid, p_arr, beta=2, average='samples')
    
    #print 'score = ', score
    
    return score





def load_preprocess_training_batch(batch_id):
    """
     Load the Preprocessed Training data and return them in arrays of features and labels
    """
    filename = 'train_batch.{}.p'.format(batch_id)
    print filename
    features, labels = pickle.load(open(filename, mode='rb'))
    print 'feature len=', len(features), "labels len=", len(labels)

    # Return the training data in arrays of features and labels
    return np.array(features), np.array(labels)



def run_KFold(X_train, Y_train, X_valid, Y_valid, num_fold):
    
    print '######## run kFold, num_fold = ', num_fold, '##########'
  
    input_shape=(64, 64, 3)
    
    kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
    
    model = define_model(input_shape)
          
    epochs_arr = [20, 5, 5]
    learn_rates = [0.001, 0.0001, 0.00001]
     
    checkpoint = ModelCheckpoint(kfold_weights_path, monitor='val_loss', verbose=1, save_best_only=True)
       
    for learning_rate, num_epochs in zip(learn_rates, epochs_arr):
        opt = Adam(lr=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print '#####learning_rate = ', learning_rate, ' num_epochs = ', num_epochs, " ###########"
        model.fit(X_train, Y_train,
                  batch_size=128,
                  epochs=num_epochs,
                  verbose=1,
                  validation_data=(X_valid, Y_valid),
                  callbacks=[checkpoint])
        
    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    score = _fbeta_score(model, X_valid, Y_valid)
    print '########## fbeta_Score = ', score
        
    backend.clear_session()
        
                                           

# execute above function in a loop  below                  
def run_training_kfold(start, end, nfolds):
    
    x_train = []
    y_train = []
    for batch_i in range(start, end):
        x, y = load_preprocess_training_batch(batch_i)
        x_train.extend(x)
        y_train.extend(y)
        
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)
    
    num_fold = 0
    
    for train_index, test_index in kf:       
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]
        
        
        print('KFold number {} of  {}'.format(num_fold, nfolds))
        print('Split train: X_train/Y_train', len(X_train), len(Y_train))
        print('Split valid: X_Valid/Y_valid', len(X_valid), len(Y_valid))
       
        run_KFold(X_train, Y_train, X_valid, Y_valid, num_fold)      
        num_fold += 1

run_training_kfold(0, 41, nfolds = 10)
                            

