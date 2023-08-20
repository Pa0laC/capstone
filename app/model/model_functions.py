from keras.callbacks import ModelCheckpoint
import sys
sys.path.append('../..')
import numpy as np
from extract_bottleneck_features import *
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import load_files
from keras.utils import to_categorical
from glob import glob
import joblib

def load_model_data(data_filepath):
    '''
    Load bottleneck features for model, and train, test, valid data
    '''
    bottleneck_features = np.load(data_filepath)
    train_InceptionV3 = bottleneck_features['train']
    valid_InceptionV3 = bottleneck_features['valid']
    test_InceptionV3 = bottleneck_features['test']
    return train_InceptionV3, valid_InceptionV3, test_InceptionV3 

# define function to load train, test, and validation datasets

def load_dataset(path):
    '''
    Load dataset from path and return dog files and target files
    '''
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def build_model(input_shape):
    '''
    Build model and split data into training and test dataset
    '''
    
    # Define model
    print('...Definind CNN architecture')
    InceptionV3_model = Sequential()
    InceptionV3_model.add(GlobalAveragePooling2D(input_shape=input_shape))
    InceptionV3_model.add(Dense(133, activation='softmax'))

    print(InceptionV3_model.summary())
    InceptionV3_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return InceptionV3_model
    
def save_model(model, model_filepath):
    '''
    Save model in file for later use
    '''
    # save the model as a pickle file
    joblib.dump(model, open(model_filepath, 'wb'))
    
    
def main ():
    '''
    Define and train model
    Output: None, model saved as pickle
    '''
    
    # Load datasets
    train_files, train_targets = load_dataset('../../data/dog_images/train/')
    valid_files, valid_targets = load_dataset('../../data/dog_images/valid/')
    test_files, test_targets = load_dataset('../../data/dog_images/test/')
    train_InceptionV3, valid_InceptionV3, test_InceptionV3 = load_model_data('../bottleneck_features/DogInceptionV3Data.npz')
    
    # Build model
    InceptionV3_model = build_model(input_shape=train_InceptionV3.shape[1:])
    
    # Train model
    print('...Training model')
    checkpointer = ModelCheckpoint(filepath='../saved_models/weights.best.InceptionV3.hdf5', 
                                   verbose=1, save_best_only=True)

    InceptionV3_model.fit(train_InceptionV3, train_targets, 
              validation_data=(valid_InceptionV3, valid_targets),
              epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
    
    InceptionV3_model.load_weights('../saved_models/weights.best.InceptionV3.hdf5')
    
    # Save model for later use
    print('...Saving model')
    save_model(InceptionV3_model, 'model.pkl')
    
if __name__ == '__main__':
    main()