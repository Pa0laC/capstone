import sys
import os
sys.path.append('../static')
sys.path.append('../../../data/dog_images/train')
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import keras.utils as image
from keras.callbacks import ModelCheckpoint  

from glob import glob
from tqdm import tqdm
import cv2   
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import image as mpimg
from extract_bottleneck_features import *

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    '''
    Given an image at img_path, detect if there is a face
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    '''
    Takes a numpy array of string-valued image paths as input
    Returns a 4D tensor with shape (nb_samples,224,224,3).
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    '''
    Takes a numpy array of string-valued image paths as input
    Returns a 4D tensor with shape (nb_samples,224,224,3).
    '''
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    '''
    Returns an array whose 𝑖-th entry is the model's predicted probability that the image belongs to the 𝑖-th ImageNet category
    '''
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    '''
    Given an image at img_path, detect if there is a dog
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def InceptionV3_predict_breed(img_path, trained_model):
    '''
    Given an image path, run the image through the InceptionV3 model and return the predicted breed
    '''
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = trained_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    
    dog_names = [item[20:-1] for item in sorted(glob("../../data/dog_images/train/*/"))]
    dog_names = dog_names[np.argmax(predicted_vector)]
    
    return dog_names

def dog_resemblance(img_path, trained_model):
    '''
    Given an image,
    Returns predicted breed if a dog is detected + images of dog that resembles them
    Returns resembling dog breed if a human is detected + images of dog that resembles them
    Returns an error if neither is detected.
    '''
    if dog_detector(img_path) :
        human=0
        output = InceptionV3_predict_breed(img_path, trained_model)
        
        # Find path of example image of output dog
        abs_path = os.getcwd()
        dog_dir = output[8:]
        
        # Clean output
        output = output[12:]
        return human, dog_dir, output.replace('_', ' ')
    
    if face_detector(img_path):
        human = 1
        output = InceptionV3_predict_breed(img_path, trained_model)
        
        # Find path of example image of output dog
        abs_path = os.getcwd()
        dog_dir = output[8:]
        
        # Clean output
        output = output[12:].replace('_', ' ')
        return human, dog_dir, output
    else:
        return -1, '','Neither a human nor dog'
    
def get_image_dog_resemblance(path, image_no):
    '''
    Displays image_no number of images from path directory
    '''
    dir = os.listdir( path )

    for file in dir[:image_no]:

        image = mpimg.imread(path+file)
        plt.imshow(image)
        plt.show()