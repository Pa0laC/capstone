from flask import Flask, render_template, request, flash
import joblib
from fileinput import filename
import os
import cv2
import sys
sys.path.append('..\..\dog-project')
sys.path.append('..\..\data\dog-images\train')
import shutil

from werkzeug.utils import secure_filename
from model.predictor_functions import dog_resemblance

UPLOAD_FOLDER = 'static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# load model
InceptionV3_model = joblib.load("model/model.pkl")

@app.route('/')
@app.route('/index')
def home():
    return render_template('home.html')


# web page that handles user query and displays model results
@app.route('/success', methods = ['POST'])  
def process_input(): 
    file = request.files['file']
    
    if file:
        # If file inputted, add into uploads folder and analyse dog breed
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)


        human, dog_dir, model_output = dog_resemblance(img_path, InceptionV3_model)
        
        # If image has a dog or human detected
        if human !=-1:
            
            abs_path = os.getcwd()
            # Path to folder with example images of predicted dog
            dog_dir_path =os.path.join(abs_path, '../../data/dog_images/train/', dog_dir)
            dir = os.listdir(dog_dir_path)
            
            # First image of directory that can be used as an example of the predicted dog
            example_image_name = dir[1]
            
            # Location of that image
            source = os.path.join(dog_dir_path, example_image_name)
            
            # Image can be moved so it is accessible by front end of app
            destination =   os.path.join(abs_path, UPLOAD_FOLDER)
            shutil.copy2(source, destination)
            
            # New path of image, to be used in front end of app
            img_dog = os.path.join(app.config['UPLOAD_FOLDER'],example_image_name)        

            # This will render the go.html Please see that file. 
            return render_template(
                'go.html',
                img_path = img_path,
                img_dog = img_dog,
                human=human,
                model_output=model_output)
        
        # If neither a dog or human, nothing to be displayed
        if human ==-1:
            return render_template(
                'go.html',
                human=human,
                img_path = img_path,
                )


if __name__ == '__main__':
    app.run(debug=True)