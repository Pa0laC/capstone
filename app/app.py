from flask import Flask, render_template, request, flash
import joblib
from fileinput import filename
import os
import sys
sys.path.append('..')

from werkzeug.utils import secure_filename
from model.predictor_functions import dog_resemblance

UPLOAD_FOLDER = 'uploads'

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
        # if user does not select file, browser also
        # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        
        
    model_output = dog_resemblance(InceptionV3_model,'../'+img_path)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        file = filename,
        model_output=model_output)



if __name__ == '__main__':
    app.run(debug=True)