### Table of Contents

1. [Project Motivation](#motivation)
2. [Instructions](#Instructions)
3. [Repository structure](#structure)
4. [Analysis](#analysis)
5. [Conclusion](#conclusion)
6. [Acknowledgments](#ack)

## Project Motivation <a name="motivation"></a>

The primary aim of this project was to learn how to use a neural network to build a classifier. However, this project was also an opportunity to use Flask for the first time, and build a web application that displays the output of the classifier.

The classifier built was trained to recognize a human or a dog on an image, and predict the dog breed it most resembles. To do this, we first tested a neural network built from scratch before deploying a more complex Convolutional Neural Networks (CNN) that uses transfer learning. The pre-trained model used was an Inception V3 mode.

The web application was then deployed onto flask. This is what it looked like and what it produced:
![Sample home](.home.png)

![Sample dog output](.go_dog.png)

![Sample human output](.go.png)

![Sample Bad Output](.go_bad_input.png)


## Instructions <a name="Instructions"></a>
### Setting up your repo (copied from original repo and slightly edited)

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `../data/dog_images`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `../data/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `./bottleneck_features`.

5. Download images you will use to test your algorithm at the location `./images`.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```
	
7. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

8. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

9. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

### Running the webapp

1. First change to the `./app` directory using the terminal: ``` cd app```
2. Run the app in the terminal: ```python app.py```
3. Open up : ```http://127.0.0.1:5000/```

### Additional requirements

You will need to install Tensor Flow and Flask.

## Repository Structure  <a name="structure"></a>
This repository corresponds to the dog-project folder. Its structure is displayed below. The data folder corresponds to the folder with all the dog and human images used for training. It is intentionally placed outside of this repo for storage issues.

-/data/

-/dog-project/

    - app/
    
            - app.py
            
         - model/  
         
                - model.pkl
            
                - model_functions.py
                
                - predictor_functions.py
                
         - static/  
         
                - KanyeWest.jpg
            
                - German_pinscher_04838.jpg
                
                - style.css
         
         
         - templates/
         
         
                - go.html
            
                - home.html
                
         
 
    - bottlecneck_features/
    
    - haarcascades/
    
    - saved_models/
    
    - images/
    
    - dog_app.ipynb
    
    - dog_app.html
    
    - home.jpg
    
    - go.jpg
    
    - go_dog.jpg
    
    - go_bad_input.jpg
    
The empty folders correspond to folders that are present in my home repository but that could not be exported to GitHub as they are too big in size.  Follow the instructions to download the files that should go there.

## Analysis  <a name="analysis"></a>

The model was trained to recognize 133 dog breeds using 6680 images of dogs. A further 835 images were used for testing and validation each. Analysis and results are discussed in ```jupyter notebook dog_app.ipynb```.

## Conclusion <a name="conclusion"></a>

To evaluate our models, we used accuracy as a metric. Indeed, this metric determines how good a classifier is at predicting both positives and negatives. In our case, both positives and negatives were equally important hence our choice.

The first model we built from scratch performed very poorly (with an accuracy of just above 2%), and took a lot of time to train, despite only using 10 epochs. This made us reconsider our approach.

We thus tested transfer learning, which uses a pre-trained model that has already spent a lot of time being fitted on many layers. This was a great way to increase performance to up to 70%.

Finally, we tested our own transfer learning method of choice, increasing the accuracy further to about 80%. This was an excellent improvement.

Nevertheless, a couple of points were identified that could be improved the future. Firstly, we could try augmenting the training set. We can augment it by randomly translating, rotating the dogs in our training dataset.
Moreover, the hyperparemeters of the model could be tuned further. As of now, it takes a while to run the model and it is thus challenging to do this. However, in an ideal case where I had more computing power, this would be something to do.
Finally, we could simply add more data to our training dataset. 133 images is not a lot and there are many other breeds of dogs that exist such as doodles etc.

To conclude, the web application that was deployed, while far from perfect as this was my first time coding in html, was a great way to showcase how this model could be used to interact with users.

 ## Acknowledgments  <a name="ack"></a>
 This repository and the instruction section above was cloned from [GitHub](https://github.com/udacity/dog-project.git).
 Since this was my first time using Flask and using html, the following website had some great tools to help with the layout design: [W3Schools](https://www.w3schools.com/howto/).
 This [StackOverflow](https://stackoverflow.com/questions/44926465/upload-image-in-flask) page was also very useful to learn how to upload an image onto Flask.
 