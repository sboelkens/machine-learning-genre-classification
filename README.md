# machine-learning-genre-classification

This repository uses Convolutional Neural Networks and melspectograms to train a music genre classicifaction model in 
Python as part of a Machine Learning course for the Master's programme at the University of Groningen in 2017/2018. 
This was done using the GTZAN dataset as training, test and validation data in order to train a model.<br />
There are three folders, **test** is used for the testing of features in order to create the final program, 
**Related ML Articles** is used for the gathering of interesting articles to help us understand more about CNN and 
**genre-classification** is where our main program is located. See the section **Programs** for more about this.

### Dependencies
* Keras
* Tensorflow
* Librosa
* Numpy
* Scikit-learn
* Scipy
* h5py

### Prerequisites
To be able to run the program, you first have to download the GTZAN folder and add it to this repository. 
It was intentionally left out and ignored by Git because it is larger then a Gigabyte of data.<br />
You must place it inside **genre-classification** in a folder named **gtzan** in order for the program to find it. <br />
The resulting folder structure looks like this:
* genre-classification/
    * gtzan/
        * blues/
            * blues.00000.au
            * blues.00001.au
            * ...
        * classical/
        * ...

### Programs
Here each program is listed, together with its functionality.

##### training.py 
This is the main part, this is the program that trains a genre classification model. This is all done in the **_main_**
function, which is called with the right parameters. If run without any parameters, it will use some standard parameters
which already will give a good result.

##### keras_models.py
This is where the Keras model is created. There are four different functions which create four different models, 
and there is one folder which creates a model based on the parameters included.

##### plot_creator.py
This was written to be able to plot different created models into one figure for a better overview.

##### Other
The other programs were helper programs not used in the main training program.

### Garageband dataset
We also used another dataset, from garageband.com, made by people at the 
[University of Dortmund](http://www-ai.cs.uni-dortmund.de/audio.html). 
We altered this slightly to work with our program, unfortunately it was not possible to upload to Github because of its sheer size (255 MB). 