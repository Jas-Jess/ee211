# Code for Hand Gesture Recognition using Various Convolutional Neural Network Models
UCLA EE211A Project by Jasmine Moreno & Jessica Fu

## Introduction to the Code
The topic of Human Computer Interaction (HCI) studies and develops computer design in relation to the interactions between humans and computer technology. Gesture recognition technologies are represented in items such as video games and smart technologies.

Most applications require a two-dimensional plus depth camera. The project creates a hand gesture recognition system using two-dimensional static images, without needing a depth map. We implement different convolutional neural network (CNN) models to classify each hand gesture according to American Sign Language (ASL). To increase the accuracy of recognizing the ASL letter, we pass a training set of data through multiple models. The best model is evaluated based on accuracy, loss, recall, and precision.

We implemented our own hand detection. However, our initial data set was too small to run on our CNN models. Therefore, we had to look for another data set to use in the CNN models. 

## Github Repo
This github repo includes our CNN code & hand dection code. We have the following folders/files:
1. __Confusion matrix:___ This folder has the confusion matrix for our best model (Multiscale4)
    
    a. __confusion_matrix.xlsx__ excel version of the confusion matrix
    
    b. __ConfusionMatrix.png__ Image version of the confusion matrix. 

2. __Data:__: Dataset used for CNN models
    
    a. __amer_sign3.png__ images example of dataset
    
    b. __sign_mnist_test.csv__ testing data set with 7,172 images (flatten) with labels
    
    c. __sign_mnist_train.csv__ training data set with 27,455 images (flatten) with labels

3. __Dectection Example:__ image examples of our dectection algorithm. The images used our from our intial data set (look under Data Set section for link).

4. __mat2tiles:__ Folder with matrix-spliting function [https://www.mathworks.com/matlabcentral/fileexchange/35085-mat2tiles--divide-array-into-equal-sized-sub-arrays] that is used in the hand dectection
    
    a. __license__ license to use this function
    
    b. __mat2tiles.m__ Function to split matrix to num of sections needed

5. __Models:__ After running all the various CNN models, we save the models in this folder to use in the future. Each folder inside corresponds to the model talked about in the report. 

6. __cnn_models.py__ Heading files that has the CNN model function definition.

7. __confusion_matrix.py__ Code to save the confusion matrix of the best model to a PNG or Excel file. 

8. __mnistDataset.py__ Creates a DataSet class for the training and testing data. 

9. __model_evaluation.py__ Trains and tests the CNN of choosing. 

10. __preprocessing_images.m__ MATLAB code to detect hand gestures, crop the hand, greyscale, and resize to 28x28 image. 

## Data Set
### Sign Language MNIST for CNN model
Data set is already downloaded to this repo. 

For our CNN models, we use an American Sign Language data set called “Sign Language MNIST” [https://www.kaggle.com/datamunge/sign-language-mnist/data]. This data set provides 27455 images of training data and 7172 images of testing data. This data set is relatively small compared to other research works (80k+ of data). The data includes grayscale 28x28 images (Figure 1) of hands signing each letter of the alphabet, excluding “J” and “Z” which require motion.

### Initial Data Set used for Hand Dectection
Download this data set at http://sun.aei.polsl.pl/~mkawulok/gestures/

This data set provides a collection of photos for the American Sign Language and the Polish Sign Language. We used the 'hgr2B' data set. This set has a total of 341 American Sign Language. Our detection worked fairly well except on a few complex background images. 

## Libraries Used
The libraries used in this code are: 
* Numpy 1.14.1
* CSV 1.0
* TensorFlow 1.6.0
* Pandas 0.22.0
* Matplotlib 2.2.0
* Sklearn 0.19.1