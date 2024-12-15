# Food-Classification
![Uploading Resized_thumbnail_image.png…]()

# AI PROJECT ON 101 Food Classification :-
# BUSINESS CASE:- Based on the image data we need to predict the 101 food clases. 
## TASK: MULTICLASS CLASSIFICATION :-
## WE DEVICE THIS PROJECT INTO MULTIPLE STEPS :-
* Importing library
* Data Processing [Prepare training and testing data]
* Build Arcitecture
* Model Compilation
* Training
* Evaluation
* Model saving
* Prediction
* Testing
* Visualise Test Images
* Deploy model using flask framework

## DATA SUMMARY :- 

* This dataset has 101000 images in total. It's a food dataset with 101 categories(multiclass).
* images folder contains 101 folders with 1000 images, Each folder contains images of a specific food class.

## LODING DATA / PREPARING DATA :-

* The dataset for this project is loaded and prepared using the OpenCV library, which provides a versatile set of tools for image processing and manipulation. OpenCV is utilized to read and preprocess the food images before feeding them into the Convolutional Neural Network (CNN) model.

## DATA PROCESSING :-



* Images are loaded,their labels using OpenCV and glob from a directory containing subdirectories, and creating a dataset by  resized the images to a consistent size of 128x128 pixels, and converted to grayscale to reduce computational complexity. The dataset is divided into training, validation, and testing sets. Additionally, label encoding is applied to convert class labels into numerical values.
* The training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 128 pixels.

 [PREPARE TRAINING, TESTING, VALIDATION DATA]
* Training: 8080 images belonging to 101 classes.
* Validation: 2020 images belonging to 101 classes.
* Testing: 3030 images belonging to 101 classes.

## BUILD ARCHITECTURE:-
* A Convolutional Neural Network (CNN) architecture is designed for food image classification. The CNN consists of convolutional layers with batch normalization and max-pooling, followed by a flattening layer and fully connected layers. The final layer employs a softmax activation function to output predicted class probabilities.

## MODEL COMPILATION AND TRAINING :-

* The CNN model is compiled using the categorical cross-entropy loss function and the RMSprop optimizer. It is trained using the training data and validated using the validation data. The training history is monitored for training and validation accuracy.

##  MODEL EVALUATION:-
* The trained model is evaluated on both the training and testing datasets to assess its performance. The evaluation results include accuracy and loss values. The model demonstrates strong performance, achieving high accuracy on both training and testing data.
* Training Loss: 0.398312
* Validation Loss: 0.389305
* Training Accuracy: 0.935396 (or 93.54%)
* Validation Accuracy: 0.938284 (or 93.83%)

## VISUALISE  TESTING IMAGES :-
![image](https://github.com/user-attachments/assets/89707ff2-966d-4379-aff7-830bd3c34efb)

![image](https://github.com/user-attachments/assets/1d7d102a-839f-480c-b44d-d1a990c1b08d)

## MODEL SAVING:-
* The trained CNN model is saved using the "h5"

## CONCLUSION:-
* In this project, a Convolutional Neural Network (CNN) was successfully developed and trained for food image classification. The model demonstrated excellent accuracy on both training and testing datasets, indicating its ability to effectively classify food images into their respective categories.

## DEPLOY MODEL USING FLASK FRAMWORK .

## LIBRARY USED:-
* Tensorflow
* Keras
* Matplotlib
* Glob
* Numpy
* Open CV
* OS

## TOOL USED:-
* Jupyter Notebook
