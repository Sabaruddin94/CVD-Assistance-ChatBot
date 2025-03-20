## Classification of Healthy and Unhealthy Artery

This project focuses on building a deep learning model for the classification of healthy and unhealthy arteries using Convolutional Neural Networks (CNNs) and Transfer Learning with the MobileNetV2 architecture. The model is trained to achieve an accuracy of over 80%, enabling it to classify artery images into two categories: Healthy and Unhealthy.

## Machine Learning Workflow
The project follows the typical machine learning workflow, which includes Problem Formulation, Data Preparation, Model Development, and Model Deployment.

## 1. Problem Formulation
The task is formulated as a binary classification problem, where the objective is to classify artery images as either:

-Healthy
-Unhealthy 

This classification task is essential for early detection of cardiovascular diseases, which could significantly help in medical diagnoses.

2. Data Preparation
Data preparation is essential for training an effective model. The following steps were taken:

a. Data Loading:
Dataset: The dataset consists of labeled images of arteries, categorized into Healthy and Unhealthy classes. The dataset is stored in the datasets directory.
Image Size: The images are resized to 600x600 pixels.
Batch Size: A batch size of 32 is used to load images efficiently during training.
b. Data Splitting:
The data is split into training (70%) and validation (30%) sets.
The validation set is further split into validation and test datasets.
c. Data Augmentation:
Data augmentation is applied to improve model generalization by randomly applying transformations to the images, such as:

-Horizontal and vertical flipping
-Random rotations
-Random zoom
-Random contrast and brightness adjustments

d. Data Visualization:
Several images from the dataset are visualized to ensure that the data is correctly loaded and augmented.

3. Model Development
a. Transfer Learning with MobileNetV2:
Base Model: The project utilizes MobileNetV2, a pre-trained model, as a feature extractor. It is frozen to retain its learned weights from ImageNet while only training the classifier for our specific task.
Regularization: L2 regularization and dropout layers are used to reduce overfitting.
Model Architecture: The model consists of:


## Classification of Healthy and Unhealthy Artery

This project focuses on building a deep learning model for the classification of healthy and unhealthy arteries using Convolutional Neural Networks (CNNs) and Transfer Learning with the MobileNetV2 architecture. The model is trained to achieve an accuracy of over 80%, enabling it to classify artery images into two categories: Healthy and Unhealthy.

## Machine Learning Workflow

The project follows the typical machine learning workflow, which includes Problem Formulation, Data Preparation, Model Development, and Model Deployment.

## 1. Problem Formulation
The task is formulated as a binary classification problem, where the objective is to classify artery images as either:

- Healthy
- Unhealthy

This classification task is essential for early detection of cardiovascular diseases, which could significantly help in medical diagnoses.

## 2. Data Preparation
Data preparation is essential for training an effective model. The following steps were taken:

a. Data Loading:
-Dataset: The dataset consists of labeled images of arteries, categorized into Healthy and Unhealthy classes. The dataset is stored in the datasets directory.
-Image Size: The images are resized to 300x300 pixels.
-Batch Size: A batch size of 32 is used to load images efficiently during training.

b. Data Splitting:

The data is split into training (70%) and validation (30%) sets.
The validation set is further split into validation and test datasets.

c. Data Augmentation:

Data augmentation is applied to improve model generalization by randomly applying transformations to the images, such as:

- Horizontal and vertical flipping
- Random rotations
- Random zoom
- Random contrast and brightness adjustments

d. Data Visualization:
Several images from the dataset are visualized to ensure that the data is correctly loaded and augmented.

## 3. Model Development
a. Transfer Learning with MobileNetV2:

Base Model: The project utilizes MobileNetV2, a pre-trained model, as a feature extractor. It is frozen to retain its learned weights
from ImageNet while only training the classifier for our specific task.

Regularization: L2 regularization and dropout layers are used to reduce overfitting.

Model Architecture: The model consists of:


Input layer with image preprocessing Augmentation layers for data enhancement MobileNetV2 as the feature extractor Global average pooling layer Fully connected output layer with softmax activation for multi-class classification.

b. Model Compilation:
The model is compiled with:

Optimizer: Adam with a learning rate of 0.0001
Loss function: SparseCategoricalCrossentropy (since the labels are integers)
Metrics: Accuracy
c. Model Training:
First Stage: The model is trained for 10 epochs using the frozen MobileNetV2 feature extractor.
Second Stage: After evaluating the model, the base MobileNetV2 model is unfrozen, and the training continues with a lower learning rate to fine-tune the entire model.
4. Model Evaluation
The model’s performance is evaluated using the test dataset, and accuracy metrics are plotted for both the first and second stages of training. The model achieves an accuracy of over 80% on the test dataset.

5. Model Deployment

b. Model Compilation:
- Optimizer: Adam with a learning rate of 0.0001
- Loss function: SparseCategoricalCrossentropy (since the labels are integers)
- Metrics: Accuracy
  
c. Model Training:
-First Stage: The model is trained for 10 epochs using the frozen MobileNetV2 feature extractor.
-Second Stage: After evaluating the model, the base MobileNetV2 model is unfrozen, and the training continues with a lower learning ---rate to fine-tune the entire model.

## 4. Model Evaluation
The model’s performance is evaluated using the test dataset, and accuracy metrics are plotted for both the first and second stages of training. The model achieves an accuracy of over 80% on the test dataset.

## 5. Model Deployment

The trained model is used to make predictions on new images. After loading and preprocessing a new image, the model can predict whether the artery is Healthy or Unhealthy.

To make a prediction, the following steps are followed:

Load and preprocess the image (resize, normalize, etc.).
Make predictions using the trained model.
Display the predicted class.

6. TensorBoard for Training Monitoring
TensorBoard is used for visualizing and monitoring the training process. It provides useful insights into the training and validation loss, accuracy, and other metrics.

![alt text](<static/loss epoch.jpg>)
Loss Graph: Shows the loss values during both stages of training.
![alt text](<static/accuracy epoch.jpg>)
Accuracy Graph: Shows the accuracy achieved during both stages of training.

a. Setting Up TensorBoard:
TensorBoard is integrated into the project using Keras callbacks. The following steps are followed to enable TensorBoard monitoring:

Callback Setup: A TensorBoard callback is created, and the log files are stored in the transfer_learning_log directory with a timestamp.

Early Stopping: To avoid overfitting, an EarlyStopping callback is used to stop training early if the validation loss doesn’t improve after 5 epochs.

Credit: You can get full raport datasets [here](https://www.kaggle.com/datasets/harideepak/stenosis-new)

