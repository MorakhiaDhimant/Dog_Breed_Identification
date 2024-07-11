# Dog Breed Identification using CNN Model

## Overview
This project implements a Convolutional Neural Network (CNN) model to classify dog breeds. The model achieves an accuracy of 99% on a dataset containing 200 different dog breeds. The implementation is based on the Scharet architecture.

## Dataset
The dataset is organized into training, validation, and test sets. It comprises a total of 200 dog breeds, with images for each breed provided in separate directories.

- Training set
- Validation set
- Test set

## Preprocessing
Image data is loaded and preprocessed using TensorFlow and Keras. Images are resized to 224x224 pixels and normalized to a scale of 0 to 1.

## Model Architecture
The CNN model is constructed with the following layers:
- Convolutional layers with max pooling for feature extraction
- Dropout layers for regularization
- Fully connected layers for classification

The model is compiled using the Adam optimizer and categorical cross-entropy loss function.

## Training
The model is trained for 20 epochs with a batch size of 32. Model checkpoints are saved to 'saved_models/weights.best.from_scratch.hdf5' to track the best performing weights during training.

## Evaluation
Training and validation loss/accuracy curves are plotted for visual assessment of model performance.

## Dependencies
- scikit-learn
- NumPy
- TensorFlow
- Keras
- Matplotlib

## How to Use
1. Ensure the dataset is organized as specified in the comments.
2. Run the provided code to train the model.
3. Evaluate model performance using the generated loss and accuracy plots.

Feel free to customize the model architecture, hyperparameters, or training duration based on your specific requirements.
