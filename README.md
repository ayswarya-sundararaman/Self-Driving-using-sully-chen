
# Self-Driving Car Steering Angle Prediction

This project predicts the steering angle of a car based on front-view images using CNN (Convolutional Neural Networks). The model is built using **NVIDIA's CNN architecture** and trained on images from SullyChen's self-driving dataset.

## Dataset
- **SullyChen's Dataset**: 
  - Front-view images taken at 30 frames per second from a driving car for 25 minutes.
  - Each image is paired with the corresponding steering angle.

## Problem Statement
- **Objective**: Predict the steering angle of the car based on input images, using a regression model.
  
## Techniques Used
1. **Preprocessing**:
   - Images are resized to 200x66 pixels and normalized.
   - Temporal train-test split (70% training, 30% testing).
   - Used OpenCV for visualization and image preprocessing.

2. **CNN Model**:
   - **NVIDIA's CNN architecture** designed for self-driving cars.
   - Five convolutional layers followed by fully connected layers.
   - **Dropout** applied to prevent overfitting.
   - **Adam optimizer** with a learning rate of 1e-3 used for training.
   
3. **Training**:
   - Trained over 30 epochs with Mean Squared Error (MSE) as the loss function.
   - Loss includes L2 regularization to prevent overfitting.

## Results
- The model achieved continuous improvement in reducing training and validation losses across epochs.
- Loss values decreased significantly as the model learned to predict the steering angles accurately.

## Conclusion
The NVIDIA-based CNN architecture proved effective in predicting steering angles from front-view car images. Hyperparameter tuning, such as dropout rates and optimizer configurations, further improved the model's accuracy.

