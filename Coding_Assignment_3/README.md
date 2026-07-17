# Image-Based Cancer Diagnosis Using Convolutional Neural Networks (Histopathologic Oral Cancer Detection)

## Overview

This study uses Convolutional Neural Networks (CNNs) to propose an end-to-end deep learning pipeline automated oral cancer diagnosis using histopathological images. To correctly identify malignant and non-cancerous images, the pipeline consists of image preprocessing, data augmentation, CNN model building, training, assessment, and visualization.

## Dataset Information

**Dataset:** Oral Squamous Cell Carcinoma (OSCC) Histopathological Image Dataset

**Source:** Kaggle

The dataset consists of histopathological images categorized into **Normal** and **OSCC (Oral Squamous Cell Carcinoma)** classes for binary image classification.

---

## Deep Learning Model

A custom Convolutional Neural Network (CNN) was developed from scratch with the following components:

- Image preprocessing and normalization
- Data augmentation
- Convolutional layers with ReLU activation
- Max Pooling layers
- Batch Normalization
- Dropout regularization
- Fully Connected classification layer
- Sigmoid output for binary classification

---

## Evaluation

The CNN model was evaluated using multiple performance metrics and visualization techniques, including:

- Training and Validation Accuracy
- Training and Validation Loss
- ROC Curve and AUC Score
- Confusion Matrix
- Model Performance Analysis

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Torchvision

---

## Conclusion

This project demonstrates the effectiveness of Convolutional Neural Networks for automated oral cancer detection using histopathological images
