# Pneumonia Detection Deep Learning Model

This repository contains a deep learning model designed to detect pneumonia from chest X-ray images using the VGG16 architecture.

## Overview

The goal of this project is to create a robust pneumonia detection model leveraging deep learning techniques. The model is trained on a dataset obtained from Kaggle, consisting of chest X-ray images categorized into pneumonia and normal classes.

## Dataset

The dataset used for training and evaluation is sourced from Kaggle's Chest X-ray Images (Pneumonia) dataset, which includes X-ray images labeled as normal and pneumonia. The dataset contains a total of 5,863 images.

The dataset can be accessed from: [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Architecture

The deep learning model is built upon the VGG16 (Visual Geometry Group 16) architecture, a widely used convolutional neural network (CNN) architecture pre-trained on ImageNet. The model is fine-tuned using transfer learning to adapt it to the pneumonia detection task.

## Files

- `pneumonia-detection-project.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `pneumonia.h5`: Trained model weights in HDF5 format.
## Usage

1. **Data Preparation:**
   - Download the Kaggle dataset and organize it into appropriate folders (e.g., 'train', 'test').

2. **Training:**
   - Open and run the `pneumonia-detection-project.ipynb` notebook to train the model. Adjust hyperparameters as needed.

3. **Inference:**
   - Use the trained `pneumonia.h5` file for inference.
