# Skin Disease Classification using CNN (HAM10000)

This project implements a Convolutional Neural Network (CNN) to classify skin diseases using dermatoscopic images.

The model is trained and evaluated on the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

---

## Dataset

The HAM10000 dataset contains 10,015 dermatoscopic images across 7 skin disease classes.

Due to its large size (~600 MB uncompressed), the dataset is **not included** in this repository.

### Official dataset source:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

### Dataset structure used in this project:


The dataset is split into training, validation, and test sets using stratified sampling.

---

## Model Architecture

- Input size: 224 × 224 × 3 (RGB)
- 3 Convolutional layers with ReLU activation
- MaxPooling after each convolution
- Fully connected layer with Dropout
- Softmax output layer for 7 classes

Framework: **TensorFlow / Keras**

---

## Training

- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Epochs: 15
- Data augmentation applied to training data

---

## Evaluation

The model is evaluated on a separate test set to measure generalization performance.

---

## How to Run

1. Download the dataset from Kaggle
2. Organize it according to the described folder structure
3. Run the provided Jupyter/Colab notebook

---

## Notes

This repository contains **code only**.  
The dataset must be downloaded from the official source due to size limitations.
