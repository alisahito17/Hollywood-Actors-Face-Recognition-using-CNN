# 🎬 Hollywood Actors Face Recognition using CNN

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Convolutional Neural Network (CNN)** project designed to recognize popular Hollywood actors from their facial images. This repository covers the complete deep learning workflow, including **image preprocessing**, **data augmentation**, **model training**, **evaluation**, and **showing real-world predictions**.

Built with **TensorFlow** and **Keras** for an accurate, reproducible, and easy-to-understand deep learning solution.

---

## 📚 Table of Contents

1.  [Overview](#1-overview)
2.  [Getting Started](#2-getting-started)
    * [Installation](#installation)
    * [Project Run Commands](#project-run-commands)
3.  [Dataset](#3-dataset)
4.  [How It Works (Pipeline)](#4-how-it-works-pipeline)
5.  [CNN Model Architecture](#5-cnn-model-architecture)
6.  [Training Parameters](#6-training-parameters)
7.  [Evaluation & Results](#7-evaluation--results)
8.  [Folder Structure](#8-folder-structure)
9.  [Future Improvements](#9-future-improvements)
10. [Author](#10-author)

---

## 1. Overview

This project implements a robust Convolutional Neural Network (CNN) to classify a dataset of Hollywood actors based on their facial images.

The comprehensive pipeline is designed to be **clear, reproducible, and educational** for deep learning enthusiasts. It encompasses all critical steps:
* Loading and preprocessing raw image data.
* Applying advanced data augmentation techniques.
* Training a custom-designed CNN model.
* Evaluating performance and visualizing predictions.

## 2. Getting Started

### Installation

First, clone the repository and navigate into the project directory:

git clone https://github.com/alisahito17/hollywood-actors-face-recognition-cnn.git
cd hollywood-actors-face-recognition-cnn

Install the required Python dependencies using the provided `requirements.txt` file:

# Create a requirements.txt if you haven't already
pip freeze > requirements.txt 

# Install dependencies
pip install -r requirements.txt
**Required Libraries:** `tensorflow`, `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`

### Project Run Commands

To train the model and see the evaluation:

python main.py

## 3. Dataset

The project expects the dataset to be structured in a class-per-folder format.

dataset/
    ├── pins_alycia_dabnem_carey/
    │   ├── alycia_dabnem_carey0_0.jpg
    │   ├── alycia_dabnem_carey1_1.jpg
    │   └── ...
    ├── pins_amber_heard/
    │   ├── amber_heard0_0.jpg
    │   └── ...
    └── pins_zendaya/
        ├── zendaya0_0.jpg
        └── ...

* **Rule:** Each directory within `dataset/` represents a single actor (a class).
* **Images:** Images inside each folder belong to that specific actor.
* **Note:** If the dataset is large, **do not upload it to GitHub**. Instead, include a `README.txt` in the `dataset/` folder with clear instructions (e.g., a link) on how to download the data.

## 4. How It Works (Pipeline)

The deep learning pipeline is executed in the following steps:

1.  **Load Libraries:** Import necessary modules like TensorFlow, Keras, NumPy, and OpenCV.
2.  **Image Preprocessing:** Load images from the dataset and apply uniform resizing (e.g., to $100 \times 100$) and normalization (scaling pixel values to $0-1$).
3.  **Data Split:** Divide the dataset into **training** and **testing** sets.
4.  **Label Encoding:** Convert actor names (categorical labels) into **one-hot encoding** for model training.
5.  **Data Augmentation:** Apply real-time transformations (rotation, zoom, flips, shifts) to the training data using **ImageDataGenerator** to prevent overfitting.
6.  **Model Build:** Construct the custom CNN architecture (see below).
7.  **Training:** Train the model using the Adam optimizer, Categorical Crossentropy loss, and a learning rate scheduler.
8.  **Evaluation:** Evaluate the trained model on the test set and display performance metrics and sample predictions.

## 5. CNN Model Architecture

The model is a sequential CNN with **four Convolutional Blocks** designed for robust feature extraction from facial images.

| Layer Type | Parameters | Output Shape |
| :--- | :--- | :--- |
| **Input** | $100 \times 100 \times 3$ (RGB Image) | - |
| **Conv Block 1** | Conv(64) + BatchNorm + ReLU + MaxPooling | $50 \times 50 \times 64$ |
| **Conv Block 2** | Conv(128) + BatchNorm + ReLU + MaxPooling | $25 \times 25 \times 128$ |
| **Conv Block 3** | Conv(256) + BatchNorm + ReLU + MaxPooling | $12 \times 12 \times 256$ |
| **Conv Block 4** | Conv(512) + BatchNorm + ReLU + MaxPooling | $6 \times 6 \times 512$ |
| **Flatten** | - | - |
| **Dense Layer** | Dense(512) + ReLU + **Dropout(0.4)** | $512$ |
| **Output Layer** | Softmax (**Number of Actors**) | $N$ (Classes) |



## 6. Training Parameters

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Standard choice for deep learning |
| **Loss Function** | Categorical Crossentropy | Used for multi-class classification |
| **Batch Size** | 32 | Number of samples per gradient update |
| **Epochs** | 50 | Total passes over the training data (Adjustable) |
| **Data Usage** | Augmented Data | Uses real-time augmented images via `ImageDataGenerator` |

## 7. Evaluation & Results

The model's performance is gauged by:

* **Test Accuracy:** The final accuracy score on the unseen test set is printed to the console upon completion.
* **Visual Predictions:** A set of sample images from the test set are displayed, showing the **True Label** vs. the **Predicted Label**. This helps to quickly visualize the model's performance and identify patterns in misclassifications.



## 8. Folder Structure

hollywood-actors-face-recognition-cnn/
    ├── main.py                  # Main execution script
    ├── README.md                # Project description
    ├── requirements.txt         # Python dependencies
    ├── model/
    │   └── trained_model.h5     # Optional trained model
    ├── samples/
    │   ├── training_examples.png
    │   └── predictions.png
    └── dataset/
        └── README.txt           # Dataset instructions

## 9. Future Improvements

* **Face Detection:** Implement a dedicated face detection algorithm (e.g., **MTCNN** or **Haar Cascades**) as a preprocessing step to ensure only the face is fed to the CNN.
* **Real-time Recognition:** Integrate with a webcam to perform live, real-time face recognition.
* **Deployment:** Convert the Keras model to a production-ready format like **TFLite** (for mobile) or **ONNX**.
* **Web/Desktop Interface:** Build a simple user interface using frameworks like **Streamlit**, **Flask**, or **Gradio**.
* **Hyperparameter Tuning:** Use tools like **Keras Tuner** or **Weights & Biases (W&B)** to optimize hyperparameters for better classification accuracy.

## 10. Author

* **Ali** – AI student and deep learning enthusiast
* **GitHub:** [https://github.com/alisahito17](https://github.com/alisahito17)
