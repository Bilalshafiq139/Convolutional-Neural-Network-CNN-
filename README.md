# Convolutional-Neural-Network-CNN-
Convolutional Neural Network (CNN) for MNIST Handwritten Digit Classification



# Convolutional Neural Network (CNN) for MNIST Handwritten Digit Classification

## Overview
This repository contains a Python script that implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset**. The model is trained to recognize digits (0-9) from grayscale images and provides predictions with high accuracy.

---

## **Problem Statement**
Handwritten digit recognition is a fundamental task in computer vision. This project uses **CNN** to classify **28x28 grayscale images** of digits into one of 10 classes, helping automate handwritten digit recognition tasks such as postal code identification and bank check processing.

### **Dataset Description**
The dataset used in this project is the **MNIST Handwritten Digits Dataset**:
- **60,000 training images** and **10,000 test images**.
- Each image is **28x28 pixels** in grayscale.
- Labels range from **0 to 9**, representing digits.

---

## **How the Script Works**
1. **Data Preprocessing**:
   - Loads the MNIST dataset.
   - Normalizes pixel values to the range **0-1**.
   - Reshapes images to `(28, 28, 1)` to match CNN input format.
   - One-hot encodes the labels.
2. **CNN Model Architecture**:
   - **Input Layer**: Accepts 28x28 grayscale images.
   - **Convolutional Layer 1**: 32 filters, `(3,3)` kernel, **ReLU activation**.
   - **MaxPooling Layer 1**: `(2,2)` pool size.
   - **Convolutional Layer 2**: 64 filters, `(3,3)` kernel, **ReLU activation**.
   - **MaxPooling Layer 2**: `(2,2)` pool size.
   - **Flatten Layer**: Converts feature maps into a **1D vector**.
   - **Dense Layer**: 128 neurons, **ReLU activation**.
   - **Output Layer**: 10 neurons, **Softmax activation**.
3. **Compilation & Training**:
   - Uses the **Adam optimizer**.
   - Trains for **10 epochs** with a batch size of **32**.
4. **Model Evaluation**:
   - Computes test accuracy.
   - Generates classification report (Precision, Recall, F1-Score).
5. **Prediction on Sample Images**:
   - Selects 5 random test images.
   - Displays images with **actual vs. predicted** labels.
6. **Model Saving & Reloading**:
   - Saves the trained model to `mnist_cnn_model.keras`.
   - Reloads the model for further predictions.

---

## **Dependencies**
Ensure you have the following Python libraries installed:
```sh
pip install numpy tensorflow matplotlib scikit-learn
```

---

## **How to Run the Script**
1. Clone the repository:
   ```sh
   git clone <repository_url>
   ```
2. Navigate to the project folder:
   ```sh
   cd <project_folder>
   ```
3. Run the Python script:
   ```sh
   python cnn.py
   ```
4. Follow the output to view test accuracy and predictions.

---

## **Contributing**
Contributions are welcome! Feel free to fork this repository and submit a pull request with improvements or additional features.


