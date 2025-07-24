# 😷 Face Mask Detection using CNN

A Convolutional Neural Network (CNN) model that detects whether a person is wearing a face mask or not from an image. Built as part of the AI & Deep Learning Scholarship – Computer Vision Track by NTI.

---

## 🧠 Model Overview

This model classifies images into two categories:

- ✅ **With Mask**
- ❌ **Without Mask**

---

## 📁 Dataset

- The dataset used contains thousands of labeled images of people **with** and **without** masks.
- Images were preprocessed and resized to `(128, 128, 3)` before training.
- Data was split into **Training**, **Validation**, and **Testing** sets.

---

## 🏗️ Model Architecture

- Built using **Convolutional Neural Networks (CNN)** with:
  - `Conv2D` + `BatchNormalization` + `MaxPooling`
  - `Dropout` to prevent overfitting
  - Final output layer with **sigmoid** activation for binary classification

---

## 📈 Results

- Achieved high accuracy on training and validation datasets.
- Model was saved in `.h5` and `.keras` formats.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Asmaaelkashef/FaceMask-Detection-Model.git
cd FaceMask-Detection-Model
