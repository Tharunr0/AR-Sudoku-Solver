# 📷 AR Sudoku Solver

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Server-lightgrey?style=for-the-badge&logo=flask&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **A real-time Augmented Reality web application that detects, solves, and overlays Sudoku solutions using Computer Vision and Deep Learning.**

---

## 🌟 Overview

This project bridges the gap between **Computer Vision** and **Web Development**. It captures a video feed of a Sudoku puzzle, processes the image using **OpenCV** to extract the grid, identifies the digits using a custom-trained **Convolutional Neural Network (CNN)**, solves the puzzle algorithmically, and projects the solution back onto the original image in the correct perspective.

The user interface features a modern, dark-mode design with smooth motion graphics, built on **Flask**.

## ✨ Features

* **🔍 Smart Detection:** automatically identifies the largest sudoku grid in the camera frame.
* **🧠 Deep Learning Recognition:** Uses a custom CNN model trained on the MNIST dataset to read digits.
* **⚡ Real-Time Solving:** utilizes a backtracking algorithm to solve even the hardest puzzles in milliseconds.
* **🌐 Web Interface:** A responsive Flask web app with motion-designed loading screens and scanning effects.
* **📐 Perspective Warping:** dynamically flattens the grid for processing and un-warps the solution to match the camera angle.

---

## 📸 Demo & Screenshots

| **Web Interface** | **Solved Overlay** |
|:---:|:---:|
| *(Replace this text with a screenshot of your UI)* | *(Replace this with a screenshot of a solved puzzle)* |

---

## 🛠️ Tech Stack

### **Backend & AI**
* **Python:** Core logic.
* **OpenCV:** Image preprocessing, contour detection, and perspective transforms.
* **TensorFlow / Keras:** Building and training the digit recognition model.
* **NumPy:** Matrix operations for the grid logic.

### **Web & Frontend**
* **Flask:** Lightweight web server to handle image requests.
* **HTML5 / CSS3:** Modern, dark-themed UI with CSS animations.
* **JavaScript:** Handling camera streams and asynchronous server requests (AJAX).

---

## 📂 Repository Structure

```text
AR-Sudoku-Solver/
├── models/
│   └── digit_model.h5       # Pre-trained CNN model
├── templates/
│   └── index.html           # Frontend UI
├── training/
│   └── train_model.py       # Script to train the AI on MNIST
├── app.py                   # Main Flask Application
├── utils.py                 # Computer Vision & Solver Helper functions
├── requirements.txt         # Dependencies
└── README.md                # Documentation
