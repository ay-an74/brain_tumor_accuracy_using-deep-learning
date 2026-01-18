🧠 Brain Tumor Accuracy Using Deep Learning

A deep learning–based medical imaging project that uses Convolutional Neural Networks (CNNs) to classify brain tumor MRI images and evaluate model accuracy. The project focuses on data preprocessing, visualization, model training, and performance evaluation using TensorFlow and Keras.

📌 Project Overview

Brain tumor detection from MRI scans is a critical task in medical diagnosis.
This project implements a CNN-based image classification pipeline to identify different types of brain tumors from MRI images and measure classification accuracy.

The implementation is done using Python, TensorFlow, and Keras, and executed in Google Colab with dataset access via Google Drive.

🎯 Objectives

Load and preprocess MRI image datasets

Perform exploratory data analysis and visualization

Build and train a CNN model

Evaluate model accuracy and performance

Visualize predictions and training metrics

🧪 Dataset Description

MRI brain scan images

Organized into class-wise folders

Used for training and validation

Images loaded using ImageDataGenerator

⚠️ Dataset is accessed from Google Drive in the notebook.

🛠️ Technologies & Libraries Used

🐍 Python

🧠 TensorFlow & Keras

📊 NumPy

📈 Pandas

📉 Matplotlib

🎨 Seaborn

☁️ Google Colab

🧩 Model Architecture

The CNN model includes:

Convolutional layers (Conv2D)

Pooling layers (MaxPooling2D)

Fully connected (Dense) layers

Activation functions (ReLU, Softmax)

Compiled with appropriate optimizer and loss function

📊 Data Visualization

The project includes:

Class-wise image count visualization

Sample MRI image visualization

Training & validation accuracy plots

Loss curves for performance analysis

▶️ How to Run the Project

Upload the notebook to Google Colab

Upload the dataset to Google Drive

Mount Google Drive:

drive.mount('/content/drive')


Update dataset path if required

Run all cells sequentially

✔ No local setup required if using Colab

📈 Results

Successfully trained CNN model on MRI images

Accuracy evaluated on validation data

Performance visualized using graphs

Model capable of classifying tumor categories based on input images

⚠️ Limitations

Dataset size affects accuracy

Model performance depends on image quality

Not intended for real-world medical diagnosis

Requires further validation for clinical use

🚀 Future Enhancements

Improve accuracy using transfer learning (VGG16, ResNet, MobileNet)

Hyperparameter tuning

Add confusion matrix and classification report

Deploy model using Flask or Streamlit

Integrate Grad-CAM for explainability

🎓 Academic Use

Machine Learning / Deep Learning mini project

Medical image analysis reference

CNN implementation practice

College final-year project base

🧾 Resume Project Description

Brain Tumor Detection Using Deep Learning
Developed a CNN-based deep learning model to classify brain tumor MRI images and evaluate model accuracy. Implemented data preprocessing, visualization, and performance analysis using TensorFlow, Keras, and Python in Google Colab.

⚖️ Disclaimer

This project is intended strictly for educational and research purposes.
It is not a medical diagnostic tool.
All medical decisions must be made by certified professionals.

👤 Author

Ayan Mondal
Deep Learning | Medical Imaging Project
