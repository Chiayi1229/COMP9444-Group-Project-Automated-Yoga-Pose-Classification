# COMP9444-Group-Project-Automated-Yoga-Pose-Classification

Course: COMP9444 Neural Networks and Deep Learning
University: UNSW (University of New South Wales)

---

## Introduction

This project develops a yoga pose classification system using deep learning techniques.
Yoga has many complex postures and is widely used in areas such as health monitoring, motion analysis, and interactive fitness systems.

With the development of deep learning, especially convolutional neural networks, pose estimation performance has significantly improved.

---

## Challenges

Recognizing yoga poses is challenging due to:

* Complex body movements and balancing postures
* Similarity between different poses
* Different environments, lighting conditions, and camera angles

For example, some poses such as Dolphin Pose and Wide-Legged Forward Bend Pose are visually similar and require detailed feature recognition.

---

## Dataset and Methodology

This project uses the **Yoga-82 dataset**, which contains complex yoga pose images that reflect real-world diversity.

The dataset was expanded to **107 pose categories** to improve model training and classification performance.

The project treats pose estimation as a **classification problem** and applies hierarchical pose classification to improve recognition accuracy.

---

## Technical Approach

The system combines pose detection and image classification models.

Models used:

* YOLOv8n-pose for human pose detection
* FastViT for image classification
* DenseNet as a reference model

Workflow:

1. Detect human pose using YOLOv8n-pose
2. Remove background interference
3. Extract pose features
4. Classify yoga poses using FastViT

This approach improves both **accuracy** and **real-time performance**.

---

## System Capabilities

The system is designed to:

* Recognize yoga poses
* Classify poses in near real-time
* Provide accurate pose recognition in different environments

Possible applications include:

* Personal yoga practice
* Interactive fitness systems
* Motion analysis

---

## Data Sources

Primary dataset:

Yoga Pose Image Classification Dataset (Yoga-82)
Available on Kaggle

The dataset contains multiple perspectives of yoga poses to improve model training and recognition performance.

---

## Project Structure

data/
models/
data_process/
inference/
export_model/
Project_notebook.ipynb

---

## Data Reproducibility

The project includes scripts and configurations that allow researchers to reproduce the dataset processing and model training process.

This ensures:

* Reproducibility
* Reliable model training
* Easy experiment replication
