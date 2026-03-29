# COMP9444 Default Group Project notebook
## Automated Yoga Pose Classification 
## Introduction

### Project Purpose
The main purpose of this project is to develop a yoga pose classification system. The reason our group chose this project is that yoga, as a very popular sport recently, has many different postures which have a wide range of applications in various fields such as health monitoring, motion analysis and interactive fitness systems. In recent years, with the rapid development of deep learning techniques, especially the wide application of convolutional neural networks, the performance of pose estimation has been significantly improved.

### Challenges
However, the current pose estimation system still faces great challenges when facing complex yoga poses. The diversity and refinement of yoga postures are far beyond the common daily activity postures. For example, many yoga postures involve body contortions, inversions, or unusual balancing postures, all of which pose difficulties for traditional pose estimation systems. In addition, the similarity of yoga postures increases the likelihood of miscalculation, e.g. Dolphin Pose and Wide-Legged Forward Bend Pose, both of which are spread-legged, head-down postures, where attention needs to be paid to the orientation of the hands in order to determine which pose is which. Yoga practice is often performed in a variety of environments, including different lighting conditions, background complexity, and camera angles, which further increases the difficulty of accurate identification.

### Dataset and Methodology
This study references the Yoga-82 dataset, which contains complex yoga poses that reflect the diversity and complexity of the real world. These postures are difficult to be accurately captured by traditional pose annotation methods due to their richness in details and nuances. The method they use is DenseNet, whose results are highly accurate, but still lacking in real-time performance. In order to do so, we borrowed the method of defining pose estimation as a classification task, performed a hierarchical pose classification to handle and recognize various yoga poses more accurately, and used different image classification models to try to improve his real-time performance.

### Technical Approach
In our study, we have chosen a highly friendly technique for real-time performance - using the latest YOLOv8n-pose pose detection with FastViT image classification model. Our approach combines the state-of-the-art pose detection techniques with an efficient image classification model, aiming to achieve fast and accurate yoga pose recognition. Specifically, we use YOLOv8n-pose for initial human pose detection to effectively remove background interference and localize the human body. Subsequently, we perform fine-grained pose classification using the FastViT model, a lightweight but efficient model that captures subtle features of yoga poses.

### System Capabilities
Our system is able to recognize and classify poses in near real-time while maintaining high accuracy. This allows the system to be used in a variety of scenarios, from individual home practices to large yoga classes, providing users with real-time, accurate pose feedback. By combining innovative pose detection techniques with advanced deep learning models, our system not only improves the accuracy of recognizing yoga poses, but also significantly improves the real-time performance, enabling real-time and accurate feedback in a variety of real-world application scenarios, both for individual and group practices. This not only promotes the application of computer vision technology in the field of complex human postures recognition, but also provides yoga enthusiasts with a powerful learning and practicing tool, which significantly improves the safety and effectiveness of practice.

### Data Reproducibility
To ensure the reproducibility and accessibility of the data, we provide the complete dataset download location and processing steps, including specific script commands and necessary package dependencies. In this way, other researchers can easily re-generate the same training and test datasets and conduct further experiments and analysis as needed. This choice of methodology ensures the efficiency of data processing and the reliability of model training.

## Data Sources

### Primary Data Source
The main data source used in this project is the Yoga-82 resource which is “Yoga Pose Image Classification Dataset” that is publicly available on the Kaggle platform, which can be found under the Provenance tab, and to which some data has been added to add up to 107 categories. The original Yoga-82 dataset provides a rich set of yoga pose images, but these images need to be downloaded individually through multiple, possibly defunct, links, which makes the data acquisition process both tedious and inefficient. To solve this problem, we chose a more stable version of the dataset that has been integrated and extended.

### Dataset Details
This expanded version of the dataset contains 107 different yoga pose categories, each of which consists of thousands of images captured from multiple perspectives. Such a rich set of perspectives is necessary to train a model that can accurately recognize and distinguish subtle differences in poses. The pre-processing phase of the data involves the use of advanced pose detection algorithms to recognize and extract human skeletal structures from the raw images, which is essential to minimize background interference and improve the accuracy of the classification model. Next, these skeletal maps are transformed into a format suitable for deep learning training using customized data processing scripts.
