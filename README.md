# 📸 Anomaly-Behavior-Classify (رَقـيـبّ)

# Table of Contents

- [📌 Introduction](#-Introduction)
- [💡 What is our goal ?](#-What-is-our-goal-)
- [🚩 What is our scope ?](#-what-is-our-scope-)
- [🤖 Our Model ](#-Our-Model-)
  - [🏅 Accurate Model ](#-Accurate-Model-)
- [🛠️ Implementation ](#%EF%B8%8F-Implementation-)
  - [✅ Data Description ](#-Data-Description-)
  - [⚙️ Feature Extractor ](#%EF%B8%8F-Feature-Extractor-)
  - [😉 MIL ](#-MIL)
  - [⭐ Learner Model](#Learner-Model-)
  - [📈 Evaluation](#Evaluation-)
  - [⚠️ Challenges](#Challenges-)
 - [🧩 How can you run the Demo ? ](#-How-can-you-run-the-Demo-)


# 📌 Introduction

With the growing need for enhanced security to safeguard lives and public property, surveillance cameras are increasingly deployed in various public spaces, including markets, shopping malls, hospitals, banks, streets, and educational institutions. The primary goal of this task is to monitor daily activities and detect anomalous events early. Anomalies in videos refer to events or behaviors that are out of the ordinary and indicate abnormal behavior, such as fights, car accidents, crimes, or illegal activity.
 Detecting anomalies in video is a critical task in many cases where human intervention is necessary to prevent crime. Nevertheless, this process demands human effort and constant monitoring, which is a tedious process, as abnormal events occur only 0.01% of the time, resulting in 99.9% of surveillance time being wasted. Additionally, surveillance systems generate a large amount of redundant video data, requiring unnecessary storage space. 
 Therefore, to reduce the waste of labor and time, there is an urgent need to develop intelligent computer vision algorithms for automatic video anomaly detection. Recently, this problem has garnered significant attention in computer vision research. Numerous researchers have sought to determine the best method for accurately detecting anomalies in video streams while minimizing false alarms. The outcomes demonstrated that deep learning-based methods provide highly intriguing outcomes in this field. Therefore, in this work, we proposed a real-time video anomaly detection model leveraging deep learning techniques, specifically employing ResNet.

 # 💡 What is our goal ?

 The objectives of our project include:
Develop a deep learning framework to detect anomalies in surveillance videos using weakly labeled data.
Enhance the system's ability to accurately identify anomalies while reducing false positives.
Design the system to detect a wide range of anomalous events without needing specific models for each type of event.

# 🚩 What is our scope ?

The scope of our project will focus on:
Using training videos that are weakly labeled, where only the overall video is labeled as either normal or containing an anomaly, but the specific location of the anomaly within the video is not known.
Implementing a **multiple instance learning (MIL)** framework to process these weakly labeled videos.
Designing a deep learning model that can learn to identify and rank anomalous segments within a video.
Ensuring the system can handle diverse and changing environments captured by surveillance cameras, reducing false alarm rates.

# 🤖 Our Model

We using a **ResNeXt Bottleneck** pre-trained model combined with a **custom classifier (Learner model)** to detects anomaly in videos. ResNeXtBottleneck is pre-trained on UCF-Crime datasets , which allows it to learn rich feature representations. This pre-training helps the model extract meaningful features from new datasets with fewer training samples. Additionally, **ResNeXt Bottleneck's** deep and wide architecture, which includes multiple paths for feature extraction, enables it to capture a variety of features at different levels of abstraction. This flexibility means the output features from **ResNeXt Bottleneck** can be easily adapted to various downstream tasks without re-training the entire network. **ResNeXt Bottleneck** has also demonstrated strong performance on various benchmarks, making it a reliable choice for feature extraction.

**The custom Learner model** a simple feedforward neural network, takes the features extracted by **ResNeXt Bottleneck** and performs the final classification. Its simplicity makes it easy to customize and tune for specific tasks. Including a dropout layer helps regularize the model, reducing the risk of overfitting, especially with limited data. By training only the Learner model and keeping the feature extractor frozen (or fine-tuning only the last few layers), we can achieve good performance with reduced computational resources and training time. This combination leverages the strengths of both models, providing an efficient solution for video anomaly detection.

###
![photo_2024-05-29_12-01-15](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/752d5a2d-43a0-4e53-b790-d70dee87d8f3)

## 🏅 Accurate Model

Compared to 6 studies in the field of detecting anomaly behaviors in surveillance cameras, the performance of our model was the most accurate, as it topped the six studies with a rate of 84%!
###
![Colorful Modern Line Chart Diagram Graph](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/283f12a7-ebed-4eac-9db3-a76c622013d0)
 ### The numberes shown in figure above indecite:
 1. Claws: Clustering assisted weakly supervised learning with normalcy suppression for anomalous event detection , with accuracy (83.03%)
 2. Weakly-supervised joint anomaly detection and classification , with accuracy (82.12%)
 3. Anomalous event recognition in videos based on joint learning of motion and appearance with multiple ranking measures , with accuracy (81.91%)
 4. Anomaly event detection in security surveillance using two-stream based model , with accuracy (81.22%)
 5. Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection , with accuracy (82.14%)
 6. Cleaning label noise with clusters for minimally supervised anomaly detection , with accuracy (78.27%)
 7. Our System (رَقـيـبّ) , with accuracy (84%) 🥇!!
 

# 🛠️ Implementation

In this section, we will explain the methods of building a video Anomaly detection system, and therefore we divided this section into five parts.  Data Description , Feature Extractor ,  MIL , Learner Model , and Evaluation .

## ✅ Data Description

In this project, we used a large-scale dataset called the **UCF-Crime dataset** specifically constructed to evaluate anomaly detection methods in surveillance videos. The dataset includes 13 types of real-world anomalies, selected for their significant impact on public safety: Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism. Videos were sourced from YouTube and LiveLeak using text search queries. The dataset consists of 1900 videos in total. Among these videos, 950 contain clear anomalies videos, while the rest are considered normal.  Figure shown below depicts a sample of anomalies from the UCF dataset.
 As for the challenges we faced while dealing with this dataset, we didn't encounter any issues. However, during the download process, it took a lot of time due to the large size of the videos, which were long and untrimmed.
 ###
 &nbsp;
 ![image](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/4c419c34-4914-429b-bda6-de439b5a3dc6)

You can explore our data from here: ![link](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?e=1&dl=0)
## ⚙️ Feature Extractor

The ResNeXtBottleneck model acts as a powerful feature extractor by leveraging grouped convolutions and deep hierarchical layers to capture complex features from video data. The extracted features can then be used in downstream tasks (anomaly detection).

## 😉 MIL

The MIL function implements the Multiple Instance Learning loss calculation. This loss function is designed to distinguishes between anomaly and normal instances within each batch and calculates a loss that encourages the model to separate anomaly scores from normal scores. To compute the MIL for anomaly detection, we distinguish between anomaly and normal instances within each batch. The loss function encourages the model to separate anomaly scores from normal scores effectively. The initial loss is calculated using the maximum scores for both anomaly and normal instances, with sparsity and smoothness penalties added. Finally, the average loss across the batch is computed. The final equation for the average loss is:
###
 &nbsp;
![Screenshot 2024-05-28 102202](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/607f4218-84a7-4d66-aee1-1d6d27266fd9)

## ⭐ Learner Model

The Learner class is responsible for implementing a neural network classifier with a sequential architecture comprising linear layers, ReLU activations, dropout regularization, and a final sigmoid activation. During initialization, it sets up the classifier architecture and initializes the weights using Xavier normal initialization, ensuring a balanced initialization for effective training. The class collects the parameters of the classifier within self.vars for easy management and access during training. In the forward pass, the Learner class defines the sequence of operations using the collected parameters, sequentially applying linear transformations, ReLU activations, and dropout regularization. Finally, it returns the output after passing through the sigmoid activation function, providing a streamlined approach to neural network classification tasks.

## 📈 Evaluation


 We can see that our approach showed an accuracy of 84% for the UCF-Crime dataset, as indicated by the receiver operating characteristic (ROC) curve in Figure 1 shown below. This represents a significant improvement over previous studies in the field of anomaly detection. The 84% accuracy highlights the effectiveness of our model in classifying events. Additionally, our model achieved an F1 score of 85%, further confirming its reliability and precision in detecting unusual activities, as shown below in Figure 2.
 The significant improvement in accuracy and F1 score shows that our model is robust and can be used in real-world surveillance systems. These results indicate that the system can effectively enhance public safety and security by accurately and promptly detecting anomalous activities, thus supporting the goals of Saudi Vision 2030 to create safer communities.
###
 ![roc_curve](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/bff24ff7-d9f8-4ece-9546-16b6af92a930)
### 📉 Figure 1 ((ROC) curve)
![f1_scores](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/f2a5130b-e50b-417a-9675-2ab76b390db1)
### 〽️ Figure 2 (F1 Score)

## ⚠️ Challenges

### Dataset Challenge: 
As for the challenges we faced while dealing with this dataset, we didn’t encounter any issues. However, during the download process, it took a lot of time due to the large size of the videos, which were long and untrimmed.
### First Experminet Issue:
The batch size of the architecture caused problems for the model during testing. The model failed to recognize different video batches (segments), which varied according to each video and its range. A solution was tested by making the segments of the video static, which enabled the model to work but resulted in overfitting, with both detection part accuracy and classification part accuracy reaching 100% from the first epochs. We attempted to optimize the model by using dropout layers, changing the segment sizes to different ranges, and increasing the training data. However, due to lack of time, we decided to stop fixing the code and shifted the focus to only anomaly detection.

# 🧩 How can you run the Demo ? 

To run (رَقــيــبّ) system you must follow these steps: 
### 1 . Clone our repo
### 2 . Download the requirments.py to import the neccesary libraries
### 3 . Open the terminal at your VS code
### 4 . Run ' Streamlit run Demo.py ' , Note: You should install the Streamlit library !

# Try it and tell us about your feedback 🤝!
