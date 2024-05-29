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
Implementing a multiple instance learning (MIL) framework to process these weakly labeled videos.
Designing a deep learning model that can learn to identify and rank anomalous segments within a video.
Ensuring the system can handle diverse and changing environments captured by surveillance cameras, reducing false alarm rates.

# 🤖 Our Model

We using a ResNeXtBottleneck pre-trained model combined with a custom classifier (Learner model) to detects anomaly in videos. ResNeXtBottleneck is pre-trained on UCF-Crime datasets , which allows it to learn rich feature representations. This pre-training helps the model extract meaningful features from new datasets with fewer training samples. Additionally, ResNeXtBottleneck's deep and wide architecture, which includes multiple paths for feature extraction, enables it to capture a variety of features at different levels of abstraction. This flexibility means the output features from ResNeXtBottleneck can be easily adapted to various downstream tasks without re-training the entire network. ResNeXtBottleneck has also demonstrated strong performance on various benchmarks, making it a reliable choice for feature extraction.

The custom Learner model, a simple feedforward neural network, takes the features extracted by ResNeXtBottleneck and performs the final classification. Its simplicity makes it easy to customize and tune for specific tasks. Including a dropout layer helps regularize the model, reducing the risk of overfitting, especially with limited data. By training only the Learner model and keeping the feature extractor frozen (or fine-tuning only the last few layers), we can achieve good performance with reduced computational resources and training time. This combination leverages the strengths of both models, providing an efficient solution for video anomaly detection.

## 🏅 Accurate Model

Compared to 6 studies in the field of detecting anomaly behaviors in surveillance cameras, the performance of our model was the most accurate, as it topped the six studies with a rate of 84%!

# 🛠️ Implementation

In this section, we will explain the methods of building a video Anomaly detection system, and therefore we divided this section into five parts.  Data Description , Feature Extractor ,  MIL , Learner Model , and Evaluation .

## ✅ Data Description

In this project, we used a large-scale dataset called the UCF-Crime dataset, specifically constructed to evaluate anomaly detection methods in surveillance videos. The dataset includes 13 types of real-world anomalies, selected for their significant impact on public safety: Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism. Videos were sourced from YouTube and LiveLeak using text search queries. The dataset consists of 1900 videos in total. Among these videos, 950 contain clear anomalies videos, while the rest are considered normal.  Figure shown below depicts a sample of anomalies from the UCF dataset.
 As for the challenges we faced while dealing with this dataset, we didn't encounter any issues. However, during the download process, it took a lot of time due to the large size of the videos, which were long and untrimmed.
 
 ![image](https://github.com/Ralrouqi/Anomaly-Behavior-Classify/assets/93721390/4c419c34-4914-429b-bda6-de439b5a3dc6)
