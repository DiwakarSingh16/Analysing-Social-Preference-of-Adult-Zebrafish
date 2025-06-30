# 🐟 Analysing Social Preference of Adult Zebrafish  
### A Laboratory-based Behavioral Study through Development of a Deep Learning Algorithm



![Computer Vision](https://img.shields.io/badge/Computer%20Vision-🔍-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics%20YOLOv8-vision-green)
![Roboflow](https://img.shields.io/badge/Roboflow-Inference-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FPyTorch-purple)
![Tracking](https://img.shields.io/badge/Object%20Tracking-Centroid%20+%20TrackID-yellow)



[![Lab Setup](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRneKxQWdss4aYIzJ83723xU0dGxqdKoMfkCA&s)](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRneKxQWdss4aYIzJ83723xU0dGxqdKoMfkCA&s)


---

## 📘 Overview

This project investigates the **social behavior of adult zebrafish (Danio rerio)** using a **deep learning-based object detection and tracking algorithm**. The study leverages video analysis and region-based metrics to quantify **social preferences** in controlled laboratory conditions.

---

## 🎯 Objective

- To automatically detect, track, and analyze the movement of zebrafish in a test tank.
- To identify **zones of social preference** based on spatial and temporal metrics.
- To compute **close encounter probabilities** using bounding box distances and track IDs.

---

## 🧪 Experimental Setup

- **Test Tank**: Divided into distinct zones (e.g., Region C and Region D).
- **Stimulus**: Fish in adjacent tanks serve as social stimuli.
- **Input**: Surveillance video recorded during experiments.
- **Output**: Annotated video with counts, bounding boxes, and heatmaps.



## 💡 Methodology

### 1. 🎥 Video Input

- Format: `.mp4`
- Frame Rate: 5 FPS
- Resolution: e.g., 720p

### 2. 🧠 Detection & Tracking

- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Framework: [Roboflow Inference](https://github.com/roboflow/inference)
- Object detection and fish ID tracking applied across video frames



## 🧠 Fish Motion Detection using Object Detection (YOLO-based Model)

---

### 📦 1. Environment Setup

| Parameter                | Details                                                      |
|--------------------------|--------------------------------------------------------------|
| **Platform**             | Google Colab (Free Tier)                                     |
| **Language**             | Python 3.10+                                                 |
| **Frameworks Used**      | Ultralytics YOLOv8, Roboflow SDK                             |
| **Training Location**    | Roboflow Hosted Training                                     |
| **Inference Location**   | Google Colab + Roboflow Hosted Inference                     |
| **Minimum RAM Required** | 4–8 GB RAM (Colab Free Tier works well)                      |
| **GPU Requirements**     | Optional (for local training); not required with Roboflow    |
| **Dataset Used**         | Fish Motions Dataset (annotated via Roboflow)                |
| **Classes**              | Fish A1, Fish A2, Fish A3, Fish B                            |

---

### ⚙️ 2. Model Configuration

| Parameter           | Value                                              |
|---------------------|----------------------------------------------------|
| **Model Type**      | Roboflow 3.0 Object Detection (Fast)               |
| **Pretrained On**   | COCO Dataset                                       |
| **Backbone**        | CSPDarknet-like                                    |
| **Detection Method**| One-stage Object Detector (YOLO-style)             |
| **Model Hosted On** | Roboflow Cloud                                     |
| **Export Formats**  | ONNX / TorchScript / CoreML / TensorFlow Lite      |
| **Input Image Size**| 416×416 or 640×640 (auto-scaled)                   |
| **Confidence Threshold** | 50%                                          |
| **IoU Threshold (NMS)** | 50%                                           |

---

### 🛠️ 3. Training Hyperparameters

| Parameter                   | Value       |
|-----------------------------|-------------|
| **Epochs**                  | ~300        |
| **Optimizer**               | SGD (Roboflow default) |
| **Learning Rate**           | 0.01 (decayed) |
| **Weight Decay**            | 0.0005      |
| **Batch Size**              | 16 (approx.)|
| **Early Stopping (Patience)** | Enabled  |

---

### 📊 4. Evaluation Metrics (from Roboflow)

#### 🔹 Overall Metrics

| Metric         | Value   |
|----------------|---------|
| **mAP@50**     | 88.5%   |
| **Precision**  | 94.7%   |
| **Recall**     | 81.0%   |

#### 🔹 Class-wise mAP@50

| Class     | AP (%) |
|-----------|--------|
| Fish A1   | 100.0  |
| Fish A2   | 61.0   |
| Fish A3   | 94.0   |
| Fish B    | 100.0  |
| **Average** | **88.5%** |



---

### 📈 5. Training Graph Summary

- **Box Loss**: Reduced from `2.8 → 1.5`  
- **Class Loss**: Dropped from `4.5 → 1.0`  
- **Objectness Loss**: Stabilized around `1.3`  
- **mAP Curve**: Converged smoothly after ~100 epochs  
- **Overfitting**: Not observed during training

---

## 🎬 Input & Output Videos

### 🎥 Input Video: Clear f1.mp4  
👉 [Click to Watch on Google Drive](https://drive.google.com/file/d/1RFa7ih00yVJvOX-YX-CtfbAbZRNId08c/view?usp=drive_link)



---

### 🐟 Output Video: YOLOv8 Object Detection  
👉 [Click to Watch on Google Drive](https://drive.google.com/file/d/1qqEVjmq9nxeB1-KE0-Kf1AX4YSLlzikK/view?usp=drive_link)

---

## ✅ Conclusion

This project demonstrates the successful application of **YOLOv8-based object detection** for analyzing social preferences in adult zebrafish under controlled laboratory conditions. The use of **deep learning**, **region-based tracking**, and **Roboflow-powered annotation** enabled accurate detection and behavioral inference from video data.

By integrating tools like **Ultralytics**, **OpenCV**, and **Google Colab**, we established a reproducible and scalable workflow suitable for biological research and motion analysis.

---

## 🤝 Contributing

Contributions are welcome! Whether it’s improving detection accuracy, adding more behavioral metrics, or enhancing visualization — feel free to fork the repo and submit a pull request.

If you're a researcher working with fish behavior or similar video-based datasets, we’d love to collaborate!

---

## 📬 Contact & Support

For queries, collaborations, or suggestions:

- 📧 Email: ds6918821@gmail.com
- 🧠 GitHub Issues: Feel free to open an issue for any bug or improvement idea

---

## ⭐ Acknowledgements

- **Roboflow** for dataset management and training
- **Ultralytics YOLOv8** for real-time detection
- **Google Colab** for free training/inference
- Special thanks to our lab team and dataset annotators!

---

> 🔍 _This work is a part of our ongoing research on automated behavioral analysis of aquatic life._

