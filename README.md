# Computer Vision Mini Projects Motion Detection & Segmentation

A compact and practical collection of **computer vision mini-projects** focused on:

-  Video Analysis  
-  Motion Detection & Monitoring  
-  Object & People Segmentation  

These projects demonstrate real-world pipelines using **Python + OpenCV**, with optional use of deep learning models like **Mask R-CNN**.

---

## 📌 Overview

This repository showcases **modular and independent computer vision workflows**, where each project can be executed separately and easily extended.

It is designed for:
- Learning computer vision fundamentals  
- Building real-world applications  
- Rapid prototyping for AI systems  

---

## 📂 Projects

| Project | Description | Main Script |
|--------|------------|-------------|
| 🛑 Burglar Detection | Motion-based security system with alerts and recording | `Burglar detection/burglar detection.py` |
| 🎥 Live Object Segmentation | Real-time object segmentation on live video streams | `segment objects from a live video/objects segmentation.py` |
| 🧍 People Segmentation | Extract and segment people from images and videos | `segment people/pepole segmentation.py` |

---

## 🗂️ Repository Structure

```text
.
├── README.md
├── Burglar detection/
│   ├── burglar detection.py
│   ├── burglar_detection_results/
│   ├── emergency sound effect/
│   │   └── Sound Effects Emergency Alarm.mp3
│   └── videos/
│       ├── Burglar Robber Break Door Lock Entering Stock Footage Video 100.mp4
│       └── Burglar Robber Break Door Lock Entering Stock Footage Video 100.mkv
├── segment objects from a live video/
│   └── objects segmentation.py
└── segment people/
  ├── pepole segmentation.py
  ├── final result.png
  ├── image/
  │   └── group walking.jpg
  └── masked images/
    ├── detected_people.jpg
    ├── full_comparison.jpg
    └── masked_people.jpg
```

---

## ⚙️ Key Features

-  Real-time video processing  
-  Motion detection using frame differencing  
-  Contour-based object detection  
-  Instance segmentation workflows (Mask R-CNN)  
-  Alarm / alert system integration  
-  Saving processed videos and output masks  
-  Configurable thresholds and parameters  

---

## 🧠 How It Works (Technical Insight)

### 🔹 Motion Detection (Burglar Detection)
- Frame differencing between consecutive frames  
- Grayscale conversion + Gaussian blur  
- Thresholding to isolate motion  
- Contour detection to identify moving objects  

### 🔹 Segmentation (Mask R-CNN)
- Region Proposal Network (RPN)  
- ROI Align for precise feature extraction  
- Pixel-wise mask prediction  
- Bounding box + class labeling  

---

## 📦 Requirements

Make sure you have:

- Python 3.7+
- OpenCV
- NumPy

Optional (depending on project):
- imutils  
- Pytorch or TensorFlow / Keras (for Mask R-CNN)  
- Pretrained weights (COCO dataset)  

---

## 🚀 Setup

### 1. Clone the repository

    git clone https://github.com/samir-m0hamed/Computer-Vision-Mini-Projects-Motion-Detection-and-Segmentation.git
    cd projects

### 2. Install dependencies

    pip install opencv-python numpy

### 3. Install additional dependencies (if needed)

    pip install imutils tensorflow keras

---

## ▶️ How to Run

Run each script individually:

    # Burglar Detection
    python "Burglar detection/burglar detection.py"

    # Live Object Segmentation
    python "segment objects from a live video/objects segmentation.py"

    # People Segmentation
    python "segment people/pepole segmentation.py"

---

## 📤 Output

-  Burglar Detection  
  Saves processed videos inside:  
  `burglar_detection_results/`

-  People Segmentation  
  Outputs:
  - Masked images  
  - Processed results inside:  
    `segment people/`

-  Terminal Output  
  Each script logs:
  - Frame processing  
  - Detection status  
  - Runtime info  

---

## 🎯 Use Cases

-  Home Security Systems  
-  Surveillance & Monitoring  
-  Human Detection & Tracking  
-  Smart Video Processing  

---

## ⚠️ Notes

- The file name `pepole segmentation.py` is intentionally preserved as-is from the original project.  

- Burglar Detection is **NOT based on deep learning**, it uses traditional computer vision techniques.  

- Video encoding and FPS may affect:
  - Frame count  
  - Playback smoothness  

---

## 🤝 Contributing

Contributions are welcome!

Steps:
1. Fork the repository  
2. Create a new branch  
3. Commit your changes  
4. Open a Pull Request  

---

## 📜 License

No license has been specified yet.  

---

## 👨‍💻 Author

**Samir Mohamed**   

**AI Engineer | Data Science & Computer Vision**
