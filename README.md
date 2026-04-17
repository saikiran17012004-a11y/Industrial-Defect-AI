# 🏭 Project 1: Industrial Part Defect Detection & Segregation

This project is an AI-powered quality control system designed for manufacturing lines. Using **Computer Vision (YOLOv8)**, it automatically detects surface defects in industrial parts and provides real-time segregation logic via a web dashboard.

## 🚀 Live Demo
[Link to your Streamlit App (e.g., https://saikiran-defect-ai.streamlit.app)]

## 🛠️ Features
- **Real-time Detection:** Identifies cracks, holes, and surface irregularities.
- **Automated Segregation:** Logic to sort parts into 'OK' or 'REJECT' bins.
- **Web Dashboard:** Built with Streamlit for easy operator interaction.
- **High Accuracy:** Trained on custom industrial casting datasets.

## 🏗️ Tech Stack
- **Language:** Python
- **AI Model:** YOLOv8 (Ultralytics)
- **Web Framework:** Streamlit
- **Libraries:** OpenCV, PIL, NumPy

## 📂 Project Structure
```text
├── app.py                # Main Web Application code
├── best.pt               # Trained AI model weights
├── requirements.txt      # List of dependencies for deployment
├── train_data/           # Folder containing training images
└── README.md             # Project documentation
