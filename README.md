# 🤟 Sign Language Translator

> A real-time sign language gesture recognition system using CNN and webcam technology to translate ASL gestures into text, making communication accessible for deaf and hearing communities.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)](https://opencv.org)

## 🎯 Overview

This project implements a real-time American Sign Language (ASL) translator that uses computer vision and deep learning to recognize hand gestures captured via webcam and convert them into readable text. The system combines MediaPipe for hand tracking, custom CNN architecture for gesture classification, and Flask for web deployment.

### ✨ Key Features

- **Real-time Recognition**: Instant ASL gesture detection 
- **Web-based Interface**: Accessible through any modern web browser
- **High Accuracy**: Achieves 90%+ recognition accuracy on ASL alphabet
- **Educational Mode**: Interactive learning features for ASL practice
- **No Installation Required**: Direct access via web interface


## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Webcam Input  │───▶│  MediaPipe Hand  │───▶│ CNN Classifier  │
│                 │    │     Tracking     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│  Text Output    │◀───│  Flask Web App   │◀────────────┘
│                 │    │                  │
└─────────────────┘    └──────────────────┘
```



## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam access
- Modern web browser (Chrome, Firefox, Safari)


## 🎮 Usage

### Basic Usage

1. **Access the Web Interface**: Open your browser to `http://localhost:5000`
2. **Allow Camera Access**: Grant webcam permissions when prompted
3. **Position Your Hand**: Place your hand in the camera frame
4. **Make ASL Gestures**: Form clear ASL alphabet letters
5. **View Results**: See real-time translation on screen



