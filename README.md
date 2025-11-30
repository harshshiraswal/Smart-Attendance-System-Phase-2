# Smart Attendance System - Phase 2: GUI-Based Attendance Management

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange)
![Face Recognition](https://img.shields.io/badge/Face--Recognition-1.3-green)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-yellow)

## 📋 Project Overview
This is **Phase 2** of the **Smart Attendance System Using Facial Recognition** project. This phase implements a complete GUI-based attendance management system with real-time face recognition and registration capabilities.

## 🎯 Features
- **Real-time Face Detection & Recognition** using webcam
- **GUI Interface** with three main functions:
  - Mark Attendance (Check-in)
  - Leaving from Office (Check-out)
  - New Registration
- **Employee ID-based Registration**
- **Automatic Attendance Logging** with timestamps
- **Pickle-based Face Encoding Storage**
- **Multiple Directory Support** (db, logs, known_faces)

## 🏗️ Project Structure
Smart-Attendance-System-Phase2/
├── main.py # Main GUI application
├── db/ # Database for face encodings (.pickle files)
├── logs/ # Attendance logs
├── known_faces/ # Original face images (reference)
├── requirements.txt # Project dependencies
└── README.md # Project documentation


## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (internal or external)

### Usage
Starting the Application
```bash
python main.py