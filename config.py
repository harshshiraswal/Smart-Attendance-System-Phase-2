"""
Configuration settings for Face Detection Module
Phase 1 - Smart Attendance System
Developed during Vocational Training at OLF
"""

import os
import cv2

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, 'known_faces')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'attendance_log.txt')

# Face recognition settings
FACE_MATCH_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 97.0
FRAME_SCALE_FACTOR = 0.25  # For performance optimization

# Camera settings
CAMERA_INDEX = 0  # 0 for default camera, 1 for external camera
CAMERA_API_PREFERENCE = cv2.CAP_DSHOW  # For external webcam on Windows

# Display settings
DISPLAY_WINDOW_NAME = 'Smart Attendance System - Phase 1'
BBOX_COLOR = (0, 0, 255)  # Red color for bounding boxes
TEXT_COLOR = (255, 255, 255)  # White color for text
TEXT_FONT = cv2.FONT_HERSHEY_DUPLEX
TEXT_SCALE = 0.8
TEXT_THICKNESS = 1

# Performance settings
PROCESS_EVERY_N_FRAME = 2  # Process every other frame for better performance