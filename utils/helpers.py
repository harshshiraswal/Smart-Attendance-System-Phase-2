"""
Utility functions for Face Detection Module
Phase 1 - Smart Attendance System
Developed during Vocational Training at OLF
"""

import math
import datetime
import cv2
import numpy as np
import os


def calculate_face_confidence(face_distance, face_match_threshold=0.6):
    """
    Calculate confidence percentage for face matches
    
    Args:
        face_distance: Distance from known face encoding
        face_match_threshold: Threshold for considering a match (0.6 default)
    
    Returns:
        str: Confidence percentage as string with % symbol
    """
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * 
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def setup_directories():
    """Create necessary directories if they don't exist"""
    from config import KNOWN_FACES_DIR, LOG_DIR
    
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Directories verified:")
    print(f"- Known faces: {KNOWN_FACES_DIR}")
    print(f"- Logs: {LOG_DIR}")


def log_attendance(name, log_path):
    """
    Log attendance with timestamp to file
    
    Args:
        name: Name of the recognized person
        log_path: Path to log file
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f'{name}, {timestamp}, Attendance marked!\n'
    
    try:
        with open(log_path, 'a') as f:
            f.write(log_entry)
        print(f"✓ Attendance logged for: {name} at {timestamp}")
    except Exception as e:
        print(f"✗ Error logging attendance: {e}")


def draw_face_annotations(frame, face_locations, face_names, scale_factor=4):
    """
    Draw bounding boxes and names on detected faces
    
    Args:
        frame: Original video frame
        face_locations: List of face location coordinates (top, right, bottom, left)
        face_names: List of names corresponding to faces
        scale_factor: Factor to scale coordinates back to original size
    
    Returns:
        frame: Frame with drawn annotations
    """
    from config import BBOX_COLOR, TEXT_COLOR, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale coordinates back to original frame size
        top = int(top * scale_factor)
        right = int(right * scale_factor)
        bottom = int(bottom * scale_factor)
        left = int(left * scale_factor)

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), BBOX_COLOR, 2)
        
        # Draw background rectangle for name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), 
                     BBOX_COLOR, cv2.FILLED)
        
        # Draw name and confidence text
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                   TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    
    return frame


def validate_known_faces(known_faces_dir):
    """
    Validate known faces directory and images
    
    Args:
        known_faces_dir: Path to known faces directory
    
    Returns:
        list: List of valid image files
    """
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = []
    
    if not os.path.exists(known_faces_dir):
        print(f"✗ Known faces directory not found: {known_faces_dir}")
        return image_files
    
    for file in os.listdir(known_faces_dir):
        if file.lower().endswith(valid_extensions):
            image_files.append(file)
    
    print(f"✓ Found {len(image_files)} valid face images")
    return image_files