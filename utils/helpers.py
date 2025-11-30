"""
Utility functions for Face Detection Module
Phase 1 - Smart Attendance System
Developed during Vocational Training at OLF
"""

import os
import pickle
import tkinter as tk
from tkinter import messagebox
import face_recognition
import cv2

def get_button(window, text, color, command, fg='white', width=20):
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=width,
        font=('Helvetica bold', 16)
    )
    return button

def get_img_label(window):
    label = tk.Label(window, bg="black")
    label.grid(row=0, column=0)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text, bg="black", fg="white")
    label.config(font=("sans-serif", 16), justify="left")
    return label

def get_entry_text(window):
    inputtxt = tk.Text(
        window,
        height=2,
        width=20,
        font=("Arial", 16)
    )
    return inputtxt

def msg_box(title, description):
    messagebox.showinfo(title, description)

def recognize(img, db_path):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted(os.listdir(db_path))
    match = False
    j = 0
    
    while not match and j < len(db_dir):
        path_ = os.path.join(db_path, db_dir[j])
        file = open(path_, 'rb')
        embeddings = pickle.load(file)
        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        j += 1

    if match:
        return db_dir[j-1][:-7]  # Remove .pickle extension
    else:
        return 'unknown_person'

def draw_face_boxes(image, face_locations, names):
    """Draw bounding boxes around faces and display names"""
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Draw box around face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label with name
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    return image