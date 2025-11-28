"""
Face Detection and Recognition Module
Phase 1 - Smart Attendance System
Developed during Vocational Training at OLF
"""

import face_recognition
import os
import sys
import cv2
import numpy as np
from utils.helpers import (
    calculate_face_confidence, 
    log_attendance, 
    draw_face_annotations,
    setup_directories,
    validate_known_faces
)
from config import *


class FaceRecognitionSystem:
    """
    Main class for face detection and recognition system
    Handles face encoding, recognition, and attendance logging
    """
    
    def __init__(self):
        # Initialize face tracking variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.previously_recognized = []
        self.frame_count = 0
        
        # Setup directories and encode known faces
        setup_directories()
        self.encode_known_faces()
        
        print("=" * 50)
        print("Face Recognition System Initialized Successfully!")
        print(f"Loaded {len(self.known_face_names)} known faces")
        print("=" * 50)

    def encode_known_faces(self):
        """
        Encode all known faces from the known_faces directory
        Converts face images to encodings for comparison
        """
        try:
            # Validate known faces directory
            image_files = validate_known_faces(KNOWN_FACES_DIR)
            
            if not image_files:
                print("⚠️  No face images found in known_faces directory")
                print("Please add face images to proceed.")
                return

            for image_file in image_files:
                image_path = os.path.join(KNOWN_FACES_DIR, image_file)
                
                # Load and encode face
                face_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_image)
                
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    # Store name without file extension
                    name = os.path.splitext(image_file)[0]
                    self.known_face_names.append(name)
                    print(f"✓ Encoded face: {name}")
                else:
                    print(f"⚠️  No face found in: {image_file}")
            
            if not self.known_face_encodings:
                print("❌ Error: No faces could be encoded from known_faces directory")
                print("Please check image quality and ensure faces are visible.")
                
        except Exception as e:
            print(f"❌ Error encoding known faces: {e}")
            sys.exit(1)

    def process_frame(self, frame):
        """
        Process a single frame for face detection and recognition
        
        Args:
            frame: Video frame to process
            
        Returns:
            tuple: (processed_frame, recognized_names)
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, 
                                fy=FRAME_SCALE_FACTOR)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        # Find all faces in current frame
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(
            rgb_small_frame, self.face_locations)
        
        self.face_names = []
        recognized_names = []

        for face_encoding in self.face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=FACE_MATCH_THRESHOLD)
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            
            # Find best match
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else 0
            name = "Unknown"
            confidence = "??%"
            
            if len(face_distances) > 0:
                confidence = calculate_face_confidence(
                    face_distances[best_match_index], FACE_MATCH_THRESHOLD)
                confidence_value = float(confidence[:-1])

                # Check if match meets confidence threshold
                if confidence_value > CONFIDENCE_THRESHOLD and matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    recognized_names.append(name)
                    
                    # Log attendance if not previously recognized in this session
                    if name not in self.previously_recognized:
                        log_attendance(name, LOG_FILE)
                        self.previously_recognized.append(name)

            display_name = f'{name} ({confidence})'
            self.face_names.append(display_name)

        # Draw annotations on the frame
        processed_frame = frame.copy()
        if self.face_locations and self.face_names:
            processed_frame = draw_face_annotations(
                processed_frame, self.face_locations, self.face_names, 
                int(1/FRAME_SCALE_FACTOR)
            )
        
        return processed_frame, recognized_names

    def run_recognition(self):
        """
        Main method to run face recognition system
        Handles camera initialization and main processing loop
        """
        print("🚀 Starting face recognition system...")
        print("📷 Initializing camera...")
        
        video_capture = cv2.VideoCapture(CAMERA_INDEX, CAMERA_API_PREFERENCE)

        if not video_capture.isOpened():
            print("❌ Error: Could not access camera")
            print("Please check:")
            print("1. Camera is connected")
            print("2. No other application is using the camera")
            print("3. Camera drivers are installed")
            sys.exit(1)

        print("✅ Camera initialized successfully")
        print("🎯 Press 'q' to quit the application")
        print("-" * 50)

        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break

                # Process every other frame to improve performance
                self.frame_count += 1
                if self.frame_count % PROCESS_EVERY_N_FRAME == 0:
                    processed_frame, recognized_names = self.process_frame(frame)
                    self.process_current_frame = True
                else:
                    processed_frame = frame
                    self.process_current_frame = False

                # Display the resulting frame
                cv2.imshow(DISPLAY_WINDOW_NAME, processed_frame)

                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Stopping face recognition system...")
                    break

        except KeyboardInterrupt:
            print("\n⏹️  System interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Clean up resources
            video_capture.release()
            cv2.destroyAllWindows()
            print("✅ Camera resources released")
            print("🎉 Face recognition system stopped successfully.")
            print(f"📊 Total faces recognized this session: {len(self.previously_recognized)}")


if __name__ == '__main__':
    try:
        face_system = FaceRecognitionSystem()
        face_system.run_recognition()
    except Exception as e:
        print(f"❌ Failed to start face recognition system: {e}")