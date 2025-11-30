"""
Smart Attendance System - Phase 2: OpenCV-Based Version
Uses OpenCV's built-in face detection for maximum compatibility
"""

import os
import datetime
import pickle
import cv2
import numpy as np
from datetime import datetime
import time


class CV2AttendanceSystem:
    def __init__(self):
        # Initialize variables
        self.db_dir = './db'
        self.logs_dir = './logs'
        self.known_faces_dir = './known_faces'
        
        # Create necessary directories
        self.setup_directories()
        
        # Load known faces (using simple feature matching)
        self.known_face_data = []  # Store (name, features) pairs
        self.load_known_faces()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.mode = "MAIN"  # MAIN, REGISTRATION
        self.registration_emp_id = ""
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize feature detector
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        print("=" * 60)
        print("🚀 Smart Attendance System - Phase 2")
        print("📷 OpenCV-Based Version - No External Dependencies")
        print("=" * 60)
        print("🎯 Controls:")
        print("  A - Mark Attendance")
        print("  L - Leaving from Office") 
        print("  R - New Registration")
        print("  Q - Quit")
        print("=" * 60)

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        print("✓ Directories verified:")
        print(f"  - Database: {self.db_dir}")
        print(f"  - Logs: {self.logs_dir}")
        print(f"  - Known Faces: {self.known_faces_dir}")

    def extract_face_features(self, face_image):
        """Extract features from face image using SIFT"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and descriptors
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            return descriptors
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def compare_faces(self, descriptors1, descriptors2):
        """Compare two face descriptors"""
        if descriptors1 is None or descriptors2 is None:
            return 0.0
            
        try:
            # Match descriptors
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate similarity score
            similarity = len(good_matches) / max(len(descriptors1), len(descriptors2))
            return min(similarity * 100, 100.0)
            
        except Exception as e:
            return 0.0

    def load_known_faces(self):
        """Load known faces from database"""
        try:
            pickle_files = [f for f in os.listdir(self.db_dir) if f.endswith('.pickle')]
            
            for file in pickle_files:
                file_path = os.path.join(self.db_dir, file)
                with open(file_path, 'rb') as f:
                    face_data = pickle.load(f)
                    self.known_face_data.append((file[:-7], face_data))  # Remove .pickle
                    
            print(f"✓ Loaded {len(self.known_face_data)} known faces from database")
            
        except Exception as e:
            print(f"✗ Error loading known faces: {e}")

    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces

    def recognize_face(self, frame):
        """Recognize face in the given frame"""
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return "no_persons_found", 0.0
            
            # Use the largest face (assuming it's the main person)
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract features from current face
            current_descriptors = self.extract_face_features(face_roi)
            
            if current_descriptors is None:
                return "error", 0.0
            
            # Compare with known faces
            best_match = None
            best_confidence = 0.0
            
            for name, known_descriptors in self.known_face_data:
                confidence = self.compare_faces(current_descriptors, known_descriptors)
                if confidence > best_confidence and confidence > 40:  # 40% threshold
                    best_confidence = confidence
                    best_match = name
            
            if best_match:
                return best_match, best_confidence
            else:
                return "unknown_person", 0.0
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "error", 0.0

    def log_attendance(self, name, action):
        """Log attendance with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{name}, {timestamp}, {action}\n"
        
        log_file = os.path.join(self.logs_dir, "attendance_log.txt")
        
        try:
            with open(log_file, 'a') as f:
                f.write(log_entry)
            print(f"✓ Attendance logged: {name} - {action} at {timestamp}")
            return True
        except Exception as e:
            print(f"✗ Error logging attendance: {e}")
            return False

    def draw_main_interface(self, frame):
        """Draw the main interface with buttons"""
        display_frame = frame.copy()
        
        # Draw header
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(display_frame, "Smart Attendance System - Phase 2", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Draw buttons
        button_height = 80
        button_width = 400
        margin = 20
        start_y = 100
        
        # Button 1: Mark Attendance
        cv2.rectangle(display_frame, (margin, start_y), 
                     (margin + button_width, start_y + button_height), (0, 255, 0), -1)
        cv2.putText(display_frame, "Mark Attendance (Press A)", 
                   (margin + 10, start_y + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        
        # Button 2: Leaving from Office
        cv2.rectangle(display_frame, (margin, start_y + button_height + margin), 
                     (margin + button_width, start_y + 2*button_height + margin), (0, 0, 255), -1)
        cv2.putText(display_frame, "Leaving from Office (Press L)", 
                   (margin + 10, start_y + button_height + margin + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        # Button 3: New Registration
        cv2.rectangle(display_frame, (margin, start_y + 2*(button_height + margin)), 
                     (margin + button_width, start_y + 3*button_height + 2*margin), (200, 200, 200), -1)
        cv2.putText(display_frame, "New Registration (Press R)", 
                   (margin + 10, start_y + 2*(button_height + margin) + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw instructions
        cv2.putText(display_frame, "Press Q to quit", 
                   (frame.shape[1] - 200, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame

    def draw_registration_interface(self, frame):
        """Draw the registration interface"""
        display_frame = frame.copy()
        
        # Draw header
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(display_frame, "New Registration - Enter EMP_ID", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(display_frame, "Please, Enter EMP_ID:", (50, 120),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Draw EMP ID box
        cv2.rectangle(display_frame, (50, 150), (500, 220), (255, 255, 255), 2)
        cv2.putText(display_frame, self.registration_emp_id, (60, 190),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Draw buttons
        cv2.rectangle(display_frame, (50, 250), (250, 320), (0, 255, 0), -1)
        cv2.putText(display_frame, "Accept (Press Enter)", (60, 290),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.rectangle(display_frame, (300, 250), (500, 320), (0, 0, 255), -1)
        cv2.putText(display_frame, "Try Again (Press ESC)", (310, 290),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame

    def show_message(self, frame, title, message, duration=3000):
        """Show a message overlay"""
        display_frame = frame.copy()
        
        # Draw message box
        box_height = 150
        box_width = 600
        start_x = (frame.shape[1] - box_width) // 2
        start_y = (frame.shape[0] - box_height) // 2
        
        cv2.rectangle(display_frame, (start_x, start_y), 
                     (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (start_x, start_y), 
                     (start_x + box_width, start_y + box_height), (255, 255, 255), 2)
        
        # Draw title and message
        cv2.putText(display_frame, title, (start_x + 20, start_y + 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Split message if too long
        if len(message) > 40:
            parts = [message[i:i+40] for i in range(0, len(message), 40)]
            for i, part in enumerate(parts):
                cv2.putText(display_frame, part, (start_x + 20, start_y + 80 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(display_frame, message, (start_x + 20, start_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Smart Attendance System', display_frame)
        cv2.waitKey(duration)
        return frame

    def handle_registration_input(self, key):
        """Handle keyboard input during registration"""
        if key == 13:  # Enter key
            if self.registration_emp_id.strip():
                self.complete_registration()
            else:
                self.show_message(self.current_frame, "Error", "Please enter EMP_ID")
        elif key == 27:  # ESC key
            self.mode = "MAIN"
            self.registration_emp_id = ""
        elif key == 8:  # Backspace
            self.registration_emp_id = self.registration_emp_id[:-1]
        elif 48 <= key <= 57:  # Numbers 0-9
            self.registration_emp_id += chr(key)
        elif 65 <= key <= 90 or 97 <= key <= 122:  # Letters A-Z, a-z
            self.registration_emp_id += chr(key)

    def complete_registration(self):
        """Complete the registration process"""
        try:
            # Detect faces
            faces = self.detect_faces(self.current_frame)
            
            if len(faces) == 0:
                self.show_message(self.current_frame, "Error", "No face detected. Please try again.")
                return
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = self.current_frame[y:y+h, x:x+w]
            
            # Extract features
            descriptors = self.extract_face_features(face_roi)
            
            if descriptors is None:
                self.show_message(self.current_frame, "Error", "Could not extract face features.")
                return
            
            # Save features to database
            encoding_file = os.path.join(self.db_dir, f"{self.registration_emp_id}.pickle")
            with open(encoding_file, 'wb') as f:
                pickle.dump(descriptors, f)
            
            # Save original image for reference
            image_file = os.path.join(self.known_faces_dir, f"{self.registration_emp_id}.jpg")
            cv2.imwrite(image_file, self.current_frame)
            
            # Reload known faces
            self.load_known_faces()
            
            self.show_message(self.current_frame, "Success!", f"User {self.registration_emp_id} registered successfully!")
            self.mode = "MAIN"
            self.registration_emp_id = ""
            
        except Exception as e:
            self.show_message(self.current_frame, "Registration Error", f"Error: {str(e)}")

    def run(self):
        """Main application loop"""
        print("🚀 Starting face recognition system...")
        
        if not self.cap.isOpened():
            print("❌ Error: Could not access camera")
            return
        
        print("✅ Camera initialized successfully")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                self.current_frame = frame
                
                # Draw appropriate interface
                if self.mode == "MAIN":
                    display_frame = self.draw_main_interface(frame)
                elif self.mode == "REGISTRATION":
                    display_frame = self.draw_registration_interface(frame)
                
                # Show the frame
                cv2.imshow('Smart Attendance System', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("⏹️  System shutdown initiated...")
                    break
                
                if self.mode == "MAIN":
                    if key == ord('a') or key == ord('A'):
                        self.mark_attendance()
                    elif key == ord('l') or key == ord('L'):
                        self.leaving_office()
                    elif key == ord('r') or key == ord('R'):
                        self.mode = "REGISTRATION"
                        self.registration_emp_id = ""
                        print("📝 Registration mode activated")
                
                elif self.mode == "REGISTRATION":
                    self.handle_registration_input(key)
                        
        except KeyboardInterrupt:
            print("\n⏹️  System interrupted by user")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            self.cleanup()

    def mark_attendance(self):
        """Handle mark attendance"""
        name, confidence = self.recognize_face(self.current_frame)
        
        if name == "no_persons_found":
            self.show_message(self.current_frame, "No Face Detected", 
                            "No face detected. Please ensure your face is visible.")
        elif name == "unknown_person":
            self.show_message(self.current_frame, "Unknown User", 
                            "Unknown user. Please register or try again.")
        elif name == "error":
            self.show_message(self.current_frame, "Recognition Error", 
                            "Face recognition error. Please try again.")
        else:
            if self.log_attendance(name, "IN"):
                self.show_message(self.current_frame, "Welcome Back!", 
                                f"Welcome, {name}!\nConfidence: {confidence:.1f}%")

    def leaving_office(self):
        """Handle leaving office"""
        name, confidence = self.recognize_face(self.current_frame)
        
        if name == "no_persons_found":
            self.show_message(self.current_frame, "No Face Detected", 
                            "No face detected. Please ensure your face is visible.")
        elif name == "unknown_person":
            self.show_message(self.current_frame, "Unknown User", 
                            "Unknown user. Please register or try again.")
        elif name == "error":
            self.show_message(self.current_frame, "Recognition Error", 
                            "Face recognition error. Please try again.")
        else:
            if self.log_attendance(name, "OUT"):
                self.show_message(self.current_frame, "Goodbye!", 
                                f"Goodbye, {name}!\nConfidence: {confidence:.1f}%")

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera resources released")
        print("🎉 System shutdown complete")
        print("=" * 60)


def main():
    """Main function to start the application"""
    try:
        app = CV2AttendanceSystem()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")


if __name__ == "__main__":
    main()