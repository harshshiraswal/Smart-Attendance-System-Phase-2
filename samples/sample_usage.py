"""
Sample usage and testing script for Face Detection Module
Phase 1 - Smart Attendance System
"""

from face_detection import FaceRecognitionSystem
import cv2
import os


def test_system():
    """Test the face recognition system with sample functionality"""
    print("🧪 Testing Face Recognition System...")
    print("This script demonstrates the system functionality.")
    
    try:
        # Initialize system
        system = FaceRecognitionSystem()
        
        print("\n✅ System initialized successfully!")
        print(f"📊 Known faces loaded: {len(system.known_face_names)}")
        
        if system.known_face_names:
            print("👤 Known persons:")
            for name in system.known_face_names:
                print(f"  - {name}")
        
        # Ask user if they want to start real-time recognition
        response = input("\n🎥 Start real-time face recognition? (y/n): ")
        if response.lower() == 'y':
            system.run_recognition()
        else:
            print("🔚 Test completed without starting camera.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")


def check_requirements():
    """Check if all required directories and files exist"""
    print("🔍 Checking system requirements...")
    
    required_dirs = ['utils', 'known_faces', 'logs', 'samples']
    required_files = ['main.py', 'face_detection.py', 'config.py', 'requirements.txt']
    
    all_ok = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory found: {dir_name}")
        else:
            print(f"❌ Directory missing: {dir_name}")
            all_ok = False
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File found: {file_name}")
        else:
            print(f"❌ File missing: {file_name}")
            all_ok = False
    
    return all_ok


if __name__ == '__main__':
    print("=" * 50)
    print("Smart Attendance System - Sample Usage Script")
    print("Phase 1: Face Detection Module")
    print("=" * 50)
    
    # First check requirements
    if check_requirements():
        print("\n✅ All requirements satisfied!")
        test_system()
    else:
        print("\n❌ Please ensure all required files and directories exist.")
        print("💡 Run the system using: python main.py")