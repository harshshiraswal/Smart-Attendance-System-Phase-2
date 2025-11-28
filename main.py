"""
Main entry point for Smart Attendance System - Phase 1
Face Detection and Recognition Module
Developed during Vocational Training at OLF
"""

from face_detection import FaceRecognitionSystem
import argparse
import sys


def display_banner():
    """Display application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         SMART ATTENDANCE SYSTEM - PHASE 1                    ║
    ║         Face Detection and Recognition Module                ║
    ║                                                              ║
    ║         Developed during Vocational Training at OLF          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """
    Main function to run the face recognition system
    """
    parser = argparse.ArgumentParser(
        description='Smart Attendance System - Phase 1: Face Detection Module'
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (for testing)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (0 for default, 1 for external)')
    
    args = parser.parse_args()
    
    # Display banner
    display_banner()
    
    print("🚀 Initializing Smart Attendance System...")
    print("📍 Phase 1: Face Detection Module")
    print("🏢 Developed during Vocational Training at OLF")
    print("=" * 60)
    
    if args.demo:
        print("🔧 Running in DEMO mode...")
        # Demo mode specific configurations can be added here
    
    if args.camera != 0:
        print(f"📷 Using camera index: {args.camera}")
    
    try:
        # Initialize and run face recognition system
        print("🔄 Starting face recognition system...")
        face_system = FaceRecognitionSystem()
        face_system.run_recognition()
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)
    finally:
        print("\n👋 Thank you for using Smart Attendance System!")
        print("📍 Phase 1 - Face Detection Module Complete")


if __name__ == '__main__':
    main()