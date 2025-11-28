# Known Faces Directory

## Purpose
This directory stores images of known individuals for face recognition.

## Adding New Faces
1. **Image Requirements:**
   - Clear, front-facing photos
   - Good lighting conditions
   - No sunglasses or hats
   - Minimum resolution: 640x480 pixels

2. **Naming Convention:**
   - Format: `firstname_lastname.jpg` or `firstname_lastname.png`
   - Examples: 
     - `john_doe.jpg`
     - `jane_smith.png`
     - `alex_wilson.jpeg`

3. **Supported Formats:**
   - JPEG (.jpg, .jpeg)
   - PNG (.png)

## Best Practices
- Use recent photos
- Ensure face is clearly visible
- Avoid group photos
- Use neutral expressions for better recognition
- Crop image to focus on face

## Example Structure
known_faces/
├── john_doe.jpg
├── jane_smith.png
├── alex_wilson.jpeg
└── sarah_johnson.jpg

## Note
- The system will automatically load all valid images on startup
- Each image should contain exactly one face for optimal results
- Image files are read during initialization only