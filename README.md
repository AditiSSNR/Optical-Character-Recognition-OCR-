# Optical-Character-Recognition-OCR-
Performs OCR of digits. Has an interactive GUI. Accepts inputs through multiple modalities like image, live video, and drawing on canvas

## Project Overview
This project is an Optical Character Recognition system specifically designed for digit recognition. It features an interactive GUI that accepts input through multiple modalities, making it flexible and user-friendly.

## Key Features
1. Digit Recognition: Performs OCR (Optical Character Recognition) specifically for digits\
Uses machine learning to identify and classify handwritten or printed digits

2. Multiple Input Modalities: The system accepts input through three different methods:
 - Image Upload: Process static images containing digits
 - Live Video Feed: Real-time digit recognition from webcam or video stream
 - Canvas Drawing: Interactive drawing mode where users can draw digits directly on a canvas for immediate recognition

3. Interactive GUI:
 - User-friendly graphical interface for easy interaction
 - Eliminates the need for command-line usage
 - Intuitive controls for switching between input modes

## Main Files
1. gui-checkpoint.ipynb - Jupyter Notebook containing the interactive GUI implementation and main application logic
  
2. Individual Features - Directory containing separate implementations of individual OCR components, likely including:\
a. Image processing modules\
b. Model training/inference\
c. Drawing canvas implementation\
d. Video capture functionality

## Getting Started
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required ML libraries (includes: OpenCV, TensorFlow, NumPy, Keras)

### Running the Application
- Clone the repository
- Install required dependencies
- Open gui-checkpoint.ipynb in Jupyter Notebook
- Run the notebook cells to launch the interactive GUI
- Select your preferred input modality (image, video, or canvas)
- Input your digit(s) and receive OCR results

### Features by Input Method
1. Image Input\
Upload image files containing digits\
Process and recognize digits within the image\
Display results with confidence scores (if applicable)

2. Live Video Input\
Access webcam or video source\
Real-time digit recognition\
Continuous processing frame-by-frame

3. Canvas Drawing\
Draw digits on interactive canvas\
Real-time or on-demand recognition\
Clear canvas and retry functionality

## Development Notes
The project separates functionality into individual feature components, which suggests a modular design approach. This makes it easy to:
- Test individual components independently
- Reuse components in other projects
- Understand and modify specific functionality

## Potential Improvements
- Add confidence scores or probability displays for predictions
- Support for multi-digit recognition
- Integration with external APIs or services

## Author & License
Author: Aditi Singarajipura\
Repository: https://github.com/AditiSSNR/Optical-Character-Recognition-OCR-
