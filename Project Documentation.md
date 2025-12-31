## Project Overview
Repository: AditiSSNR/Optical-Character-Recognition-OCR-
This is a comprehensive Handwritten Digit Recognition system built with Python that uses deep learning to recognize digits (0-9). It provides an interactive GUI with three distinct input modalities, making it accessible and user-friendly.

## System Architecture
### Core Technology Stack
- Deep Learning Framework: TensorFlow/Keras
- Image Processing: OpenCV (cv2)
- GUI Framework: Tkinter
- Image Handling: PIL (Python Imaging Library)
- ML Libraries: scikit-learn, NumPy
- Pre-trained Model: MNIST-like CNN model 

### Main Components
gui.py
├── Window Setup (Tkinter)
├── Image Input Module
├── Canvas Drawing Module
├── Camera/Video Capture Module
└── Model Loading & Prediction Engine

## Features & Functionality
### 1. Image Upload & Recognition
- Function: choose_image() and preprocess_image()
Allows users to upload image files containing handwritten digits and get predictions.
Supported Formats: PNG, JPG, JPEG, BMP, TIFF

- Processing Pipeline:
Load image in grayscale using OpenCV
Resize to 28×28 pixels (MNIST standard format)
Normalize pixel values (divide by 255.0)
Reshape to model input format: (1, 28, 28, 1)
Pass through pre-trained neural network
Return predicted digit using argmax on probabilities
Display result in new window with original image

- Key Functions:
preprocess_image(image_path)
  ├─ Read image in grayscale
  ├─ Resize to 28x28
  ├─ Normalize (0-1 range)
  ├─ Reshape for model input
  └─ Return prediction & original image
choose_image()
  ├─ Open file dialog
  ├─ Call preprocess_image()
  └─ Display result_window()
result_window(predicted_digit, original_img)
  ├─ Create new Toplevel window
  ├─ Display predicted digit
  └─ Show resized original image (200x200)

### 2.Canvas Drawing Mode
- Function: draw_canvas()
Interactive drawing interface where users can draw digits on a canvas and get real-time predictions.
Features:
Black canvas (300×300 pixels)
Free-hand drawing with mouse
Real-time prediction after each stroke
Visual feedback with prediction display
Image enhancement (dilate & erode)

 -Drawing Controls:
Left Mouse Button: Draw on canvas
'C' Key: Clear canvas and reset prediction
'Q' Key: Quit drawing mode

- Processing Pipeline:
1. Initialize empty 300×300 black canvas
2. Mouse callback captures drawing movements
3. When mouse button released:
Copy canvas
Resize drawn digit to 28×28
Apply morphological operations (dilate/erode)
Flatten and reshape for model
Normalize image (divide by 255)
Get prediction from model
Extract predicted class (argmax)
4. Display prediction in separate window

- Key Processing Steps:
Canvas Drawing → Resize (28x28) → Dilate → Erode 
→ Flatten → Reshape (1,28,28,1) → Normalize 
→ Model Prediction → Display Result

### 3. Live Camera/Video Capture
- Function: camera_capture()
Real-time digit recognition from webcam feed with advanced features.
Features:
Face detection using Haar Cascades
Digit detection and recognition in video frames
Bounding box drawing around detected digits
Confidence threshold filtering (>80%)
Visual feedback with labels

- Processing Pipeline for Each Frame:
1.Capture frame from webcam
2.Convert BGR to grayscale
3. Apply Gaussian Blur (5×5 kernel)
4. Thresholding (OTSU's binary inverse)
5. Find contours in thresholded image
6. For each contour ≥20×20 pixels:
Extract ROI (Region of Interest)
Resize to 28×28
Normalize and reshape
Predict digit
Filter by confidence (>0.8)
Draw bounding box and label

- Face Detection Integration:
Detects faces in frame
Draws rectangles around faces
Displays "Hello!!!" greeting
Shows smiley emoji ":)"

- Controls:
'Q' Key: Quit camera mode

## Model Information
### Pre-trained Model: 
Architecture: Convolutional Neural Network (CNN)
Training Data: MNIST Dataset (70,000 handwritten digits)
Input Shape: (28, 28, 1) - 28×28 pixel grayscale images
Output: 10 classes (digits 0-9)
Output Format: Probability distribution across all 10 classes
Prediction Method: Argmax to get class with highest probability

### Input Normalization
All inputs are normalized to 0-1 range:
normalized_pixel = pixel_value / 255.0

### Model Architecture (CNN)
Layer Configuration
The model uses a standard CNN architecture optimized for digit recognition:
Input Layer
    ↓
Conv2D (32 filters, 3×3 kernel, ReLU activation)
    ↓
MaxPooling2D (2×2 window)
    ↓
Conv2D (64 filters, 3×3 kernel, ReLU activation)
    ↓
MaxPooling2D (2×2 window)
    ↓
Flatten Layer
    ↓
Dense (64 neurons, ReLU activation)
    ↓
Dense (10 neurons, Softmax activation) → Output

### Model Specifications
- Optimizer: Adam
Adaptive learning rate optimization
Efficient convergence

- Loss Function: Categorical Crossentropy
Standard for multi-class classification
Measures difference between predicted and actual probability distributions

- Metrics: Accuracy
Tracks percentage of correct predictions

- Training Configuration:
Epochs: 8
Batch Size: 64
Validation Data: Test set (10,000 images)

### Architecture Explaination
1. Two Convolutional Blocks - Extract spatial features from handwritten digits
2. Max Pooling - Reduce dimensions while preserving important information
3. Progressive Filter Increase (32→64) - Learn increasingly complex patterns
4. Fully Connected Layers - Combine learned features for classification
5. ReLU Activation - Introduce non-linearity for better feature learning
6. Softmax Output - Convert network output to probability distribution

## Training File Overview
The mlmodelfile.py file contains the complete pipeline for building, training, and saving the CNN model used in the OCR system.
### Data Preparation
- Dataset: MNIST
Total Samples: 70,000 handwritten digits
Training Set: 60,000 images
Test Set: 10,000 images
Image Size: 28×28 pixels
Classes: 10 (digits 0-9)
Grayscale: Single channel (black and white)
- Preprocessing Steps
1. Load MNIST Dataset
   └─ train_images: (60000, 28, 28)
   └─ train_labels: (60000,)
   └─ test_images: (10000, 28, 28)
   └─ test_labels: (10000,)

2. Normalize Images
   └─ Convert to float32
   └─ Divide by 255.0 → values in range [0, 1]

3. Add Channel Dimension
   └─ Reshape: (60000, 28, 28) → (60000, 28, 28, 1)
   └─ Reason: Keras CNN expects (height, width, channels)

4. One-Hot Encode Labels
   └─ Convert: [0, 1, 2, ...] → [[1,0,0,...], [0,1,0,...], ...]
   └─ Reason: Categorical crossentropy requires this format
   └─ Output shape: (60000, 10)
- Training Parameters
Optimizer:     'adam'
Loss Function: 'categorical_crossentropy'
Metrics:       ['accuracy']
Epochs:        8
Batch Size:    64
Validation:    Test set (10,000 images)
- Training process
For each epoch (1-8):
  ├─ Shuffle training data
  ├─ Divide into batches of 64 samples
  ├─ Forward pass through network
  ├─ Calculate loss and gradients
  ├─ Backpropagation with Adam optimizer
  ├─ Update weights and biases
  ├─ Evaluate on validation set
  └─ Display epoch metrics
- Expected Performance
Expected Performance
Training with 8 Epochs on MNIST:
Final Training Accuracy: ~99%
Final Validation Accuracy: ~98-99%
Model converges quickly due to simple dataset
- Model Saving
model.save('ocr.h5')
Saves in HDF5 format
Includes all weights, architecture, and training configuration
File size: ~1-2 MB
Can be loaded with: load_model('ocr.h5')

## GUI Layout
Main Window Specifications
Window Title: "Handwritten digit recognition"
Dimensions: 700×500 pixels

## Installation & Setup
- Prerequisites
Python 3.7+
- Required Libraries
pip install tensorflow keras
pip install opencv-python
pip install pillow
pip install numpy
pip install scikit-learn
pip install tkinter  # Usually included with Python

## Usage Guide
### Method 1: Image Recognition
- Click "1. Choose Image" button
- Select an image file containing a handwritten digit
- Application processes the image automatically
- New window displays:
Predicted digit
Original image (200×200 preview)

### Method 2: Canvas Drawing
- Click "2. Write on canvas" button
- OpenCV window opens with black canvas
- Draw a digit using left mouse button
- Release mouse button to trigger prediction
- Prediction displayed in separate "Prediction" window
- Controls:
'C': Clear canvas for new digit
'Q': Exit drawing mode

### Method 3: Live Camera Recognition
- Click "3. Predict" button
- Webcam starts capturing video
- System automatically:
Detects faces and displays greeting
Detects digits in the scene
Draws bounding boxes around recognized digits
Displays predicted digits
- Press 'Q' to exit camera mode

## Image Processing Techniques
### Preprocessing Steps
- For Image Upload:
Grayscale conversion
Resize to 28×28
Normalization (÷255)
Reshape for model input
- For Canvas Drawing:
Resize to 28×28
Dilation (kernel size: 1 iteration) - increases line thickness
Erosion (kernel size: 1 iteration) - removes noise
Normalization
- For Camera Capture:
Grayscale conversion
Gaussian Blur (5×5 kernel)
Binary thresholding with OTSU
Contour detection
ROI extraction and resizing
Normalization
- Key OpenCV Operations
cv2.resize()          # Resize images to 28x28
cv2.cvtColor()        # Color space conversion
cv2.GaussianBlur()    # Blur for noise reduction
cv2.threshold()       # Binary thresholding
cv2.findContours()    # Find digit boundaries
cv2.dilate()          # Enhance features
cv2.erode()           # Remove noise
cv2.rectangle()       # Draw bounding boxes
cv2.putText()         # Display text labels

## Prediction Details
- Confidence Filtering
Canvas Mode: Displays prediction after each stroke (no filtering)
Camera Mode: Only shows predictions with confidence > 0.8 (80%)
Image Mode: Shows prediction without filtering
- Output Format
All three modes return the predicted digit as an integer (0-9) using:
predicted_digit = np.argmax(prediction)
Where prediction is the model's probability distribution across 10 classes.

## Error Handling
- Image Processing Errors
try:
    predicted_digit, original_img = preprocess_image(file_path)
    result_window(predicted_digit, original_img)
except Exception as e:
    messagebox.showerror("Error", f"An error occurred: {str(e)}")
- Displays error dialog for:
Invalid image files
Unsupported formats
Processing failures

## Modular Components
The project includes an "Individual Features" directory containing separate implementations for:
- Image preprocessing module
- Canvas drawing module
- Video capture module
- Model prediction engine
- GUI components
These can be used independently or integrated as needed.

## Performance Considerations
- Processing Speed
Image Mode: <1 second per image
Canvas Mode: Real-time (depends on system)
Camera Mode: 30+ FPS typical (depends on resolution & contours)
- Resource Requirements
CPU: Minimal for inference
RAM: ~200-500 MB
GPU: Optional (not required for inference)

## Future Enhancements
- Multi-digit recognition (sequence of digits)
- Confidence score display for all modes
- Batch image processing
- Performance optimization for real-time processing
- Web interface using Flask/Django

## Troubleshooting
Common Issues
- Model file not found (ocr.h5)
Ensure ocr.h5 is in the same directory as gui.py
Check file path and permissions
- Webcam not detected
Verify camera permissions
Check device is not in use by another application
Try cv2.VideoCapture(1) instead of cv2.VideoCapture(0)
- Poor recognition accuracy
Ensure good lighting conditions
Draw digits clearly and centrally
Use images similar to MNIST dataset style
Check model training quality
- Tkinter not found
Linux: sudo apt-get install python3-tk
macOS: Included with Python
Windows: Included with Python

## Code Quality Notes
Well-structured with separate functions for each modality
Clear variable naming conventions
Comprehensive image preprocessing pipeline
Proper resource cleanup (e.g., cv2.destroyAllWindows())
Error handling for file operations
Uses modern TensorFlow/Keras API

##cAuthor & License
Author: Aditi Singarajipura
Repository: https://github.com/AditiSSNR/Optical-Character-Recognition-OCR-
