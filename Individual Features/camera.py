import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('ocr.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box and extract ROI
    for contour in contours:
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Create a square ROI
        if w >= 20 and h >= 20:
            roi = thresh[y:y+h, x:x+w]
            roi = cv2.resize(roi, (28, 28))  # Resize to match MNIST input size
            roi = roi.reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape

            # Predict digit
            prediction = model.predict(roi)
            digit = np.argmax(prediction)

            # Check prediction confidence
            confidence = np.max(prediction)
            if confidence > 0.8:  # Adjust confidence threshold as needed
                # Draw bounding box and display predicted digit
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Handwritten Digit Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()