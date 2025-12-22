import cv2
import numpy as np

# Load the pre-trained face detection model (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display "Hello!!!" and a smiley emoji near the face
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Hello!!!", (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display a smiley emoji (use a simple smiley character as text or use a custom image)
        smiley = ":)"  # You can use a more advanced emoji display here, but for simplicity, we'll use text
        cv2.putText(frame, smiley, (x + w - 30, y + h + 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the resulting frame
    cv2.imshow('Face Detection - Press "q" to exit', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
