import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the pre-trained model
def load_pretrained_model(model_path='ocr.h5'):
    model = load_model(model_path)
    return model

# Step 2: Set up Canvas for Drawing and Prediction
def draw_canvas(model):
    # Create a black canvas
    canvas = np.zeros((300, 300), dtype=np.uint8)
    drawing = False
    last_x, last_y = None, None
    predicted_digit = None

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_x, last_y, canvas, predicted_digit
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, (last_x, last_y), (x, y), 255, 20)
                last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Predict the digit when the user stops drawing
            digit_image = canvas.copy()
            digit_image = cv2.resize(digit_image, (28, 28))  # Resize to match input shape of the model
            digit_image = cv2.dilate(digit_image, None, iterations=1)  # Enhance thickness
            digit_image = cv2.erode(digit_image, None, iterations=1)  # Clean up
            digit_image = digit_image.flatten().reshape(1, 28, 28, 1)  # Reshape for the model
            digit_image = digit_image.astype('float32') / 255  # Normalize image

            # Predict the digit
            predicted_digit = model.predict(digit_image)
            predicted_digit = np.argmax(predicted_digit, axis=1)  # Get the index of the predicted class
    
    # Set up the window and the mouse callback function
    cv2.namedWindow("Handwritten Digit Recognition")
    cv2.setMouseCallback("Handwritten Digit Recognition", draw)

    while True:
        cv2.imshow("Handwritten Digit Recognition", canvas)

        # If a digit is predicted, display the result in a separate window
        if predicted_digit is not None:
            output = np.zeros((400, 400), dtype=np.uint8)  # Increased window size
            cv2.putText(output, f"Predicted: {predicted_digit[0]}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Prediction", output)

        key = cv2.waitKey(1) & 0xFF

        # Check for user input to either clear the canvas or quit
        if key == ord('q'):  # Press 'Q' to quit
            break
        elif key == ord('c'):  # Press 'C' to clear the canvas
            canvas = np.zeros((300, 300), dtype=np.uint8)
            predicted_digit = None

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the pre-trained model from the .h5 file
    model = load_pretrained_model('ocr.h5')

    # Start the drawing and prediction loop
    draw_canvas(model)
