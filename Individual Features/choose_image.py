import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageTk

# Load the pre-trained OCR model (ensure the model file is in the correct location)
model = load_model('ocr.h5')

# Function to process image and predict the handwritten digit
def preprocess_image(image_path):
    # Read the image using OpenCV in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to match the input size expected by the model (28x28 for MNIST-like models)
    img_resized = cv2.resize(img, (28, 28))

    # Normalize the image by dividing by 255.0
    img_normalized = img_resized / 255.0

    # Reshape the image for the model (model expects a batch of images)
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    # Predict the digit using the model
    prediction = model.predict(img_reshaped)

    # Get the predicted digit (the class with the highest probability)
    predicted_digit = np.argmax(prediction)
    
    return predicted_digit, img  # Return the predicted digit and original image

# Function to open the file explorer and choose an image
def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    if file_path:
        try:
            # Process and predict the handwritten digit from the image
            predicted_digit, original_img = preprocess_image(file_path)

            # Display the result in a new window
            result_window(predicted_digit, original_img)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the image: {str(e)}")

# Function to create a result window to display the predicted digit and original image
def result_window(predicted_digit, original_img):
    result_window = tk.Toplevel(window)
    result_window.title("Prediction Result")

    # Display the predicted digit
    label = tk.Label(result_window, text=f"Predicted Handwritten Digit: {predicted_digit}", font=("Arial", 20))
    label.pack(pady=10)

    # Convert the image from OpenCV (BGR) to RGB for Pillow
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Convert to a Pillow Image
    pil_image = Image.fromarray(original_img_rgb)

    # Resize the image for better display in the window
    pil_image_resized = pil_image.resize((200, 200))

    # Display the original image in the window
    img_tk = ImageTk.PhotoImage(pil_image_resized)
    img_label = tk.Label(result_window, image=img_tk)
    img_label.image = img_tk  # Keep a reference to the image
    img_label.pack(pady=20)

# Creating the main Tkinter window
window = tk.Tk()
window.title("Handwritten Digit OCR")

# Creating the 'Choose Image' button
choose_image_button = tk.Button(window, text="Choose Image", font=("Arial", 16), command=choose_image)
choose_image_button.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
