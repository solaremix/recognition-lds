import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

# Load pre-trained digit recognition model (you can train your own or use a pre-trained model)
model = load_model('model.h5')


def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and extract digits
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_roi = thresh[y:y+h, x:x+w]

        # Resize the digit image to 28x28 pixels (MNIST model input size)
        digit_roi_resized = cv2.resize(
            digit_roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize pixel values to be between 0 and 1
        digit_roi_normalized = digit_roi_resized / 255.0

        # Reshape to match the model's input shape
        digit_input = digit_roi_normalized.reshape(1, 28, 28, 1)

        digits.append(digit_input)

    return digits


def recognize_digits(image_path):
    digits = preprocess_image(image_path)

    predictions = []
    for digit_input in digits:
        # Make predictions using the pre-trained model
        prediction = np.argmax(model.predict(digit_input), axis=1)[0]
        predictions.append(prediction)

    return predictions


if __name__ == "__main__":
    # Replace with the path to your image
    image_path = 'img\\138549320240253206.jpg'
    predictions = recognize_digits(image_path)

    print("Recognized Digits:", predictions)
