import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained deepfake detection CNN model.
# Ensure that 'deepfake_cnn_model.h5' is present in your project directory or update the path accordingly.
model = load_model('deepfake_cnn_model.h5')

def detect_deepfake(image_path):
    """
    Detect the deepfake probability for a given image using OpenCV and a CNN.
    
    This function:
      1. Reads the image from the provided path.
      2. Resizes it to the dimensions expected by the CNN (e.g., 224x224).
      3. Converts the image from BGR (OpenCV's default) to RGB.
      4. Normalizes pixel values to the range [0, 1].
      5. Expands dimensions to simulate a batch of size 1.
      6. Uses the CNN model to predict the probability that the image is fake.
      7. Returns the probability as an integer percentage.
    
    Parameters:
      image_path (str): The file path to the image.
    
    Returns:
      int: The deepfake probability (0 to 100).
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image from", image_path)
        return 0

    # Resize the image to the input size expected by the model (e.g., 224x224)
    image_resized = cv2.resize(image, (224, 224))
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to the range [0, 1]
    image_normalized = image_rgb.astype("float32") / 255.0
    
    # Expand dimensions to create a batch of size 1
    image_input = np.expand_dims(image_normalized, axis=0)
    
    # Predict the deepfake probability using the CNN model.
    # We assume the model outputs a single probability value.
    prediction = model.predict(image_input)
    fake_probability = prediction[0][0]  # Adjust indexing if your model's output is different.
    
    # Convert the probability to an integer percentage
    accuracy = int(fake_probability * 100)
    return accuracy
