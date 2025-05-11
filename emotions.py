from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Ensure the model file path is correct
model_path = 'emotion_model.h5'

# Load the model (with better error handling)
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Confidence threshold and emotion dictionary
detection_confidence = 0.3  # Lowered for better sensitivity
emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if model is None:
        return jsonify({'error': 'Emotion detection model is not available.'}), 500

    try:
        # Retrieve image data from POST request
        data = request.data
        image_data = base64.b64decode(data.split(b',')[1])
        np_img = np.frombuffer(image_data, np.uint8)
        
        # Decode the image using OpenCV
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)  # Adjust to IMREAD_COLOR if model uses RGB
        
        # Resize image and normalize
        img_resized = cv2.resize(img, (48, 48))
        img_resized = img_resized / 255.0
        img_resized = img_resized.reshape(1, 48, 48, 1)  # Adjust to (1, 48, 48, 3) if RGB

        # Predict emotion
        predictions = model.predict(img_resized, verbose=0)

        # Debugging: Print raw predictions
        print(f"Predictions (Raw): {predictions}")

        emotion_index = int(np.argmax(predictions))
        confidence = float(predictions[0][emotion_index])

        if confidence >= detection_confidence:
            emotion = emotion_dict.get(emotion_index, "Unknown")
        else:
            emotion = "Low Confidence"

        return jsonify({'emotion': emotion, 'confidence': round(confidence, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
