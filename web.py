# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load sign language recognition model
model = load_model('sign_language_model.h5')

@app.route('/api/translate', methods=['POST'])
def translate_sign_language():
    video_file = request.files['video']
    video_bytes = video_file.read()
    video_array = np.frombuffer(video_bytes, np.uint8)
    video_frames = cv2.imdecode(video_array, cv2.IMREAD_COLOR)

    # Preprocess video frames
    preprocessed_frames = []
    for frame in video_frames:
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        preprocessed_frames.append(frame)

    # Make predictions using the sign language recognition model
    predictions = model.predict(preprocessed_frames)
    translated_text = ''
    for prediction in predictions:
        translated_text += chr(np.argmax(prediction) + 32)

    return jsonify({'translated_text': translated_text})

@app.route('/api/lessons', methods=['GET'])
def get_lessons():
    lessons = [
        {'title': 'Lesson 1: Introduction to Sign Language'},
        {'title': 'Lesson 2: Basic Phrases'},
        {'