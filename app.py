from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import os
import time
from collections import Counter

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models
def load_models():
    model_path = "model/asl_model_final.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found!")
    
    model = load_model(model_path)
    
    with open("model/label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    with open("model/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    return model, label_encoder, scaler

# Initialize components
model, le, scaler = load_models()
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# App state
state = {
    'word': '',
    'sentence': '',
    'last_prediction': 0,
    'predictions': []
}

# Configuration
CONFIG = {
    'confidence_threshold': 0.7,
    'prediction_interval': 0.2,
    'buffer_size': 5
}

def extract_landmarks(frame):
    """Extract hand landmarks from frame"""
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        coords = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return np.array(coords, dtype=np.float32).flatten()
    return None

def predict_letter(landmarks):
    """Predict letter from landmarks"""
    if landmarks is None or len(landmarks) != 63:
        return None, 0.0
    
    # Preprocess and predict
    scaled = scaler.transform(landmarks.reshape(1, -1))
    reshaped = scaled.reshape(1, 21, 3)
    prediction = model(reshaped, training=False)
    probs = prediction.numpy() if hasattr(prediction, 'numpy') else np.array(prediction)
    
    confidence = float(np.max(probs))
    if confidence >= CONFIG['confidence_threshold']:
        letter_idx = int(np.argmax(probs))
        letter = le.inverse_transform([letter_idx])[0]
        return letter, confidence
    
    return None, confidence

@socketio.on('stream')
def handle_stream(data):
    """Process video stream and predict ASL letters"""
    current_time = time.time()
    
    # Rate limiting
    if current_time - state['last_prediction'] < CONFIG['prediction_interval']:
        return
    
    try:
        # Decode image
        if not data or ',' not in data:
            return
            
        img_data = base64.b64decode(data.split(',')[1])
        img = Image.open(BytesIO(img_data))
        frame = np.array(img.resize((640, 480)))
        
        # Extract landmarks and predict
        landmarks = extract_landmarks(frame)
        letter, confidence = predict_letter(landmarks)
        
        # Update predictions buffer
        if letter:
            state['predictions'].append(letter)
            if len(state['predictions']) > CONFIG['buffer_size']:
                state['predictions'].pop(0)
            
            # Get most common letter from buffer
            if len(state['predictions']) >= 2:
                most_common = Counter(state['predictions']).most_common(1)[0][0]
                if not state['word'] or most_common != state['word'][-1]:
                    state['word'] += most_common
        
        state['last_prediction'] = current_time
        
        emit('prediction', {
            'letter': letter or '?',
            'word': state['word'],
            'sentence': state['sentence'],
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        emit('error', {'message': 'Processing failed'})

@socketio.on('commit_word')
def handle_commit_word():
    """Add current word to sentence"""
    if state['word']:
        if state['sentence']:
            state['sentence'] += ' '
        state['sentence'] += state['word']
        state['word'] = ''
        state['predictions'] = []
    
    emit('prediction', {
        'letter': '?',
        'word': state['word'],
        'sentence': state['sentence'],
        'confidence': 0
    })

@socketio.on('clear_all')
def handle_clear_all():
    """Clear all text"""
    state['word'] = ''
    state['sentence'] = ''
    state['predictions'] = []
    
    emit('prediction', {
        'letter': '?',
        'word': '',
        'sentence': '',
        'confidence': 0
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("🤟 ASL Recognition App")
    print("🚀 Server: http://localhost:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
