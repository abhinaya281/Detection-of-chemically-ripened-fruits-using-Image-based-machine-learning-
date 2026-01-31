from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from pymongo import MongoClient
from datetime import datetime, timedelta
import bcrypt
from bson.objectid import ObjectId

app = Flask(name)

MongoDB Configuration

MONGODB_URI = "mongodb://localhost:27017/"  # Change to your MongoDB URI
DATABASE_NAME = "banana_detector"

Initialize MongoDB

try:
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]
users_collection = db['users']
predictions_collection = db['predictions']
print("✅ Connected to MongoDB successfully!")
except Exception as e:
print(f"❌ MongoDB connection error: {e}")
client = None
db = None
users_collection = None
predictions_collection = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

def load_banana_model():
global model
try:
print("🔍 Loading model...")
model_path = 'model/banana_classifier_best.h5'

if os.path.exists(model_path):  
        print(f"✅ Model found at: {model_path}")  
        model = load_model(model_path)  
        print("✅ Model loaded successfully!")  
    else:  
        print(f"❌ Model file not found at: {model_path}")  
        model = None  
except Exception as e:  
    print(f"❌ Error loading model: {e}")  
    model = None

def preprocess_image(image_path):
try:
img = Image.open(image_path)
if img.mode != 'RGB':
img = img.convert('RGB')
img = img.resize((150, 150))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
return img_array
except Exception as e:
print(f"❌ Image preprocessing error: {e}")
raise e

Serve the single page application for all routes

@app.route('/')
def serve_home():
return send_from_directory('.', 'index.html')

@app.route('/login')
def serve_login():
return send_from_directory('.', 'index.html')

@app.route('/signup')
def serve_signup():
return send_from_directory('.', 'index.html')

@app.route('/dashboard')
def serve_dashboard():
return send_from_directory('.', 'index.html')

@app.route('/path:filename')
def serve_static_files(filename):
return send_from_directory('.', filename)

@app.route('/health')
def health():
return jsonify({
'status': 'OK',
'model_loaded': model is not None,
'mongodb_connected': client is not None
})

User Registration

@app.route('/api/register', methods=['POST'])
def api_register():
if not client:
return jsonify({'error': 'Database not available'}), 500

try:  
    data = request.get_json()  
    name = data.get('name')  
    email = data.get('email')  
    password = data.get('password')  
      
    if not all([name, email, password]):  
        return jsonify({'error': 'All fields are required'}), 400  
      
    if users_collection.find_one({'email': email}):  
        return jsonify({'error': 'User with this email already exists'}), 400  
      
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())  
      
    user = {  
        'name': name,  
        'email': email,  
        'password': hashed_password,  
        'created_at': datetime.utcnow(),  
        'last_login': None  
    }  
      
    result = users_collection.insert_one(user)  
      
    return jsonify({  
        'success': True,  
        'message': 'User registered successfully',  
        'user_id': str(result.inserted_id)  
    })  
      
except Exception as e:  
    print(f"❌ Registration error: {e}")  
    return jsonify({'error': 'Registration failed'}), 500

User Login

@app.route('/api/login', methods=['POST'])
def api_login():
if not client:
return jsonify({'error': 'Database not available'}), 500

try:  
    data = request.get_json()  
    email = data.get('email')  
    password = data.get('password')  
      
    if not all([email, password]):  
        return jsonify({'error': 'Email and password are required'}), 400  
      
    user = users_collection.find_one({'email': email})  
    if not user:  
        return jsonify({'error': 'Invalid email or password'}), 401  
      
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):  
        return jsonify({'error': 'Invalid email or password'}), 401  
      
    # Update last login  
    users_collection.update_one(  
        {'_id': user['_id']},  
        {'$set': {'last_login': datetime.utcnow()}}  
    )  
      
    # Calculate user statistics  
    stats = calculate_user_statistics(user['_id'])  
      
    user_data = {  
        'user_id': str(user['_id']),  
        'name': user['name'],  
        'email': user['email'],  
        'stats': stats  
    }  
      
    return jsonify({  
        'success': True,  
        'message': 'Login successful',  
        'user': user_data  
    })  
      
except Exception as e:  
    print(f"❌ Login error: {e}")  
    return jsonify({'error': 'Login failed'}), 500

def calculate_user_statistics(user_id):
if not client:
return {'total_scans': 0, 'bananas_detected': 0, 'accuracy_rate': 0, 'today_scans': 0}

try:  
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)  
    today_end = today_start + timedelta(days=1)  
      
    # Total scans  
    total_scans = predictions_collection.count_documents({'user_id': user_id})  
      
    # Bananas detected  
    bananas_detected = predictions_collection.count_documents({  
        'user_id': user_id,  
        'is_banana': True  
    })  
      
    # Today's scans  
    today_scans = predictions_collection.count_documents({  
        'user_id': user_id,  
        'timestamp': {'$gte': today_start, '$lt': today_end}  
    })  
      
    # Calculate accuracy (average confidence)  
    accuracy_pipeline = [  
        {'$match': {'user_id': user_id}},  
        {'$group': {  
            '_id': None,  
            'avg_confidence': {'$avg': '$confidence'}  
        }}  
    ]  
      
    accuracy_result = list(predictions_collection.aggregate(accuracy_pipeline))  
    accuracy_rate = round(accuracy_result[0]['avg_confidence'], 2) if accuracy_result else 0  
      
    return {  
        'total_scans': total_scans,  
        'bananas_detected': bananas_detected,  
        'accuracy_rate': accuracy_rate,  
        'today_scans': today_scans  
    }  
      
except Exception as e:  
    print(f"❌ Statistics error: {e}")  
    return {'total_scans': 0, 'bananas_detected': 0, 'accuracy_rate': 0, 'today_scans': 0}

@app.route('/api/user/stats', methods=['GET'])
def get_user_stats():
if not client:
return jsonify({'error': 'Database not available'}), 500

try:  
    user_id = request.args.get('user_id')  
    if not user_id:  
        return jsonify({'error': 'User ID is required'}), 400  
      
    stats = calculate_user_statistics(ObjectId(user_id))  
    return jsonify(stats)  
      
except Exception as e:  
    print(f"❌ Stats error: {e}")  
    return jsonify({'error': 'Failed to get user stats'}), 500

Prediction Endpoint

@app.route('/predict', methods=['POST'])
def predict():
if model is None:
return jsonify({'error': 'Model not loaded'}), 500

if 'file' not in request.files:  
    return jsonify({'error': 'No file uploaded'}), 400  
  
file = request.files['file']  
if file.filename == '':  
    return jsonify({'error': 'No file selected'}), 400  
  
user_id = request.form.get('user_id')  
  
if file and allowed_file(file.filename):  
    filepath = None  
    try:  
        filename = secure_filename(file.filename)  
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  
        file.save(filepath)  
          
        processed_image = preprocess_image(filepath)  
        prediction = model.predict(processed_image, verbose=0)  
        raw_score = float(prediction[0][0])  
          
        print(f"🔍 Raw prediction score: {raw_score:.4f}")  
          
        is_banana = raw_score > 0.5  
          
        # Confidence calculation  
        if is_banana:  
            if raw_score >= 0.9:  
                confidence = 0.98  
            elif raw_score >= 0.8:  
                confidence = 0.95  
            elif raw_score >= 0.7:  
                confidence = 0.90  
            elif raw_score >= 0.6:  
                confidence = 0.85  
            else:  
                confidence = 0.80  
        else:  
            if raw_score <= 0.1:  
                confidence = 0.98  
            elif raw_score <= 0.2:  
                confidence = 0.95  
            elif raw_score <= 0.3:  
                confidence = 0.90  
            elif raw_score <= 0.4:  
                confidence = 0.85  
            else:  
                confidence = 0.80  
          
        confidence_percent = round(confidence * 100, 2)  
          
        print(f"🎯 Result: {'BANANA' if is_banana else 'NOT BANANA'} (confidence: {confidence_percent}%)")  
          
        # Convert image to base64  
        with open(filepath, "rb") as img_file:  
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')  
          
        # Store prediction in MongoDB  
        if user_id and client:  
            try:  
                prediction_record = {  
                    'user_id': ObjectId(user_id),  
                    'filename': filename,  
                    'is_banana': is_banana,  
                    'confidence': confidence_percent,  
                    'raw_score': raw_score,  
                    'timestamp': datetime.utcnow()  
                }  
                  
                predictions_collection.insert_one(prediction_record)  
                print(f"✅ Prediction saved for user {user_id}")  
                  
            except Exception as e:  
                print(f"⚠️ Failed to save prediction: {e}")  
          
        result = {  
            'is_banana': bool(is_banana),  
            'confidence': confidence_percent,  
            'image': f"data:image/jpeg;base64,{img_base64}"  
        }  
          
        return jsonify(result)  
          
    except Exception as e:  
        print(f"❌ Prediction error: {e}")  
        return jsonify({'error': f'Processing error: {str(e)}'}), 500  
      
    finally:  
        if filepath and os.path.exists(filepath):  
            try:  
                os.remove(filepath)  
            except Exception as e:  
                print(f"⚠️ Could not remove file: {e}")  
  
return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

Load model when app starts

load_banana_model()

if name == 'main':
print("🚀 Starting Banana Detection App...")
print("🌐 Access at: http://127.0.0.1:5000")
if client:
print("🗄️  MongoDB connected successfully!")
app.run(debug=True, port=5000)
