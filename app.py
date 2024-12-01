import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify, render_template
import requests
from sklearn.preprocessing import LabelEncoder
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# Retrieve the API key
api_key = os.getenv("API_KEY")

# Set up environment to disable oneDNN optimization warnings (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Initialize Flask app
app = Flask(__name__)

# Path to the saved model and label encoder
MODEL_PATH = "painting_recognition_model_resnet50_finetuned_1.h5"
LABEL_CLASSES_PATH = "label_classes_resnet50_1.npy"

# Load the saved model
model = load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load the label encoder
label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
        img_array = img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return tf.keras.applications.resnet.preprocess_input(img_array)  # Preprocess for ResNet
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to fetch Google search results
def fetch_google_search_results(query, api_key, cx):
    encoded_query = quote(query)
    search_url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={api_key}&cx={cx}"
    try:
        response = requests.get(search_url)
        data = response.json()
        links = []
        if "items" in data:
            for item in data["items"]:
                title = item["title"]
                link = item["link"]
                snippet = item["snippet"]
                links.append({"title": title, "link": link, "snippet": snippet})
        return links
    except Exception as e:
        print(f"Error fetching Google search results: {e}")
        return []

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    image_path = os.path.join("static", file.filename)
    file.save(image_path)
    
    # Preprocess the image and make prediction
    image = load_and_preprocess_image(image_path)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Make prediction
    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # Fetch additional information about the painting using Google Search
    query = predicted_class + " by Edvard Munch"
    google_results = fetch_google_search_results(query, api_key, cx="20b79b15ec0d844c2")
    
    # Render the result on the homepage template
    return render_template('index.html', predicted_class=predicted_class, google_results=google_results)

if __name__ == '__main__':
    app.run(debug=True)
