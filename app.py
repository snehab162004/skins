from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
from prediction import medications  # Import the medications dictionary
import base64

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = "skin_disease_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' is missing.")
model = tf.keras.models.load_model(model_path)

# Ensure the uploads directory exists
uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# Define class names
class_names = [
    "acne_level_0", "acne_level_1", "acne_level_2", 
    "atopic_dermatitis", "melanoma", "normal", 
    "psoriasis", "tinea", "not_skin"
]

# Define a function to preprocess the image and make predictions
def predict_image(img_path, model, class_names, img_size=128):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))  # Convert to Python float
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]

        # Get the medication details for the predicted class
        medication_info = medications.get(predicted_class, "No specific medication available.")

        # Return the prediction results
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "medication": medication_info,
            "predictions": predictions.tolist()  # Convert to list
        }
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime.html')
def realtime():
    return render_template('realtime.html')


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """
    Handle image upload and make predictions.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        img_path = os.path.join(uploads_dir, file.filename)
        file.save(img_path)
        try:
            result = predict_image(img_path, model, class_names)
            return jsonify(result), 200
        finally:
            os.remove(img_path)

@app.route('/realtime-predict', methods=['POST'])
def realtime_predict():
    """
    Handle real-time image prediction.
    """
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode the base64 image
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Save the image to a temporary path
        temp_img_path = os.path.join(uploads_dir, "realtime_image.jpg")
        img.save(temp_img_path)

        # Predict using the model
        result = predict_image(temp_img_path, model, class_names)

        # Clean up the temporary file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        return jsonify(result), 200
    except Exception as e:
        # Log the error to identify the issue
        print(f"Error during real-time prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
