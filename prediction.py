from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Updated `class_labels` dictionary to include subcategories
class_labels = {
    "acne_level_0": "Acne - Level 0",
    "acne_level_1": "Acne - Level 1",
    "acne_level_2": "Acne - Level 2",
    "atopic-dermatitis": "Atopic-Dermatitis",
    "melanoma": "Melanoma",
    "normal": "Normal Skin",
    "psoriasis": "Psoriasis",
    "tinea": "Tinea",
    "not-skin": "Not Skin"
}

# Updated `medications` dictionary for acne subcategories
medications = {
    "acne_level_0": "Mild acne: Use benzoyl peroxide or salicylic acid. Maintain good hygiene.",
    "acne_level_1": "Moderate acne: Use prescription topical treatments or consult a dermatologist.",
    "acne_level_2": "Severe acne: Seek professional dermatological care and consider oral medications.",
    "atopic-dermatitis":"use topical corticosteroids or calcineurin inhibitors, moisturize frequently.",
    "melanoma": "Seek immediate consultation with a dermatologist.",
    "normal": "Maintain a healthy skincare routine.",
    "psoriasis": "Use medicated shampoos or topical treatments. Consult a dermatologist.",
    "tinea": "Apply antifungal creams or consult a doctor for severe cases.",
    "not-skin": "No medication needed. This is not a skin image."
}

# Load the pre-trained model
model_path = 'skin_disease_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' is missing.")
model = load_model(model_path)

def predict_image(img_path, model, class_labels, medications, img_size=128):
    """
    Predict the disease class from an image.
    """
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        predicted_index = np.argmax(predictions)
        predicted_class = list(class_labels.keys())[predicted_index]

        # Classification logic
        if predicted_class != "not-skin":
            return {
                "image_path": img_path,
                "disease_name": class_labels[predicted_class],
                "confidence": confidence,
                "medication": medications[predicted_class]
                
            }
        else:
            return {
                "message": "Not skin or invalid input.",
                "confidence": confidence,
                "image_path": img_path
            }
    except Exception as e:
        return {"message": f"Error processing image: {e}"}
