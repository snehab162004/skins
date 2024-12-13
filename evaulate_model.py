import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from data_agumentation import test_data  # Import test dataset iterator
from prediction import class_labels

# Load the pre-trained model
model_path = 'skin_disease_model.h5'
model = load_model(model_path)

# Initialize lists to store confidences
not_skin_confidences = []
other_confidences = []

# Iterate through the test dataset
for img_batch, label_batch in test_data:  # Use a proper iterator
    predictions = model.predict(img_batch)
    
    for pred in predictions:
        confidence = np.max(pred)
        predicted_class = np.argmax(pred)
        
        if predicted_class == 5:  # Assuming "not-skin" is at index 5
            not_skin_confidences.append(confidence)
        else:
            other_confidences.append(confidence)
    
    # Break after iterating over all batches
    if test_data.batch_index == test_data.batch_size - 1:
        break

# Plot histograms
plt.hist(not_skin_confidences, bins=20, alpha=0.7, label='Not Skin')
plt.hist(other_confidences, bins=20, alpha=0.7, label='Skin')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution')
plt.legend()
plt.show()
