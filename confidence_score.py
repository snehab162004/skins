import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
model_path = 'skin_disease_model.h5'
model = load_model(model_path)

# Define data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    './processed_dataset/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to maintain consistency for analysis
)

# Initialize lists to store confidence scores
not_skin_confidences = []
skin_confidences = []

# Map class indices to labels
class_labels = {v: k for k, v in test_data.class_indices.items()}

# Iterate through the test dataset and collect confidence scores
for img_batch, label_batch in test_data:
    predictions = model.predict(img_batch)

    for pred in predictions:
        confidence = np.max(pred)  # Get the highest confidence score
        predicted_class = np.argmax(pred)

        # Classify as "Not Skin" or "Skin"
        if class_labels[predicted_class] == "not-skin":
            not_skin_confidences.append(confidence)
        else:
            skin_confidences.append(confidence)

    # Stop after processing all samples
    if len(not_skin_confidences) + len(skin_confidences) >= test_data.samples:
        break

# Plot confidence score distributions
plt.figure(figsize=(10, 6))
plt.hist(not_skin_confidences, bins=20, alpha=0.7, label='Not Skin', color='blue')
plt.hist(skin_confidences, bins=20, alpha=0.7, label='Skin', color='orange')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution for Skin and Not Skin')
plt.legend()
plt.show()

# Print summary statistics
print(f"Not Skin - Mean Confidence: {np.mean(not_skin_confidences):.2f}, Std: {np.std(not_skin_confidences):.2f}")
print(f"Skin - Mean Confidence: {np.mean(skin_confidences):.2f}, Std: {np.std(skin_confidences):.2f}")
