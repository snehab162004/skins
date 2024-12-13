import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits a nested dataset into train, validation, and test sets.

    Args:
        dataset_path (str): Path to the source dataset directory containing class folders.
        output_path (str): Path to save the split dataset.
        train_ratio (float): Proportion of the dataset for training.
        val_ratio (float): Proportion of the dataset for validation.
        test_ratio (float): Proportion of the dataset for testing.
    """
    # Get all leaf folders (categories) recursively
    categories = []
    for root, dirs, files in os.walk(dataset_path):
        if not dirs:  # If no subdirectories, this is a leaf folder
            categories.append(root)

    # Create directories for train, val, and test splits
    for split in ["train", "val", "test"]:
        for category_path in categories:
            category_name = category_path.replace(dataset_path, "").strip(os.sep)  # Get relative path as category name
            os.makedirs(os.path.join(output_path, split, category_name), exist_ok=True)

    # Process each category
    for category_path in categories:
        category_name = category_path.replace(dataset_path, "").strip(os.sep)
        images = [img for img in os.listdir(category_path) if img.endswith((".jpg", ".png", ".jpeg"))]

        # Skip empty directories
        if len(images) == 0:
            print(f"Skipping empty category: {category_name}")
            continue

        # Handle small categories (fewer than 3 images)
        if len(images) < 3:
            print(f"Category {category_name} has fewer than 3 images. Moving all to the training set.")
            for img in images:
                shutil.copy(os.path.join(category_path, img), os.path.join(output_path, "train", category_name))
            continue

        # Split the images into train, validation, and test
        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)

        # Copy images to the appropriate directories
        for img in train_imgs:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_path, "train", category_name))
        for img in val_imgs:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_path, "val", category_name))
        for img in test_imgs:
            shutil.copy(os.path.join(category_path, img), os.path.join(output_path, "test", category_name))

        print(f"Processed category {category_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    print("Dataset splitting complete!")

# Paths to your dataset and output directories
dataset_path = "./datasets"  # Path to the original dataset
output_path = "./processed_dataset"  # Path to save the split dataset

# Run the dataset splitting function
split_dataset(dataset_path, output_path)

def count_images(directory):
    counts = {}
    for root, dirs, files in os.walk(directory):
        if not dirs:  # If no subdirectories, count files
            category = root.replace(directory, "").strip(os.sep)
            counts[category] = len(files)
    return counts

print("Train set counts:", count_images(os.path.join(output_path, "train")))
print("Validation set counts:", count_images(os.path.join(output_path, "val")))
print("Test set counts:", count_images(os.path.join(output_path, "test")))
