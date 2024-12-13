import kagglehub

# Download latest version
path = kagglehub.dataset_download("shakyadissanayake/oily-dry-and-normal-skin-types-dataset")

print("Path to dataset files:", path)