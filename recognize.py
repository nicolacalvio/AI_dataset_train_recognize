import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import json

file_path = "drive/MyDrive/class_mappings.json"

with open(file_path, "r") as json_file:
    data = json.load(json_file)

# Estrazione dei dati nelle variabili
label2id = data["label2id"]
id2label = data["id2label"]

# Initialize the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Initialize the model architecture
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# Load the trained weights
model.load_state_dict(torch.load("model_weights.pth"))

# Ensure the model is in evaluation mode
model.eval()

# Function to predict the class of an image
def predict_image_class(image_path, model, feature_extractor):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract predicted class ID
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()

    # Map predicted ID to class name
    predicted_class_name = id2label[str(predicted_class_id)]
    return predicted_class_name

# Example usage
image_path = 'aereo.webp'  # Update this to the path of your image
predicted_class = predict_image_class(image_path, model, feature_extractor)
print(f"Predicted class: {predicted_class}")
