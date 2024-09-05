import torch
import clip
from PIL import Image
import numpy as np

# Set print options for numpy to avoid scientific notation
np.set_printoptions(suppress=True, precision=6)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess multiple images
image_paths = ["CLIP.png", "adog.png", "acat.png"]  # Add your image paths here
images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]

# Concatenate images to create a batch
image_batch = torch.cat(images, dim=0)

# Tokenize the text (assume you want to compare these images to a single text like "a dog")
text = clip.tokenize(["a dog"]).to(device)

with torch.no_grad():
    # Encode images and text
    image_features = model.encode_image(image_batch)
    text_features = model.encode_text(text)
    
    # Compute similarity between images and text
    logits_per_image, logits_per_text = model(image_batch, text)
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

# Print the probabilities for each image
print("Label probs for each image:", probs)
