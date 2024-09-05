import torch
import clip
from PIL import Image
import numpy as np

np.set_printoptions(suppress=True, precision=6)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("adog.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    probstext = logits_per_text.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 
print("Label probs_text:", probstext) 