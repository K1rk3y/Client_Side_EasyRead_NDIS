import torch
import open_clip
import os

OPENCLIP_MODEL = "ViT-L-14"
OPENCLIP_DATA = "laion2b_s32b_b82k"
print("Initializing model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    OPENCLIP_MODEL, OPENCLIP_DATA
)
model.eval()
tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

# Load and process all image features from directory
image_dir = "clips/pilot/"
image_paths = [
    os.path.join(image_dir, f)
    for f in sorted(os.listdir(image_dir))
    if f.endswith(".pt")
]

image_features_list = []
for path in image_paths:
    # Load and process each image feature
    feat = torch.load(path).unsqueeze(0)  # Add batch dimension
    pooled_features = feat.mean(dim=(1, 2))  # Average pool spatial dimensions
    pooled_features = pooled_features.to(model.visual.proj.dtype)
    image_feat = pooled_features @ model.visual.proj  # Project to joint space
    image_features_list.append(image_feat)

# Stack all image features into a single tensor
image_features = torch.cat(image_features_list, dim=0)

# Encode text query
text_query = "a man"
text = tokenizer([text_query])

with torch.no_grad(), torch.autocast("cuda"):
    # Encode text
    text_features = model.encode_text(text)

    # Normalize features
    image_features = image_features.to(text_features.dtype)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores and softmax probabilities
    similarity_scores = 100.0 * (image_features @ text_features.T)
    image_probs = similarity_scores.softmax(dim=0)

print("Image probabilities:")
sorted_pairs = sorted(
    zip(image_paths, image_probs),
    key=lambda x: x[1].item(),  # sort by the tensor’s scalar value
    reverse=True,
)

for path, prob in sorted_pairs:
    print(f"{os.path.basename(path)}: {prob.item():.4f}")
