import os
from core.entity_rec import translate
from PIL import Image
import torch
import open_clip
import os

save_dir = "app/static/images"
image_dir = "images/pilot/"
clip_dir = 'clips/pilot/'


def generate_images_from_prompts(prompts, progress_callback=None):
    images = []
    docx = []
    total_prompts = len(prompts)

    OPENCLIP_MODEL = "ViT-L-14"
    OPENCLIP_DATA = "laion2b_s32b_b82k"
    print("Initializing model...")
    model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL, OPENCLIP_DATA)
    model.eval()
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)

    image_paths = [os.path.join(clip_dir, f) for f in sorted(os.listdir(clip_dir)) if f.endswith('.pt')]

    image_features_list = []
    for path in image_paths:
        # Load and process each image feature
        feat = torch.load(path).unsqueeze(0)  # Add batch dimension
        pooled_features = feat.mean(dim=(1, 2))  # Average pool spatial dimensions
        pooled_features = pooled_features.to(model.visual.proj.dtype)
        image_feat = pooled_features @ model.visual.proj  # Project to joint space
        image_features_list.append(image_feat)

    image_features = torch.cat(image_features_list, dim=0)

    for i, base_prompt in enumerate(prompts):
        engineered_prompt = translate(base_prompt)
        print("CONCEPT: ", engineered_prompt)

        text = tokenizer([engineered_prompt])

        with torch.no_grad(), torch.autocast("cuda"):
            # Encode text
            text_features = model.encode_text(text)
            
            # Normalize features
            image_features_temp = image_features.to(text_features.dtype)
            image_features_temp /= image_features_temp.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores and softmax probabilities
            similarity_scores = 100.0 * (image_features_temp @ text_features.T)
            image_probs = similarity_scores.softmax(dim=0)

        sorted_pairs = sorted(
            zip(image_paths, image_probs),
            key=lambda x: x[1].item(),  # sort by the tensor’s scalar value
            reverse=True
        )
        
        base, _ = os.path.splitext(os.path.basename(sorted_pairs[0][0]))
        filename = base + '.png'

        img = Image.open(os.path.join(image_dir, filename))

        # Resize to 512x512
        img = img.resize((512, 512), Image.LANCZOS)

        # Define a unique name for each generated image
        generated_image_name = f"section_{i}.png"
        generated_image_filepath = os.path.join(save_dir, generated_image_name)

        # Save the resized image
        img.save(generated_image_filepath)
        
        images.append((engineered_prompt, generated_image_filepath))
        docx.append({'image_path':generated_image_filepath, 'text': base_prompt})
        
        # Update the progress
        if progress_callback:
            progress_callback(i +1, total_prompts)

    return images, docx
