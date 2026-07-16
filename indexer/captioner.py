import os
import json
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# 1. Load Fashion-CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Fashion-CLIP on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')

@torch.no_grad()
def get_image_embedding(image_path):
    try:
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    except Exception as e:
        return None

@torch.no_grad()
def get_text_embeddings(text_list):
    tokens = tokenizer(text_list).to(device)
    emb = model.encode_text(tokens)
    return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()

# 2. Define Attributes to detect
ATTRIBUTES = {
    "colors": ["red", "blue", "yellow", "white", "black", "green", "gray", "pink", "brown", "orange", "navy", "beige"],
    "clothing_type": ["t-shirt", "shirt", "pants", "dress", "suit", "blazer", "hoodie", "raincoat", "tie", "shorts", "sweater", "jacket", "jeans"],
    "environment": ["office", "park", "street", "home", "studio background", "outdoor nature", "indoor room"]
}

def detect_attributes(image_emb):
    detected = {}
    for attr_type, labels in ATTRIBUTES.items():
        if attr_type == "colors":
            prompts = [f"a photo of a {label} garment" for label in labels]
        elif attr_type == "clothing_type":
            prompts = [f"a photo of a {label}" for label in labels]
        else:
            prompts = [f"a photo in a {label}" for label in labels]
        
        text_embs = get_text_embeddings(prompts)
        sims = (text_embs @ image_emb)
        
        # Get top 2 matches
        top_indices = sims.argsort()[-2:][::-1]
        top_labels = [labels[i] for i in top_indices]
        detected[attr_type] = top_labels
    return detected

def process_all_images():
    print("Building rich manifest using Local Zero-Shot Classification...")
    
    manifest = {}
    output_file = "data/manifest.json"
    
    df_meta = {}
    if os.path.exists("data/deepfashion_meta.json"):
        with open("data/deepfashion_meta.json", "r") as f:
            df_meta = json.load(f)
            
    all_images = []
    df_folder = "data/raw_images_deepfashion"
    supp_folder = "data/raw_images_supplements"
    
    if os.path.exists(df_folder): all_images.extend([(f, os.path.join(df_folder, f), "deepfashion") for f in os.listdir(df_folder)])
    if os.path.exists(supp_folder): all_images.extend([(f, os.path.join(supp_folder, f), "supplement") for f in os.listdir(supp_folder)])
    
    for img_name, path, source in tqdm(all_images, desc="Classifying Images"):
        img_emb = get_image_embedding(path)
        if img_emb is None: continue
        
        attrs = detect_attributes(img_emb)
        existing_cap = df_meta.get(img_name, {}).get("caption", "")
        
        colors = ", ".join(attrs["colors"])
        cloth = ", ".join(attrs["clothing_type"])
        env = attrs["environment"][0]
        
        if existing_cap:
            rich_caption = f"A person wearing {colors} {cloth}. {existing_cap} Setting: {env}."
        else:
            rich_caption = f"A person wearing {colors} {cloth} in a {env} setting."
            
        manifest[img_name] = {
            "image_path": path,
            "source": source,
            "caption": rich_caption,
            "colors": attrs["colors"],
            "clothing_type": attrs["clothing_type"],
            "environment": env
        }
        
        if len(manifest) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(manifest, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Done! Saved {len(manifest)} items to manifest.json")

process_all_images()