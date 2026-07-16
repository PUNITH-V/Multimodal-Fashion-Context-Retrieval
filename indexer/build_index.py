import os
import json
import torch
import open_clip
import numpy as np
from PIL import Image
import chromadb
from tqdm import tqdm

# 1. Load Fashion-CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Fashion-CLIP on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')

@torch.no_grad()
def embed_image(path):
    try:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        emb = model.encode_image(img)
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    except:
        return None

@torch.no_grad()
def embed_text(text):
    tokens = tokenizer([text]).to(device)
    emb = model.encode_text(tokens)
    return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

# 2. Setup ChromaDB
client = chromadb.PersistentClient(path="./vector_store")
try: client.delete_collection("fashion_images")
except: pass
collection = client.get_or_create_collection(name="fashion_images", metadata={"hnsw:space": "cosine"})

# 3. Load Manifest
with open("data/manifest.json", "r") as f:
    manifest = json.load(f)

print(f"Indexing {len(manifest)} images into ChromaDB...")

batch_size = 50
items = list(manifest.items())

for i in tqdm(range(0, len(items), batch_size)):
    batch = items[i:i+batch_size]
    ids, cap_embeddings, img_embeddings, metadatas, documents = [], [], [], [], []

    for img_id, data in batch:
        try:
            img_vec = embed_image(data["image_path"])
            cap_vec = embed_text(data["caption"])
            
            ids.append(img_id)
            cap_embeddings.append(cap_vec.tolist())
            documents.append(data["caption"])
            
            metadata = {
                "image_path": data["image_path"],
                "image_vector": img_vec.tolist(),
                # Store attributes as a comma-separated string for easy searching
                "attributes": ", ".join(data.get("attributes", [])),
                "environment": data.get("environment", "unknown"),
                "source": data.get("source", "unknown")
            }
            metadatas.append(metadata)
        except Exception as e:
            print(f"Skipping {img_id}: {e}")

    if ids:
        collection.add(ids=ids, embeddings=cap_embeddings, documents=documents, metadatas=metadatas)

print("✅ Indexing complete! Vector database saved to ./vector_store")