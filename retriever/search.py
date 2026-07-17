import torch
import open_clip
import numpy as np
import chromadb

# 1. Load Fashion-CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Fashion-CLIP on {device}...")
model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionCLIP')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionCLIP')

# 2. Load ChromaDB
client = chromadb.PersistentClient(path="./vector_store")
collection = client.get_collection(name="fashion_images")

@torch.no_grad()
def embed_text(text):
    tokens = tokenizer([text]).to(device)
    emb = model.encode_text(tokens)
    return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

# 3. Define Attributes for Dynamic Weighting
COLORS = ["red", "blue", "yellow", "white", "black", "green", "gray", "pink", "brown", "orange", "navy", "beige"]
GARMENTS = ["t-shirt", "shirt", "pants", "dress", "suit", "blazer", "hoodie", "raincoat", "tie", "shorts", "sweater", "jacket", "jeans"]

def search(query, k=5):
    print(f"\n🔍 Query: '{query}'")
    q_vec = embed_text(query)
    
    # --- Dynamic Weighting Logic ---
    query_lower = query.lower()
    has_attributes = any(c in query_lower for c in COLORS) or any(g in query_lower for g in GARMENTS)
    
    if has_attributes:
        # Specific query (like Q1, Q5): Trust the caption text
        alpha, beta = 0.3, 0.7
    else:
        # Abstract query (like Q4 "casual weekend"): Trust the image vector
        alpha, beta = 0.7, 0.3
    # ------------------------------------

    # Step 1: Fetch top 50 candidates by Caption similarity
    results = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=50,
        include=["metadatas", "documents", "distances"]
    )
    
    candidates = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        
        # Step 2: Calculate Image similarity
        img_vec = np.array(meta['image_vector'])
        cap_sim = 1 - results['distances'][0][i] 
        img_sim = float(np.dot(q_vec, img_vec))  
        
        # Step 3: Fused Score (Now using dynamic alpha/beta)
        fused_score = (alpha * img_sim) + (beta * cap_sim)
        
        candidates.append({
            "id": results['ids'][0][i],
            "path": meta['image_path'],
            "caption": results['documents'][0][i],
            "fused_score": fused_score,
            "img_sim": img_sim,
            "cap_sim": cap_sim,
            "environment": meta.get("environment", "unknown"),
            # FIX: Read the correct key
            "colors": meta.get("colors", "unknown"),
            "clothing_type": meta.get("clothing_type", "unknown")
        })
    
    # Step 4: Sort by Fused Score and return Top K
    candidates = sorted(candidates, key=lambda x: -x['fused_score'])
    return candidates[:k]
