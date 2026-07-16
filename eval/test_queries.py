import sys
import os
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path so we can import the retriever
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retriever.search import search

def run_evaluation_with_images():
    queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]
    
    # Create a 5x4 grid (5 queries, 1 text column + 3 image columns)
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    
    for i, q in enumerate(queries):
        # Run search
        results = search(q, k=3)
        
        # Column 0: Display the Query Text
        axes[i, 0].text(0.5, 0.5, f"Query {i+1}:\n\n'{q}'", 
                        fontsize=12, ha='center', va='center', wrap=True)
        axes[i, 0].axis('off')
        
        # Columns 1, 2, 3: Display Top 3 Images
        for j, r in enumerate(results):
            col_idx = j + 1
            try:
                img = Image.open(r['path'])
                axes[i, col_idx].imshow(img)
                axes[i, col_idx].set_title(f"Score: {r['fused_score']:.3f}", fontsize=10)
                axes[i, col_idx].axis('off')
            except Exception as e:
                axes[i, col_idx].text(0.5, 0.5, "Image not found", ha='center', va='center')
                axes[i, col_idx].axis('off')
                
    plt.tight_layout()
    output_path = "evaluation_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visual evaluation grid saved to {output_path}!")

if __name__ == "__main__":
    run_evaluation_with_images()