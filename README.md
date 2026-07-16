# Multimodal Fashion & Context Retrieval

A fashion image search system that takes plain-English queries instead of tags or filters — things like *"a bright yellow raincoat"* or *"casual weekend outfit for a city walk"* — and returns the images that actually match.

Built this for the Glance ML internship assignment. The core idea: combine FashionCLIP with zero-shot attribute enrichment and hybrid vector retrieval so the system handles both explicit attribute queries and vaguer style/context queries well, instead of being good at one and bad at the other.

## Demo

<!-- screenshot or gif of a query + retrieved results -->
![demo](docs/evaluation_results.png)

## Why FashionCLIP and not plain CLIP

Vanilla CLIP is a good general-purpose model but it doesn't pick up on fashion-specific distinctions all that well — oversized vs. slim fit, hoodie vs. sweatshirt, that kind of thing. FashionCLIP is trained on fashion image-text pairs specifically, so it's noticeably better at this without me having to fine-tune anything myself.

## What it does

- Fashion-aware retrieval using FashionCLIP embeddings
- Zero-shot attribute enrichment — no manual labeling of images
- Hybrid retrieval that blends visual similarity and text/caption similarity
- Adaptive weighting so attribute-heavy queries lean on text matching and style queries lean on visual matching

## How it works

Each image gets two embeddings: one straight from FashionCLIP (captures overall look and style) and one from an auto-generated caption (captures explicit stuff like color and garment type). The caption comes from comparing the image against a bank of candidate prompts and keeping whatever scores highest.

At query time I score the query against both embedding sets and blend the two scores. The blend isn't fixed — if the query sounds like it's naming specific attributes ("red tie with white shirt") I weight the caption side more; if it's a looser style query ("street fashion look") the image side gets more weight.

```
Query
  |
  v
FashionCLIP Text Encoder
  |
  +---------------------+
  |                      |
  v                      v
Caption Similarity   Image Similarity
  |                      |
  +---------------------+
  |
  v
Weighted Fusion
  |
  v
Ranked Results
```

Full write-up of the pipeline is in [`docs/architecture.md`](docs/architecture.md) if you want the details — keeping this file to the overview.

## Dataset

DeepFashion for the base clothing images (~1,000 of them). It's mostly studio shots though, so I added a small set of my own supplementary images with actual environmental context — offices, streets, parks — so the model has something to work with when someone searches for context rather than just an attribute.

## Stack

Python, PyTorch, FashionCLIP, ChromaDB, NumPy, Pillow. Kept the dependencies minimal on purpose.

## Project structure

```
.
├── data/
│   ├── raw_images_deepfashion/
│   ├── raw_images_supplements/
│   └── metadata/
├── indexer/
│   ├── captioner.py
│   └── build_index.py
├── retriever/
│   └── search.py
├── eval/
│   └── test_queries.py
├── docs/
│   └── architecture.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Running it

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

```bash
# generate enriched captions for every image
python -m indexer.captioner

# build the ChromaDB index
python -m indexer.build_index

# run a search
python -m retriever.search

# run the evaluation queries
python -m eval.test_queries
```

## Some queries I tested it on

- "bright yellow raincoat" — attribute
- "professional office attire" — context
- "red tie with white shirt" — composition (this one's the hardest, more below)
- "casual weekend outfit" — style

## Where it falls short

Attribute binding is the main weak spot — because attributes get extracted per-prompt independently, a query like "red tie and white shirt" can occasionally get muddled into something closer to "red shirt." Environmental recognition is also limited to whatever's in the prompt bank, so it won't pick up on a setting I didn't think to include.

Next things I'd add if I kept working on this: cross-encoder reranking, query decomposition for compositional queries, and a proper distributed vector DB (Pinecone/Milvus/Weaviate) if this ever needed to scale past a few thousand images.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — full pipeline walkthrough, including why I made the choices I made and what's still broken

## Thanks to

DeepFashion, OpenAI CLIP, FashionCLIP, ChromaDB
