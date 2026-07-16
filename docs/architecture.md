# Architecture

This is the full walkthrough of how the retrieval pipeline actually works. The README has the short version — this is the long one.

## The basic idea

I'm using a two-tower setup: one tower handles what an outfit *looks* like, the other handles what it explicitly *contains* (color, garment type, fabric, setting). They're computed separately and combined when a query comes in.

## Step 1 — encoding the images

Every image goes through FashionCLIP and comes out as a single embedding. This embedding is mostly picking up on visual stuff — overall look, style, composition. It's not trying to name specific attributes, it's just a representation of "what this outfit looks like" that I can compare against other embeddings.

## Step 2 — zero-shot attribute enrichment

Labeling a few thousand images by hand wasn't realistic for this project, so instead I run each image against a list of candidate prompts and see which ones score highest by cosine similarity. Prompts look like:

```
a photo of a red dress
a photo in an office
a photo of casual clothing
```

Whichever prompts score highest get treated as the image's attributes, and I stitch those together (along with whatever metadata DeepFashion already provides) into a short description. Something like:

```
Person wearing a yellow raincoat.
Outdoor park environment.
Cotton fabric.
```

This is basically a cheap substitute for manual labeling — not perfect, but scales to however many images I throw at it without extra annotation work.

## Step 3 — storing two embeddings per image

For each image I end up storing:

- **Image embedding** — straight from FashionCLIP on the raw pixels. Good for style and visual similarity.
- **Caption embedding** — from the enriched description above. Good for exact attributes, colors, garment names.

I keep these separate on purpose rather than merging into one vector, because separating them is what lets me weight them differently depending on the query (more on that below). Both live in the same ChromaDB collection.

## Step 4 — handling a query

When someone types a query, I encode it with the same FashionCLIP text encoder used for the captions, then compute two similarity scores against everything in the index:

```
image_score   = similarity(query, image_embedding)
caption_score = similarity(query, caption_embedding)
```

These get combined into one ranking score:

```
score = alpha * image_score + beta * caption_score
```

Sort by that, return the top-k.

## Step 5 — adjusting the weights per query

alpha and beta aren't fixed. If the query looks like it's naming specific attributes — "blue denim jacket" — I push more weight onto beta (the caption side), since those words map pretty directly onto what's in the caption. If it reads more like a vibe or style description — "street fashion look" — alpha (image side) gets more weight, since there isn't necessarily a keyword in the query that maps cleanly onto anything in the caption text.

Right now this is a fairly simple heuristic based on whether the query contains explicit attribute-type words vs. more abstract style language. It works reasonably well but it's not bulletproof — see the design decisions doc for where this breaks.

## Why ChromaDB

Needed something that could do ANN search locally without setting up infrastructure. ChromaDB gives me persistent storage and HNSW indexing out of the box with a pretty minimal Python interface, which was the right tradeoff for a project at this scale.

## The pipeline end to end

```
Image Dataset
     |
FashionCLIP Encoding
     |
Zero-Shot Attribute Extraction
     |
Enriched Descriptions
     |
Hybrid Vector Index (ChromaDB)
     |
Query
     |
Adaptive Weighted Fusion
     |
Ranked Results
```

See [`design_decisions.md`](design_decisions.md) for the reasoning behind these choices and what I'd still want to fix.
