# Text2Breed — Streamlit Starter

A minimal two‑page feel (landing → finder) as shown in your mockups.
- Loads an embedding model once (`all-mpnet-base-v2`).
- Classifies with cosine similarity against a small corpus you can replace.
- Shows Top‑1 + Top‑3 with images.

## Quickstart

```bash
# inside a clean venv
pip install -r requirements.txt

# run
streamlit run app.py
```

## Replace the demo data

- Put your reference texts in `data/breed_corpus.csv` with columns:

```
breed,description
Poodle,"..." 
Bichon Frise,"..."
...
```

- Optionally provide `data/breed_images.json` mapping breed → image URL or local path:

```json
{
  "Poodle": "assets/poodle.jpg",
  "Bichon Frise": "https://..."
}
```

> For best results compute embeddings on concise, neutral descriptions.
> The app normalizes embeddings; scores are simple cosine similarities.

## Using your own classifier
If you have a fine‑tuned classifier head instead of cosine similarity,
replace `prepare_index()` and `top_k()` in `app.py` with your inference code.
Keep the return format as `[(breed, score), ...]` so the UI stays the same.