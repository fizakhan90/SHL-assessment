# SHL Assessment Recommender

A conversational AI agent that helps hiring managers find the right SHL assessments through dialogue. Built as a take-home assignment for the SHL Labs AI Intern role.

## What it does

- Asks clarifying questions when the query is too vague
- Recommends 1-10 relevant SHL assessments once it has enough context
- Refines recommendations when the user changes constraints mid-conversation
- Compares assessments using catalog data only, never from model memory
- Refuses off-topic questions and prompt injection attempts

## Live API

**Base URL:** `https://shl-assessment-wp26.onrender.com`

> Note: Deployed on Render free tier. First request after inactivity may take up to 120 seconds to wake up.

### Endpoints

**Health check**
```
GET /health
```
```json
{"status": "ok"}
```

**Chat**
```
POST /chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "I need a Java developer assessment"},
    {"role": "assistant", "content": "Here are some options..."},
    {"role": "user", "content": "Can you add a personality test too?"}
  ]
}
```
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

The API is fully stateless — full conversation history is sent with every request.

## Project Structure

```
SHL_assessment/
├── main.py                  # FastAPI app with /health and /chat endpoints
├── retrieval.py             # Hybrid BM25 + semantic retrieval with RRF
├── scraper.py               # Catalog scraper (requests + BeautifulSoup)
├── eval.py                  # Evaluation script (Recall@10, Precision@10, F1@10)
├── shl_catalog_clean.json   # Cleaned catalog (335 entries)
├── shl_individual_tests.json # Raw scrape (389 entries)
├── embeddings_cache/        # Pre-computed HuggingFace embeddings
├── chroma_db/               # ChromaDB vector index
├── requirements.txt
└── render.yaml
```

## Stack

| Component | Choice | Reason |
|---|---|---|
| Framework | FastAPI | Fast, async, automatic schema validation |
| LLM | Google Gemini 1.5 Flash | 1M tokens/day free tier, reliable |
| Embeddings | HuggingFace Inference API (all-MiniLM-L6-v2) | Avoids OOM on Render free tier |
| Sparse retrieval | rank-bm25 | Lightweight, handles exact name queries |
| Deployment | Render | Free tier, simple GitHub integration |

## Retrieval Pipeline

Three iterations measured using Mean Recall@10 on 10 public evaluation traces:

| Stage | Change | Recall@10 |
|---|---|---|
| Baseline | BM25 only | 0.31 |
| + Catalog fix | Recovered 7 missing entries | 0.40 |
| + Hybrid | BM25 + semantic search via RRF | 0.46 |
| + Pipeline | Query distillation + k=30 retrieve + LLM rerank | **0.63** |

## Running Locally

1. Clone the repo
```bash
git clone https://github.com/fizakhan90/SHL-assessment.git
cd SHL-assessment
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file
```
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. Test it
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need a Java developer assessment"}]}'
```

## Evaluation

Run the evaluation script against the public traces:

```bash
python eval.py
```

Outputs Recall@10, Precision@10, and F1@10 for BM25-only vs hybrid vs full pipeline, plus a gap analysis showing which expected assessments each mode missed.

## Catalog

Scraped from `https://www.shl.com/solutions/products/product-catalog/` (Individual Test Solutions only). 389 entries scraped, 335 kept after filtering companion reports with null test_type. 7 entries missing from the clean catalog were recovered manually after evaluation revealed their absence as a recall bottleneck.
