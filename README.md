# Sentinel — Conflict Misinformation Early Warning System

**Know the narrative before it spreads.**


[![Backend API](https://img.shields.io/badge/Backend%20API-HuggingFace%20Spaces-yellow)](https://AliAounMehdi-sentinel.hf.space)
[![Built for](https://img.shields.io/badge/Built%20for-GenAI%20Z%C3%BCrich%20Hackathon%202026-green)](https://genaizurich.devpost.com)

---

## What is Sentinel

Sentinel is a real-time AI system that monitors 12 global news sources simultaneously, detects forming misinformation narratives around conflict topics, and assesses which false stories are gaining the most traction — giving journalists and analysts a critical head start before damage is done.

The system does not claim to identify fake news. It identifies contradictions between sources, suspicious narrative patterns, and spread risk — helping humans know exactly where to focus verification efforts.

---


## Features

- Breaking news ticker with real headlines updating every 2 hours
- 196 articles monitored across 12 global sources
- Semantic search powered by Qdrant finds related articles by meaning not just keywords
- Misinformation risk score from 0 to 100 with severity levels LOW, MEDIUM, HIGH, CRITICAL
- Detected narratives shown as categorized intelligence items
- Contradiction detection across sources reporting opposite facts on the same event
- Narrative traction assessment identifying which story is spreading fastest
- Story development timeline showing how narratives evolved over time
- Network propagation graph showing semantic connections between articles
- Exportable intelligence report for newsroom use
- Trust scoring based on source reputation with color coded indicators

---

## Architecture

### Data Collection — Apify

Apify's RSS Feed Aggregator Actor is triggered programmatically via the Apify Python SDK. The system calls the Actor with 12 configured RSS feed URLs and collects up to 180 structured articles per run. Each article returns clean JSON with title, summary, source, link, and publication time.

Sources monitored:
- BBC World and BBC Middle East
- Al Jazeera
- Middle East Eye
- The Guardian
- Washington Post
- CNN World
- Deutsche Welle
- NPR
- Sky News
- NDTV
- Time Magazine

### Semantic Storage — Qdrant

Each article is converted into a 384-dimensional vector using the sentence-transformers all-MiniLM-L6-v2 model from Hugging Face. Vectors are stored in Qdrant Cloud running in Frankfurt for low-latency European access.

When a user searches a topic, Qdrant performs cosine similarity search to find the 15 most semantically related articles in milliseconds. This is meaning-based matching not keyword matching — it finds articles about the same event even when they use completely different vocabulary.

### AI Analysis — Featherless

The 15 matched articles are sent to Mistral-7B via Featherless API. A structured prompt extracts factual claims, identifies direct contradictions between sources, scores misinformation risk, assesses narrative traction trajectory, and generates journalist recommendations — all returned as validated JSON.

### Backend — FastAPI on HuggingFace Spaces

Five REST API endpoints serve the frontend with live data, analysis results, network graphs, refresh triggers, and statistics.

### Frontend — React via Lovable

Three-panel dashboard: live intelligence feed, D3.js force-directed network graph, AI threat assessment panel with animated risk gauge.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| GET / | API status and article count |
| GET /api/live-feed | Latest articles with trust scores |
| GET /api/analyze/{topic} | Complete AI threat assessment for any topic |
| GET /api/network/{topic} | Node and edge data for network visualization |
| GET /api/refresh | Triggers fresh Apify SDK run and updates Qdrant |
| GET /api/stats | Database statistics and source distribution |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Data collection | Apify RSS Feed Aggregator via Apify Python SDK |
| Vector database | Qdrant Cloud Frankfurt |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| AI analysis | Mistral-7B via Featherless API |
| Backend framework | FastAPI |
| Backend hosting | HuggingFace Spaces |
| Frontend | React via Lovable |
| Visualization | D3.js force simulation |

---

## How to Run Locally

Clone the repository:

```
git clone https://github.com/Ali-Aoun5/sentinel.git
cd sentinel
```

Create virtual environment:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a .env file in the project root:

```
APIFY_TOKEN=your_apify_token
APIFY_DATASET_ID=your_dataset_id
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
FEATHERLESS_KEY=your_featherless_key
```

Run the backend:

```
python backend.py
```

API available at http://localhost:8000
API documentation at http://localhost:8000/docs

---

## Project Structure

```
sentinel/
├── backend.py          # FastAPI backend with Apify SDK integration
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container for HuggingFace Spaces deployment
├── .gitignore          # Excludes .env and venv
└── README.md           # This file
```

---

## Ethical Considerations

Sentinel is a risk assessment tool not a fact-checking service. It never definitively labels any article as misinformation. Trust scores are based on source reputation and should be treated as one signal among many. All data sources are publicly available RSS feeds. No personal data is collected or processed.

---

## Apify Integration Details

This project uses the Apify Python SDK to programmatically trigger Actor runs:

```python
from apify_client import ApifyClient

client = ApifyClient(APIFY_TOKEN)
run = client.actor("eloquent_mountain/rss-feed-aggregator").call(
    run_input={
        "urls": RSS_FEEDS,
        "maxItemsPerFeed": 15
    }
)
new_dataset_id = run["defaultDatasetId"]
```

Every call to /api/refresh triggers a live Apify Actor run, collects fresh articles, generates new embeddings, and updates the Qdrant vector database — ensuring the intelligence system always has current data.

---

*Sentinel v1.0 — Built for GenAI Zürich Hackathon 2026*
*Developed by Ali Aoun Mehdi*
