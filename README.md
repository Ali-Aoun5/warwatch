# Sentinel — Conflict Misinformation Early Warning System

**Know the narrative before it spreads.**

Live Demo: https://sentinel-checker.lovable.app  
Backend API: https://AliAounMehdi-sentinel.hf.space  
Built for GenAI Zürich Hackathon 2026

---

## What is Sentinel

Sentinel is a real-time AI system that monitors global news sources, detects forming misinformation narratives around conflict topics, and predicts which false stories will spread before they reach mainstream media.

The system does not claim to identify fake news. It identifies contradictions between sources, suspicious narrative patterns, and viral spread risk — giving journalists and analysts a critical head start for verification before damage is done.

---

## Live Features

- Breaking news ticker with real headlines from 12 global sources updated every 2 hours
- 196 articles monitored simultaneously across BBC, Al Jazeera, Guardian, Washington Post, CNN, DW, NPR, Sky News, Middle East Eye, NDTV, Time and more
- Semantic search powered by Qdrant vector database finds related articles by meaning not just keywords
- AI threat assessment with misinformation risk score 0 to 100
- Detected narratives shown as categorized intelligence items
- Contradiction detection across sources reporting opposite facts on the same event
- Viral prediction identifying which narrative will spread in the next 6 hours
- Story development timeline showing how narratives evolved over time
- Network propagation graph showing semantic connections between articles
- Exportable intelligence report for newsroom use
- Trust scoring based on source reputation with color coded indicators

---

## Architecture

**Data Collection**  
Apify RSS Feed Aggregator scrapes 12 major global news sources on a scheduled basis. Each run collects structured article data including title, summary, publication time, and source URL as clean JSON.

**Semantic Storage**  
Articles are converted to 384-dimensional vectors using the sentence-transformers all-MiniLM-L6-v2 model from Hugging Face. Vectors are stored in Qdrant Cloud running in Frankfurt. When a user searches any topic, Qdrant performs cosine similarity search to find the 15 most semantically related articles — finding meaning-based connections that keyword search misses entirely.

**AI Analysis**  
The 15 matched articles are sent to Mistral-7B via Featherless API. The model extracts factual claims from each article, identifies direct contradictions between sources, scores narrative risk, generates viral spread predictions, and produces structured journalist recommendations — all returned as validated JSON.

**Backend**  
FastAPI service deployed on HuggingFace Spaces with five endpoints for live feed, topic analysis, network data, data refresh, and statistics.

**Frontend**  
React dashboard built with Lovable featuring three panels — live intelligence feed, D3.js force-directed semantic network graph, and AI threat assessment panel with animated risk gauge.

---

## API Endpoints

GET /api/live-feed — Returns latest articles with trust scores and source metadata  
GET /api/analyze/{topic} — Returns complete AI threat assessment for any topic  
GET /api/network/{topic} — Returns node and edge data for network visualization  
GET /api/refresh — Triggers fresh data collection from Apify  
GET /api/stats — Returns database statistics and source distribution  

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Data collection | Apify RSS Feed Aggregator |
| Vector database | Qdrant Cloud |
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

Create a virtual environment and install dependencies:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a .env file in the project root with your credentials:
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

The API will be available at http://localhost:8000  
API documentation at http://localhost:8000/docs

---

## System Requirements

The system requires active Apify, Qdrant, and Featherless API credentials. Free tiers are sufficient for development and demonstration purposes. The sentence-transformers model downloads automatically on first run approximately 90MB.

---

## Ethical Considerations

Sentinel is a risk assessment tool not a fact-checking service. It identifies where contradictions and suspicious patterns exist so human journalists can focus their verification efforts. The system never definitively labels any article as misinformation. Trust scores are based on source reputation and should be treated as one signal among many. All data sources are publicly available RSS feeds. No personal data is collected or processed.

---

## Project Structure
```
sentinel/
├── backend.py          # FastAPI backend with all endpoints
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration for HuggingFace Spaces
├── .gitignore          # Excludes .env and venv from version control
└── README.md           # This file
```

---

## Built With

Apify, Qdrant, Featherless, Mistral-7B, sentence-transformers, HuggingFace, FastAPI, Python, React, Lovable, D3.js

---

*Sentinel v1.0 — Built for GenAI Zürich Hackathon 2026*  
*Developed by Ali Aoun Mehdi*
