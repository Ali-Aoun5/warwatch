# Sentinel — Conflict Misinformation Early Warning System

> **Know the narrative before it spreads.**

[![Live Demo](https://img.shields.io/badge/🔴%20Live%20Demo-sentinel--checker.lovable.app-brightgreen?style=for-the-badge)](https://sentinel-checker.lovable.app)
[![Backend API](https://img.shields.io/badge/Backend%20API-HuggingFace%20Spaces-yellow?style=for-the-badge)](https://AliAounMehdi-sentinel.hf.space)
[![Built for](https://img.shields.io/badge/Built%20for-GenAI%20Zürich%20Hackathon%202026-blue?style=for-the-badge)](https://genaizurich.devpost.com)
[![Apify](https://img.shields.io/badge/Powered%20by-Apify-orange?style=for-the-badge)](https://apify.com)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red?style=for-the-badge)](https://qdrant.tech)

---

## The Problem

At 2am during the Iran-US conflict, BBC was reporting that Iran denied any attack. At the exact same moment, NDTV was reporting that Iran claimed full responsibility. Both articles were being shared by thousands of people simultaneously. Neither mentioned the other existed.

Nobody was lying. But someone was wrong. And by morning, millions of people had formed opinions based on whichever version they happened to read first.

**The window between when a false narrative forms and when a fact-checker catches it is where the real damage happens. Sentinel was built to close that window.**

---

## What Sentinel Does

Sentinel monitors 20 global news sources simultaneously, detects when different outlets are reporting contradictory facts about the same conflict event, and delivers a complete intelligence report in under 30 seconds — before false narratives have time to spread.

Search any conflict topic. Get a structured threat assessment showing exactly where the truth is being disputed, which narrative is gaining traction, and what to verify before publishing.

**Try it live:** [sentinel-checker.lovable.app](https://sentinel-checker.lovable.app)

---

## Key Results

| Metric | Result |
|--------|--------|
| Query-to-report latency | Under 30 seconds |
| Articles monitored | 154+ across 20 global sources |
| Risk score on Iran conflict | **75/100 — HIGH** (matched ground truth) |
| Embedding dimensions | 384 (meaning-based, not keyword) |
| Infrastructure cost | $0 — runs entirely on free tier |
| Hackathon ranking | **Top 28 / 450+ global teams** |

When queried on the Iran conflict, Sentinel returned a risk score of 75/100 HIGH and correctly identified that BBC and NDTV were publishing directly contradictory factual claims. It named the specific competing narratives and assessed spread trajectory. That assessment matched what appeared in broader media coverage over the following hours.

---

## Live Dashboard

The three-panel dashboard gives journalists everything they need in one view:

**Left — Live Intelligence Feed**
Real-time breaking news ticker with 154+ articles from 20 sources. Every article is color-coded by source trust score: green for Tier 1 premium outlets, amber for uncertain sources, red for flagged sources with documented reliability concerns.

**Center — Narrative Propagation Map**
D3.js force-directed network graph showing how articles about a topic are semantically connected. Nodes represent articles, connections represent meaning-based similarity. Red nodes are flagged sources being picked up by trusted ones — that is the spread pattern that matters.

**Right — AI Threat Assessment**
Mistral-7B reads 15 articles simultaneously and produces: misinformation risk score 0-100, detected narratives with specific source attribution, direct contradictions between outlets, viral spread prediction, journalist verification recommendations, and story development timeline.

---

## Architecture

```
RSS Feeds (20 sources)
        │
        ▼
  Apify RSS Actor          ← Triggers every 30 minutes via SDK
        │
        ▼
  Article JSON             ← title, summary, source, link, timestamp
        │
        ▼
  HuggingFace Embeddings   ← all-MiniLM-L6-v2 → 384-dim vectors
        │
        ▼
  Qdrant Cloud Frankfurt   ← cosine similarity search
        │
        ▼
  Mistral-7B (Featherless) ← contradiction detection + risk scoring
        │
        ▼
  FastAPI (HuggingFace)    ← REST API serving frontend
        │
        ▼
  React Dashboard (Lovable) ← D3.js network graph + risk gauge
```

### Stage 1 — Data Collection via Apify

The Apify Python SDK triggers the RSS Feed Aggregator Actor programmatically. The Actor scrapes 20 global news sources every 30 minutes and returns clean structured JSON. Apify solved the hardest part of this pipeline — getting fresh, real, clean data from the messy real world at scale.

```python
from apify_client import ApifyClient

client = ApifyClient(APIFY_TOKEN)
run = client.actor("mehdialiaoun/sentinel-rss-actor").call(
    run_input={"urls": RSS_FEEDS, "maxItemsPerFeed": 10}
)
dataset_id = run["defaultDatasetId"]
```

Sources monitored include BBC World, BBC Middle East, Reuters, AP News, Al Jazeera, The Guardian, Washington Post, CNN, Deutsche Welle, NPR, Sky News, Middle East Eye, NDTV, Time Magazine, France24, Euronews, NBC News, Wall Street Journal, TRT World, and Arab News.

### Stage 2 — Semantic Embedding

Each article is converted into a 384-dimensional dense vector using `sentence-transformers/all-MiniLM-L6-v2`. This step is what makes cross-source contradiction detection possible. BBC saying "Iran denied the attack" and NDTV saying "Iran claimed responsibility" produce vectors that are mathematically close because they describe the same event — even though the words are completely different.

### Stage 3 — Vector Storage and Retrieval via Qdrant

Vectors are stored in Qdrant Cloud running in Frankfurt for low-latency European access. When a user searches a topic, Qdrant performs cosine similarity search to retrieve the 15 most semantically related articles in milliseconds. This is not keyword matching. It finds articles about the same event regardless of vocabulary differences — which is exactly what makes contradiction detection work at scale.

### Stage 4 — AI Analysis via Mistral-7B

The 15 retrieved articles are sent to Mistral-7B via Featherless API with a structured prompt that extracts factual claims from each source, identifies direct contradictions, assigns a 0-100 risk score, assesses narrative spread trajectory, and generates journalist verification recommendations — returned as validated JSON.

### Stage 5 — Backend

FastAPI deployed on HuggingFace Spaces orchestrates the entire pipeline, manages automatic refresh cycles, and serves five REST endpoints to the React frontend.

### Stage 6 — Frontend Dashboard

React dashboard built with Lovable displays intelligence across three panels. The network graph uses D3.js force simulation. The risk gauge is animated with real-time score updates.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status, article count, model info |
| `/api/live-feed` | GET | Latest articles with trust scores and metadata |
| `/api/analyze/{topic}` | GET | Complete AI threat assessment for any search topic |
| `/api/network/{topic}` | GET | Node and edge data for D3.js network visualization |
| `/api/refresh` | GET | Triggers fresh Apify Actor run and rebuilds Qdrant index |
| `/api/stats` | GET | Full database statistics and source distribution |
| `/api/sources` | GET | All monitored sources with trust tiers |
| `/api/debug/featherless` | GET | API health check for Featherless connection |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data collection | Apify RSS Feed Aggregator | Fresh news ingestion every 30 minutes |
| Vector database | Qdrant Cloud Frankfurt | Semantic article storage and retrieval |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | 384-dim meaning-based encoding |
| AI analysis | Mistral-7B via Featherless API | Contradiction detection and risk scoring |
| Backend | FastAPI | REST API and pipeline orchestration |
| Hosting | HuggingFace Spaces | Zero-cost production deployment |
| Frontend | React via Lovable | Three-panel intelligence dashboard |
| Visualization | D3.js force simulation | Narrative propagation network graph |

---

## Trust Scoring System

Every source is assigned a trust score based on documented editorial standards and historical accuracy. This score directly affects the risk assessment — a flagged source contradicting a trusted source is weighted differently than two trusted sources disagreeing.

| Tier | Score Range | Examples |
|------|------------|---------|
| Tier 1 — Premium | 90-100 | Reuters (93), AP News (93), BBC (90) |
| Tier 2 — Reliable | 80-89 | NYT (88), Guardian (88), NPR (88), DW (87) |
| Tier 3 — Monitor | 65-79 | Al Jazeera (72), Sky News (75), Euronews (78) |
| Tier 4 — Caution | Below 65 | RT (25), Sputnik (20) |

---

## Engineering Challenges

**The RSS parsing problem.** Apify's standard Web Scraper failed on every RSS URL because RSS is XML and the browser-based scraper attempted to render it as HTML. After two hours of debugging I found the RSS Feed Aggregator Actor which solved the problem immediately. The lesson was to find the right tool rather than forcing the wrong one to work.

**The Docker build failure.** A Windows-only library in my requirements file caused the Docker build on Linux HuggingFace Spaces to crash with a confusing encoding error. The Qdrant Python client also silently changed a method name between versions, causing a search endpoint error that pointed nowhere near the actual problem. Both required systematic elimination to diagnose.

**The API token exposure.** An Apify token was accidentally pushed to GitHub. Apify's automated scanning system detected it within hours and sent a security alert. The token was rotated immediately, all credentials were migrated to HuggingFace Space environment variables, and the entire commit history was audited. This was a good lesson learned under pressure.

---

## Ethical Design

Sentinel explicitly does not label content as true or false. A system that confidently identifies misinformation creates a different kind of misinformation problem — false confidence in automated judgment.

Instead, Sentinel identifies where contradictions exist and where the risk is highest, helping humans know exactly where to focus verification efforts. The tool is designed to support editorial judgment, not replace it.

All data sources are publicly available RSS feeds. No personal data is collected or processed. Trust scores are one signal among many and are documented transparently.

---

## What is Next for Sentinel

The current version monitors news outlets only. Narratives often originate on social media hours before any outlet picks them up. The roadmap includes:

- Real-time social media ingestion via Apify Twitter and Reddit scrapers
- Automated journalist alerts when tracked topics cross CRITICAL threshold
- Election and public health monitoring modules
- Browser extension showing live Sentinel risk scores on any article
- Integration with emergency services for crisis communication verification

---

## Running Locally

```bash
git clone https://github.com/AliAounMehdi/sentinel.git
cd sentinel
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Mac/Linux
pip install -r requirements.txt
```

Create `.env` file:

```
APIFY_TOKEN=your_apify_token
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
FEATHERLESS_KEY=your_featherless_key
SENTINEL_ACTOR_ID=mehdialiaoun/sentinel-rss-actor
```

Run:

```bash
python backend.py
```

API: `http://localhost:8000`
Docs: `http://localhost:8000/docs`

---

## Project Structure

```
sentinel/
├── backend.py          # FastAPI backend — pipeline orchestration
├── requirements.txt    # Python dependencies
├── Dockerfile          # HuggingFace Spaces container config
├── .gitignore          # Excludes .env, venv, __pycache__
└── README.md           # This file
```

---

## About

Built solo in under 3 weeks for the GenAI Zürich Hackathon 2026.
Selected **Top 28 from 450+ global submissions** — Apify Challenge Category finalist.
Representing Pakistan 🇵🇰 and Institute of Space Technology, Islamabad.

**Developed by Ali Aoun Mehdi**
[LinkedIn](https://linkedin.com/in/ali-aoun-mehdi) · [GitHub](https://github.com/AliAounMehdi) · [Live Demo](https://sentinel-checker.lovable.app)

---

*The combination of Apify and Qdrant does something neither can do alone. Apify solves the hardest part of any AI application — getting clean, fresh, real data from the messy real world. Qdrant turns that data into something you can reason about semantically rather than just search through literally. Together they create an intelligence layer that operates at the speed of the news cycle itself.*

---

*Sentinel v1.0 — GenAI Zürich Hackathon 2026*
