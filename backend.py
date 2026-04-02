import os
import hashlib
import asyncio
import logging
import re
import json
import html
from datetime import datetime
from typing import Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
)
from sentence_transformers import SentenceTransformer
from apify_client import ApifyClient
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
from dotenv import load_dotenv
load_dotenv()

APIFY_TOKEN = os.environ.get("APIFY_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
FEATHERLESS_KEY = os.environ.get("FEATHERLESS_KEY")
YOUR_ACTOR_ID = os.environ.get("SENTINEL_ACTOR_ID", "mehdialiaoun/sentinel-rss-actor")

COLLECTION_NAME = "sentinel_articles_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
REFRESH_INTERVAL_SECONDS = 1800  # 30 minutes

# ============ 20 GLOBAL NEWS SOURCES ============
RSS_FEEDS = [
    # Tier 1 — Highest Trust Wire Services and Premium
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://apnews.com/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.washingtonpost.com/rss/world",
    "https://feeds.npr.org/1001/rss.xml",
    # Tier 2 — International and Regional
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.france24.com/en/rss",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://feeds.skynews.com/feeds/rss/world.xml",
    "https://www.middleeasteye.net/rss",
    "https://feeds.feedburner.com/euronews/en/news",
    # Tier 3 — Regional Perspectives
    "https://feeds.feedburner.com/ndtvnews-world-news",
    "https://feeds.feedburner.com/time/world",
    "https://feeds.feedburner.com/trtworld",
    "https://www.arabnews.com/rss.xml",
    # Tier 4 — Alternative and Watchlist
    "https://theintercept.com/feed/?rss",
    "https://feeds.nbcnews.com/nbcnews/public/world",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
]

# ============ SOURCE TRUST SCORES ============
SOURCE_TRUST = {
    "bbci.co.uk": 90,
    "bbc.com": 90,
    "reuters.com": 93,
    "apnews.com": 93,
    "nytimes.com": 88,
    "theguardian.com": 88,
    "washingtonpost.com": 85,
    "npr.org": 88,
    "aljazeera.com": 72,
    "france24.com": 82,
    "dw.com": 87,
    "skynews.com": 75,
    "middleeasteye.net": 72,
    "euronews.com": 78,
    "ndtv.com": 68,
    "time.com": 82,
    "trtworld.com": 68,
    "arabnews.com": 65,
    "theintercept.com": 65,
    "nbcnews.com": 78,
    "wsj.com": 88,
    "feedburner.com": 55,
    "rt.com": 25,
    "sputniknews.com": 20,
}

# ============ INITIALIZE SERVICES ============
app = FastAPI(title="Sentinel API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded successfully")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ============ GLOBAL STATE ============
last_refresh_time = None
last_article_count = 0
high_risk_detections_today = 0
last_reset_date = datetime.now().date()
topics_analyzed_today = []

# ============ APIFY ============

def trigger_fresh_apify_run() -> str:
    logger.info(f"Triggering Sentinel Actor: {YOUR_ACTOR_ID}")
    client = ApifyClient(APIFY_TOKEN)
    run = client.actor(YOUR_ACTOR_ID).call(
        run_input={
            "urls": RSS_FEEDS,
            "maxItemsPerFeed": 10
        }
    )
    new_dataset_id = run["defaultDatasetId"]
    logger.info(f"Actor run completed. Dataset ID: {new_dataset_id}")
    return new_dataset_id

# ============ HELPER FUNCTIONS ============

def clean_html(text: str) -> str:
    if not text:
        return ""
    clean = html.unescape(text)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'&[a-zA-Z]+;', ' ', clean)
    clean = ' '.join(clean.split())
    return clean[:800]

def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.8

def get_trust_score(source_url: str) -> int:
    if not source_url:
        return 50
    source_lower = source_url.lower()
    for domain, score in SOURCE_TRUST.items():
        if domain in source_lower:
            return score
    return 55

def get_trust_label(score: int) -> str:
    if score >= 85:
        return "trusted"
    elif score >= 65:
        return "uncertain"
    else:
        return "flagged"

def get_trust_tier(score: int) -> str:
    if score >= 90:
        return "Tier 1 — Premium"
    elif score >= 80:
        return "Tier 2 — Reliable"
    elif score >= 65:
        return "Tier 3 — Monitor"
    else:
        return "Tier 4 — Caution"

def generate_article_id(url: str, index: int = 0) -> int:
    return int(hashlib.md5(f"{url}{index}".encode()).hexdigest()[:8], 16)

def reset_daily_counters():
    global high_risk_detections_today, last_reset_date, topics_analyzed_today
    today = datetime.now().date()
    if today != last_reset_date:
        high_risk_detections_today = 0
        topics_analyzed_today = []
        last_reset_date = today

# ============ DATA FETCHING ============

def fetch_articles_from_dataset(dataset_id: str) -> list:
    logger.info(f"Fetching from dataset: {dataset_id}")
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    params = {"token": APIFY_TOKEN, "limit": 300, "clean": 1}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    raw_items = response.json()

    articles = []
    for item in raw_items:
        title = item.get("title", "")
        summary = item.get("summary", "") or item.get("description", "")
        link = item.get("link", "")
        scraped_at = item.get("scraped_at", datetime.now().isoformat())

        title = clean_html(str(title))
        summary = clean_html(str(summary))

        if not title or len(title) <= 10:
            continue
        if not is_english(title):
            continue
        if not link:
            continue

        trust_score = get_trust_score(str(link))
        articles.append({
            "title": title,
            "summary": summary,
            "link": str(link),
            "published": str(scraped_at),
            "source": str(link),
            "trust_score": trust_score,
            "trust_label": get_trust_label(trust_score),
            "trust_tier": get_trust_tier(trust_score),
            "scraped_at": datetime.now().isoformat()
        })

    logger.info(f"Fetched {len(articles)} valid English articles")
    return articles

# ============ QDRANT ============

def setup_qdrant_collection():
    collections = qdrant.get_collections().collections
    existing = [c.name for c in collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Created collection {COLLECTION_NAME}")
    else:
        logger.info(f"Collection {COLLECTION_NAME} exists")

def store_articles_in_qdrant(articles: list) -> int:
    global last_article_count
    logger.info(f"Storing {len(articles)} fresh articles in Qdrant...")

    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except:
        pass

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    texts = [f"{a['title']} {a['summary']}" for a in articles]
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32)

    points = []
    for i, (article, embedding) in enumerate(zip(articles, embeddings)):
        point_id = generate_article_id(article["link"], i)
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=article
        ))

    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    last_article_count = len(points)
    logger.info(f"Stored {len(points)} fresh articles")
    return len(points)

def do_full_refresh() -> dict:
    global last_refresh_time
    dataset_id = trigger_fresh_apify_run()
    articles = fetch_articles_from_dataset(dataset_id)
    if not articles:
        raise Exception("No articles returned from Actor")
    stored = store_articles_in_qdrant(articles)
    last_refresh_time = datetime.now().isoformat()
    return {
        "articles_fetched": len(articles),
        "articles_stored": stored,
        "sources_monitored": len(RSS_FEEDS),
        "apify_dataset_id": dataset_id,
        "refreshed_at": last_refresh_time
    }

def search_similar_articles(query: str, limit: int = 15) -> list:
    query_vector = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        score_threshold=0.25
    )
    return results.points

# ============ AI ANALYSIS ============

def analyze_with_ai(articles: list, topic: str) -> dict:
    if not articles:
        return {"error": "No articles found for this topic"}

    articles_text = ""
    for i, a in enumerate(articles[:12]):
        articles_text += f"\nARTICLE {i+1}:\n"
        articles_text += f"Source: {a.get('source', 'unknown')}\n"
        articles_text += f"Trust Score: {a.get('trust_score', 50)}/100\n"
        articles_text += f"Title: {a.get('title', '')}\n"
        articles_text += f"Content: {a.get('summary', '')}\n"
        articles_text += "---"

    prompt = f"""You are a senior conflict journalist and misinformation analyst at a major international newsroom.

Analyze these news articles about: "{topic}"

{articles_text}

You are helping a journalist decide what to verify before publishing. Be specific — name sources, name contradictions, give actionable intelligence. Respond ONLY with valid JSON:
{{
  "misinformation_risk_score": <integer 0-100, calculated based on contradiction severity and source disagreement>,
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "summary": "<2-3 sentences: what is happening RIGHT NOW, where sources disagree, why it matters today>",
  "detected_narratives": [
    "<specific narrative with source name — e.g. BBC reports Iran denied involvement in port attack>",
    "<specific narrative with source name>",
    "<specific narrative with source name>"
  ],
  "contradictions": [
    "<specific factual contradiction naming both sources — e.g. Reuters says ceasefire agreed while Al Jazeera reports fighting continues>",
    "<specific factual contradiction>"
  ],
  "suspicious_patterns": [
    "<specific pattern — e.g. Three low-trust sources amplifying unverified casualty numbers without citing original source>",
    "<specific pattern>"
  ],
  "viral_prediction": "<which specific narrative is spreading fastest, why it could mislead millions if unverified, and which audience is most at risk>",
  "recommended_actions": [
    "<specific journalist action — e.g. Contact Iranian foreign ministry press office to verify the denial claim before publishing>",
    "<specific action — e.g. Cross-check casualty figures with UN OCHA database before amplifying>",
    "<specific action — e.g. Do not publish the sea route closure claim until confirmed by a second independent source>"
  ],
  "story_timeline": [
    "<what was reported first and by which source>",
    "<how the narrative shifted and when>",
    "<current dominant narrative as of now>"
  ],
  "source_analysis": {{
    "most_reliable": "<source name and one reason why>",
    "least_reliable": "<source name and one reason why>",
    "consensus_level": "<HIGH|MEDIUM|LOW>",
    "sources_count": <number of sources analyzed>,
    "key_outlier": "<which source reports something completely different from all others and what they claim>"
  }}
}}"""

    try:
        response = requests.post(
            "https://api.featherless.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {FEATHERLESS_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.3",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a misinformation detection expert for a global newsroom. Always respond with valid JSON only. Be specific, name sources, identify exact contradictions. No markdown, no explanation outside JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 2000
            },
            timeout=45
        )

        result = response.json()
        content = result['choices'][0]['message']['content']
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(content)

    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return {
            "misinformation_risk_score": 50,
            "risk_level": "MEDIUM",
            "detected_narratives": ["Analysis temporarily unavailable"],
            "contradictions": [],
            "suspicious_patterns": [],
            "viral_prediction": "Unable to generate assessment at this time",
            "recommended_actions": ["Verify information with multiple sources"],
            "story_timeline": ["Data being processed"],
            "source_analysis": {
                "most_reliable": "BBC",
                "least_reliable": "Unknown",
                "consensus_level": "MEDIUM",
                "sources_count": len(articles),
                "key_outlier": "None identified"
            },
            "summary": f"Analysis for '{topic}' is being processed. {len(articles)} articles found."
        }

# ============ BACKGROUND AUTO-REFRESH ============

async def auto_refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            logger.info("Auto-refresh triggered...")
            result = do_full_refresh()
            logger.info(f"Auto-refresh complete: {result['articles_stored']} articles stored")
        except Exception as e:
            logger.error(f"Auto-refresh failed: {e}")

# ============ API ENDPOINTS ============

@app.on_event("startup")
async def startup_event():
    global last_refresh_time
    logger.info("Starting Sentinel API v4.0...")
    setup_qdrant_collection()

    try:
        logger.info("Fetching fresh data from Apify Actor on startup...")
        result = do_full_refresh()
        logger.info(f"Startup complete. {result['articles_stored']} fresh articles ready.")
    except Exception as e:
        logger.error(f"Startup refresh failed: {e}. API will still run.")

    asyncio.create_task(auto_refresh_loop())

@app.get("/")
async def root():
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "status": "Sentinel API is running",
        "version": "4.0",
        "articles_in_database": collection_info.points_count,
        "sources_monitored": len(RSS_FEEDS),
        "last_refresh": last_refresh_time,
        "actor_used": YOUR_ACTOR_ID,
        "auto_refresh_interval": f"Every {REFRESH_INTERVAL_SECONDS // 60} minutes",
        "high_risk_detections_today": high_risk_detections_today,
        "endpoints": [
            "/api/live-feed",
            "/api/analyze/{topic}",
            "/api/network/{topic}",
            "/api/refresh",
            "/api/stats",
            "/api/sources"
        ]
    }

@app.get("/api/live-feed")
async def get_live_feed(limit: int = 30):
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        total = collection_info.points_count

        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        articles = []
        for point in results[0]:
            p = point.payload
            articles.append({
                "id": point.id,
                "title": p.get("title", ""),
                "summary": p.get("summary", "")[:200],
                "link": p.get("link", ""),
                "source": p.get("source", ""),
                "published": p.get("scraped_at", ""),
                "trust_score": p.get("trust_score", 50),
                "trust_label": p.get("trust_label", "uncertain"),
                "trust_tier": p.get("trust_tier", "Tier 3 — Monitor"),
                "scraped_at": p.get("scraped_at", "")
            })

        articles.sort(key=lambda x: x.get("scraped_at", ""), reverse=True)

        return {
            "total_articles": total,
            "sources_monitored": len(RSS_FEEDS),
            "returned": len(articles),
            "articles": articles,
            "last_updated": last_refresh_time or datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/{topic}")
async def analyze_topic(topic: str):
    global high_risk_detections_today, topics_analyzed_today
    reset_daily_counters()

    try:
        logger.info(f"Analyzing topic: {topic}")
        similar_results = search_similar_articles(topic, limit=15)

        if not similar_results:
            raise HTTPException(
                status_code=404,
                detail=f"No articles found for topic: {topic}. Try a more specific conflict topic like 'Iran nuclear' or 'Gaza ceasefire'."
            )

        articles = []
        for r in similar_results:
            article = r.payload.copy()
            article["similarity_score"] = round(r.score, 3)
            articles.append(article)

        analysis = analyze_with_ai(articles, topic)
        avg_trust = sum(a.get("trust_score", 50) for a in articles) / len(articles)

        # Dynamic high risk tracking
        risk_level = analysis.get("risk_level", "LOW")
        if risk_level in ["HIGH", "CRITICAL"]:
            high_risk_detections_today += 1

        if topic.lower() not in [t.lower() for t in topics_analyzed_today]:
            topics_analyzed_today.append(topic)

        return {
            "topic": topic,
            "articles_analyzed": len(articles),
            "sources_represented": len(set([
                a.get("source", "").split("/")[2] if len(a.get("source", "").split("/")) > 2 else "unknown"
                for a in articles
            ])),
            "average_trust_score": round(avg_trust, 1),
            "analysis": analysis,
            "articles": articles[:10],
            "analyzed_at": datetime.now().isoformat(),
            "high_risk_detections_today": high_risk_detections_today
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/network/{topic}")
async def get_network(topic: str):
    try:
        similar_results = search_similar_articles(topic, limit=20)

        if not similar_results:
            raise HTTPException(status_code=404, detail="No articles found")

        nodes = []
        edges = []
        seen_sources = {}

        for i, result in enumerate(similar_results):
            p = result.payload
            source_domain = ""
            link = p.get("link", "")
            for domain in SOURCE_TRUST.keys():
                if domain in link:
                    source_domain = domain
                    break

            node = {
                "id": str(result.id),
                "title": p.get("title", ""),
                "source": source_domain or "unknown",
                "trust_score": p.get("trust_score", 50),
                "trust_label": p.get("trust_label", "uncertain"),
                "trust_tier": p.get("trust_tier", "Tier 3 — Monitor"),
                "similarity": round(result.score, 3),
                "size": max(10, int(result.score * 40)),
                "link": link
            }
            nodes.append(node)

            if source_domain not in seen_sources:
                seen_sources[source_domain] = str(result.id)

            if i > 0:
                edges.append({
                    "from": nodes[i-1]["id"],
                    "to": str(result.id),
                    "weight": round(result.score, 3),
                    "label": f"{round(result.score * 100)}% similar"
                })

        return {
            "topic": topic,
            "nodes": nodes,
            "edges": edges,
            "total_sources": len(seen_sources),
            "source_names": list(seen_sources.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/refresh")
async def refresh_data():
    try:
        result = do_full_refresh()
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    reset_daily_counters()
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)

        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False
        )

        trust_distribution = {"trusted": 0, "uncertain": 0, "flagged": 0}
        sources = {}

        for point in results[0]:
            p = point.payload
            label = p.get("trust_label", "uncertain")
            trust_distribution[label] = trust_distribution.get(label, 0) + 1

            link = p.get("link", "")
            for domain in SOURCE_TRUST.keys():
                if domain in link:
                    sources[domain] = sources.get(domain, 0) + 1
                    break

        return {
            "total_articles": collection_info.points_count,
            "sources_monitored": len(RSS_FEEDS),
            "high_risk_detections_today": high_risk_detections_today,
            "topics_analyzed_today": len(topics_analyzed_today),
            "trust_distribution": trust_distribution,
            "articles_by_source": sources,
            "last_refresh": last_refresh_time,
            "actor_id": YOUR_ACTOR_ID,
            "collection_status": "healthy",
            "last_checked": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sources")
async def get_sources():
    """Returns all monitored sources with trust scores — new endpoint for frontend display."""
    sources_list = []
    for feed_url in RSS_FEEDS:
        domain = ""
        for d in SOURCE_TRUST.keys():
            if d in feed_url:
                domain = d
                break
        score = SOURCE_TRUST.get(domain, 55)
        sources_list.append({
            "domain": domain or feed_url.split("/")[2],
            "feed_url": feed_url,
            "trust_score": score,
            "trust_label": get_trust_label(score),
            "trust_tier": get_trust_tier(score)
        })

    sources_list.sort(key=lambda x: x["trust_score"], reverse=True)

    return {
        "total_sources": len(RSS_FEEDS),
        "sources": sources_list,
        "tier_breakdown": {
            "tier_1_premium": len([s for s in sources_list if s["trust_score"] >= 90]),
            "tier_2_reliable": len([s for s in sources_list if 80 <= s["trust_score"] < 90]),
            "tier_3_monitor": len([s for s in sources_list if 65 <= s["trust_score"] < 80]),
            "tier_4_caution": len([s for s in sources_list if s["trust_score"] < 65]),
        }
    }

# ============ RUN SERVER ============
if __name__ == "__main__":
    print("\n" + "="*50)
    print("SENTINEL API v4.0 STARTING")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
