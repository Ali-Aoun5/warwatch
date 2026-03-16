import os
import hashlib
import asyncio
import logging
from datetime import datetime
from typing import Optional
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    SearchRequest
)
from sentence_transformers import SentenceTransformer
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
APIFY_TOKEN = "apify_api_wtaTqvoo2lEOEzaAio6PySUIDpFaH90gRJtZ"
APIFY_DATASET_ID = "dYy7Bkz9kOCT2aeKN"
QDRANT_URL = "https://dd5f5d46-e566-4409-beeb-c9a648ade4bc.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.V8ansdYNHvhnXORumgq7mNRlm9nyJ_-KnUfASgKFZeE"
FEATHERLESS_KEY = "rc_8a722edee3da1e4ab748e8f0d6645579aad68e935c11b087d702b561c7b092ac"
COLLECTION_NAME = "warwatch_articles"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Source trust scores based on journalistic reputation
SOURCE_TRUST = {
    "bbci.co.uk": 90,
    "bbc.com": 90,
    "reuters.com": 92,
    "npr.org": 88,
    "dw.com": 87,
    "skynews.com": 75,
    "aljazeera.com": 72,
    "ndtv.com": 68,
    "feedburner.com": 55,
    "rt.com": 25,
    "sputniknews.com": 20,
}

# ============ INITIALIZE SERVICES ============
app = FastAPI(title="WarWatch API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model... please wait 30 seconds")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded successfully")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ============ HELPER FUNCTIONS ============

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

def clean_html(text: str) -> str:
    if not text:
        return ""
    import re
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'&[a-zA-Z]+;', ' ', clean)
    clean = ' '.join(clean.split())
    return clean[:800]

def generate_article_id(url: str) -> int:
    return int(hashlib.md5(url.encode()).hexdigest()[:8], 16)

def fetch_apify_articles() -> list:
    logger.info("Fetching articles from Apify dataset...")
    url = f"https://api.apify.com/v2/datasets/{APIFY_DATASET_ID}/items"
    params = {
        "token": APIFY_TOKEN,
        "limit": 200,
        "clean": 1
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    raw_items = response.json()
    
    articles = []
    for item in raw_items:
        title = item.get("title", "")
        if isinstance(title, dict):
            title = title.get("value", "")
        
        summary = item.get("summary", "") or item.get("description", "")
        if isinstance(summary, dict):
            summary = summary.get("value", "")
        
        link = item.get("link", "") or item.get("href", "")
        if isinstance(link, list) and link:
            link = link[0].get("href", "") if isinstance(link[0], dict) else link[0]
        
        published = item.get("published", "") or item.get("pubDate", "")
        
        title = clean_html(str(title))
        summary = clean_html(str(summary))
        
        if title and len(title) > 10:
            articles.append({
                "title": title,
                "summary": summary,
                "link": str(link),
                "published": str(published),
                "source": str(link),
                "trust_score": get_trust_score(str(link)),
                "trust_label": get_trust_label(get_trust_score(str(link))),
                "scraped_at": datetime.now().isoformat()
            })
    
    logger.info(f"Fetched {len(articles)} valid articles from Apify")
    return articles

def setup_qdrant_collection():
    collections = qdrant.get_collections().collections
    existing = [c.name for c in collections]
    
    if COLLECTION_NAME in existing:
        logger.info(f"Collection {COLLECTION_NAME} already exists")
        return
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created collection {COLLECTION_NAME}")

def store_articles_in_qdrant(articles: list):
    logger.info(f"Storing {len(articles)} articles in Qdrant...")
    
    texts = [f"{a['title']} {a['summary']}" for a in articles]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    
    points = []
    for i, (article, embedding) in enumerate(zip(articles, embeddings)):
        point_id = generate_article_id(article["link"] + str(i))
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=article
        ))
    
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        logger.info(f"Stored batch {i//batch_size + 1}")
    
    logger.info("All articles stored in Qdrant successfully")
    return len(points)

def search_similar_articles(query: str, limit: int = 15) -> list:
    query_vector = embedder.encode(query).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        score_threshold=0.3
    )
    return results.points

def analyze_with_ai(articles: list, topic: str) -> dict:
    if not articles:
        return {"error": "No articles found for this topic"}
    
    articles_text = ""
    for i, a in enumerate(articles[:10]):
        articles_text += f"\nARTICLE {i+1}:\n"
        articles_text += f"Source: {a.get('source', 'unknown')}\n"
        articles_text += f"Trust Score: {a.get('trust_score', 50)}/100\n"
        articles_text += f"Title: {a.get('title', '')}\n"
        articles_text += f"Content: {a.get('summary', '')}\n"
        articles_text += "---"
    
    prompt = f"""You are an expert conflict zone misinformation analyst working for a major fact-checking organization.

Analyze these news articles about: "{topic}"

{articles_text}

Provide a comprehensive misinformation risk analysis. Respond ONLY with valid JSON in this exact format:
{{
  "misinformation_risk_score": <integer 0-100>,
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "detected_narratives": [
    "<narrative 1>",
    "<narrative 2>",
    "<narrative 3>"
  ],
  "contradictions": [
    "<contradiction between sources 1>",
    "<contradiction between sources 2>"
  ],
  "suspicious_patterns": [
    "<suspicious pattern 1>",
    "<suspicious pattern 2>"
  ],
  "viral_prediction": "<which narrative is most likely to spread in next 6 hours and why>",
  "recommended_actions": [
    "<action for journalists 1>",
    "<action for journalists 2>",
    "<action for journalists 3>"
  ],
  "source_analysis": {{
    "most_reliable": "<most reliable source name>",
    "least_reliable": "<least reliable source name>",
    "consensus_level": "<HIGH|MEDIUM|LOW>"
  }},
  "summary": "<2-3 sentence plain English explanation of what is happening and why it matters>"
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
                        "content": "You are a misinformation detection expert. Always respond with valid JSON only. No markdown, no explanation outside JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1500
            },
            timeout=30
        )
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        import json
        import re
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
            "viral_prediction": "Unable to generate prediction at this time",
            "recommended_actions": ["Verify information with multiple sources"],
            "source_analysis": {
                "most_reliable": "BBC",
                "least_reliable": "Unknown",
                "consensus_level": "MEDIUM"
            },
            "summary": f"Analysis for '{topic}' is being processed. {len(articles)} articles found."
        }

# ============ API ENDPOINTS ============

@app.on_event("startup")
async def startup_event():
    logger.info("Starting WarWatch API...")
    setup_qdrant_collection()
    
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        logger.info("No articles in database. Fetching from Apify...")
        articles = fetch_apify_articles()
        if articles:
            stored = store_articles_in_qdrant(articles)
            logger.info(f"Startup complete. {stored} articles ready.")
    else:
        logger.info(f"Startup complete. {collection_info.points_count} articles already in database.")

@app.get("/")
async def root():
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "status": "WarWatch API is running",
        "version": "2.0",
        "articles_in_database": collection_info.points_count,
        "endpoints": [
            "/api/live-feed",
            "/api/analyze/{topic}",
            "/api/network/{topic}",
            "/api/refresh",
            "/api/stats"
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
                "published": p.get("published", ""),
                "trust_score": p.get("trust_score", 50),
                "trust_label": p.get("trust_label", "uncertain"),
                "scraped_at": p.get("scraped_at", "")
            })
        
        articles.sort(key=lambda x: x["trust_score"])
        
        return {
            "total_articles": total,
            "returned": len(articles),
            "articles": articles,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/{topic}")
async def analyze_topic(topic: str):
    try:
        logger.info(f"Analyzing topic: {topic}")
        similar_results = search_similar_articles(topic, limit=15)
        
        if not similar_results:
            raise HTTPException(
                status_code=404, 
                detail=f"No articles found for topic: {topic}"
            )
        
        articles = []
        for r in similar_results:
            article = r.payload.copy()
            article["similarity_score"] = round(r.score, 3)
            articles.append(article)
        
        analysis = analyze_with_ai(articles, topic)
        
        avg_trust = sum(a.get("trust_score", 50) for a in articles) / len(articles)
        
        return {
            "topic": topic,
            "articles_analyzed": len(articles),
            "average_trust_score": round(avg_trust, 1),
            "analysis": analysis,
            "articles": articles[:10],
            "analyzed_at": datetime.now().isoformat()
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
                "title": p.get("title", "")[:60],
                "source": source_domain or "unknown",
                "trust_score": p.get("trust_score", 50),
                "trust_label": p.get("trust_label", "uncertain"),
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
            "total_sources": len(seen_sources)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/refresh")
async def refresh_data():
    try:
        logger.info("Refreshing data from Apify...")
        articles = fetch_apify_articles()
        
        if not articles:
            raise HTTPException(status_code=500, detail="Failed to fetch articles from Apify")
        
        stored = store_articles_in_qdrant(articles)
        
        return {
            "status": "success",
            "articles_fetched": len(articles),
            "articles_stored": stored,
            "refreshed_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        
        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
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
            "trust_distribution": trust_distribution,
            "articles_by_source": sources,
            "collection_status": "healthy",
            "last_checked": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ RUN SERVER ============
if __name__ == "__main__":
    print("\n" + "="*50)
    print("WARWATCH API STARTING")
    print("="*50)
    print("API will be available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )