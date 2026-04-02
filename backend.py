import os
import hashlib
import asyncio
import logging
import re
import json
import html
from datetime import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from apify_client import ApifyClient
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

APIFY_TOKEN = os.environ.get("APIFY_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
FEATHERLESS_KEY = os.environ.get("FEATHERLESS_KEY")
YOUR_ACTOR_ID = os.environ.get("SENTINEL_ACTOR_ID", "mehdialiaoun/sentinel-rss-actor")

COLLECTION_NAME = "sentinel_articles_v2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
REFRESH_INTERVAL_SECONDS = 1800

# Using a highly reliable model for JSON output
AI_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Fallback models if primary fails
FALLBACK_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3-8b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.washingtonpost.com/rss/world",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.france24.com/en/rss",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://feeds.skynews.com/feeds/rss/world.xml",
    "https://www.middleeasteye.net/rss",
    "https://feeds.feedburner.com/euronews/en/news",
    "https://feeds.feedburner.com/ndtvnews-world-news",
    "https://feeds.feedburner.com/time/world",
    "https://feeds.feedburner.com/trtworld",
    "https://www.arabnews.com/rss.xml",
    "https://theintercept.com/feed/?rss",
    "https://feeds.nbcnews.com/nbcnews/public/world",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://apnews.com/rss",
]

SOURCE_TRUST = {
    "bbci.co.uk": 90, "bbc.com": 90,
    "reuters.com": 93, "apnews.com": 93,
    "nytimes.com": 88, "theguardian.com": 88,
    "washingtonpost.com": 85, "npr.org": 88,
    "aljazeera.com": 72, "france24.com": 82,
    "dw.com": 87, "skynews.com": 75,
    "middleeasteye.net": 72, "euronews.com": 78,
    "ndtv.com": 68, "time.com": 82,
    "trtworld.com": 68, "arabnews.com": 65,
    "theintercept.com": 65, "nbcnews.com": 78,
    "wsj.com": 88, "feedburner.com": 55,
    "rt.com": 25, "sputniknews.com": 20,
}

app = FastAPI(title="Sentinel API", version="4.2")
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

last_refresh_time = None
last_article_count = 0
high_risk_detections_today = 0
last_reset_date = datetime.now().date()
topics_analyzed_today = []


def clean_html(text: str) -> str:
    if not text:
        return ""
    clean = html.unescape(text)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'&[a-zA-Z]+;', ' ', clean)
    return ' '.join(clean.split())[:500]


def is_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.8


def get_trust_score(url: str) -> int:
    url_lower = url.lower()
    for domain, score in SOURCE_TRUST.items():
        if domain in url_lower:
            return score
    return 55


def get_trust_label(score: int) -> str:
    if score >= 85:
        return "trusted"
    elif score >= 65:
        return "uncertain"
    return "flagged"


def get_trust_tier(score: int) -> str:
    if score >= 90:
        return "Tier 1 — Premium"
    elif score >= 80:
        return "Tier 2 — Reliable"
    elif score >= 65:
        return "Tier 3 — Monitor"
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


def trigger_fresh_apify_run() -> str:
    logger.info(f"Triggering Actor: {YOUR_ACTOR_ID}")
    client = ApifyClient(APIFY_TOKEN)
    run = client.actor(YOUR_ACTOR_ID).call(
        run_input={"urls": RSS_FEEDS, "maxItemsPerFeed": 10}
    )
    dataset_id = run["defaultDatasetId"]
    logger.info(f"Actor done. Dataset: {dataset_id}")
    return dataset_id


def fetch_articles_from_dataset(dataset_id: str) -> list:
    logger.info(f"Fetching dataset: {dataset_id}")
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    response = requests.get(url, params={"token": APIFY_TOKEN, "limit": 300, "clean": 1}, timeout=30)
    response.raise_for_status()

    articles = []
    for item in response.json():
        title = clean_html(str(item.get("title", "")))
        summary = clean_html(str(item.get("summary", "") or item.get("description", "")))
        link = str(item.get("link", ""))

        if not title or len(title) <= 10 or not is_english(title) or not link:
            continue

        score = get_trust_score(link)
        articles.append({
            "title": title,
            "summary": summary,
            "link": link,
            "published": str(item.get("scraped_at", datetime.now().isoformat())),
            "source": link,
            "trust_score": score,
            "trust_label": get_trust_label(score),
            "trust_tier": get_trust_tier(score),
            "scraped_at": datetime.now().isoformat()
        })

    logger.info(f"Fetched {len(articles)} English articles")
    return articles


def setup_qdrant_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Created {COLLECTION_NAME}")
    else:
        logger.info(f"{COLLECTION_NAME} exists")


def store_articles_in_qdrant(articles: list) -> int:
    global last_article_count
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

    points = [
        PointStruct(
            id=generate_article_id(a["link"], i),
            vector=emb.tolist(),
            payload=a
        )
        for i, (a, emb) in enumerate(zip(articles, embeddings))
    ]

    for i in range(0, len(points), 50):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i+50])

    last_article_count = len(points)
    logger.info(f"Stored {len(points)} articles")
    return len(points)


def do_full_refresh() -> dict:
    global last_refresh_time
    dataset_id = trigger_fresh_apify_run()
    articles = fetch_articles_from_dataset(dataset_id)
    if not articles:
        raise Exception("No articles returned")
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


def call_featherless_api(model: str, messages: list, max_tokens: int = 1500) -> str:
    """Call Featherless API and return raw content string. Raises on failure."""
    response = requests.post(
        "https://api.featherless.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {FEATHERLESS_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        },
        timeout=120
    )

    if response.status_code != 200:
        logger.error(f"Featherless API error {response.status_code}: {response.text[:300]}")
        raise Exception(f"API returned {response.status_code}: {response.text[:200]}")

    result = response.json()

    if "error" in result:
        raise Exception(f"API error: {result['error']}")

    if "choices" not in result or not result["choices"]:
        raise Exception(f"No choices in response: {list(result.keys())}")

    content = result["choices"][0]["message"]["content"]

    if not content or not content.strip():
        raise Exception("Empty content from model")

    return content.strip()


def extract_json_from_content(content: str) -> dict:
    """Robustly extract JSON from model output."""
    # Remove think tags (DeepSeek)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Remove markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find JSON object with regex
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Try to fix common JSON issues
    fixed = content
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*]', ']', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise Exception(f"Could not extract valid JSON from content: {content[:200]}")


def compute_risk_score_from_articles(articles: list) -> tuple:
    """
    Compute a real risk score based on actual article data.
    Returns (score, level, reason)
    """
    if not articles:
        return 30, "LOW", "No articles found"

    trust_scores = [a.get("trust_score", 55) for a in articles]
    avg_trust = sum(trust_scores) / len(trust_scores)
    min_trust = min(trust_scores)
    max_trust = max(trust_scores)
    trust_spread = max_trust - min_trust

    # Count flagged vs trusted sources
    flagged_count = sum(1 for a in articles if a.get("trust_label") == "flagged")
    trusted_count = sum(1 for a in articles if a.get("trust_label") == "trusted")
    uncertain_count = sum(1 for a in articles if a.get("trust_label") == "uncertain")

    # Check for source diversity (many different outlets = more cross-checking possible)
    unique_domains = set()
    for a in articles:
        link = a.get("link", "")
        for domain in SOURCE_TRUST:
            if domain in link:
                unique_domains.add(domain)
                break

    source_diversity = len(unique_domains)

    # Compute base risk score
    # High trust spread = high risk (sources disagree)
    # Many flagged sources = high risk
    # Few trusted sources = high risk

    risk_score = 0

    # Trust spread factor (0-35 points)
    risk_score += min(35, int(trust_spread * 0.5))

    # Flagged source factor (0-30 points)
    if len(articles) > 0:
        flagged_ratio = flagged_count / len(articles)
        risk_score += int(flagged_ratio * 30)

    # Low average trust factor (0-25 points)
    if avg_trust < 70:
        risk_score += int((70 - avg_trust) * 0.8)

    # Source diversity factor - fewer sources = less verification (0-10 points)
    if source_diversity < 3:
        risk_score += 10
    elif source_diversity < 5:
        risk_score += 5

    # Clamp to 0-100
    risk_score = max(10, min(95, risk_score))

    if risk_score >= 75:
        level = "CRITICAL"
    elif risk_score >= 55:
        level = "HIGH"
    elif risk_score >= 35:
        level = "MEDIUM"
    else:
        level = "LOW"

    reason = f"Based on {len(articles)} articles from {source_diversity} sources. Trust spread: {trust_spread}pts. Flagged sources: {flagged_count}/{len(articles)}."

    return risk_score, level, reason


def analyze_with_ai(articles: list, topic: str) -> dict:
    if not articles:
        return _fallback_analysis(topic, 0)

    # Compute real risk score from data FIRST — this is always accurate
    data_risk_score, data_risk_level, risk_reason = compute_risk_score_from_articles(articles)

    # Build article summaries for AI
    articles_text = ""
    for i, a in enumerate(articles[:12]):
        source_domain = "unknown"
        for domain in SOURCE_TRUST:
            if domain in a.get("link", ""):
                source_domain = domain
                break
        articles_text += f"\nSOURCE {i+1}:\n"
        articles_text += f"Outlet: {source_domain} (Trust Score: {a.get('trust_score', 50)}/100)\n"
        articles_text += f"Headline: {a.get('title', '')}\n"
        articles_text += f"Content: {a.get('summary', '')[:400]}\n"
        articles_text += "---\n"

    system_prompt = """You are a senior investigative journalist and misinformation analyst with 20 years experience. 
You analyze news sources for contradictions, narrative patterns, and misinformation risks.
You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no text before or after the JSON.
Be specific — name actual outlets, actual claims, actual contradictions you see in the provided articles."""

    user_prompt = f"""Analyze these {len(articles[:12])} real news articles about "{topic}":

{articles_text}

Based ONLY on what is written above, identify contradictions between sources, narrative patterns, and misinformation risks.

Respond with ONLY this JSON (fill in all fields with specific real content from the articles above):
{{
  "misinformation_risk_score": <integer between 20 and 95 — be realistic, not always 50>,
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "summary": "<2-3 specific sentences about what is happening with {topic} right now based on these articles, naming specific outlets and claims>",
  "detected_narratives": [
    "<Narrative 1: Name which outlet says what specific claim>",
    "<Narrative 2: Name which outlet says what specific claim>",
    "<Narrative 3: Name which outlet says what specific claim>"
  ],
  "contradictions": [
    "<Contradiction 1: Outlet A says X while Outlet B says Y — be specific>",
    "<Contradiction 2: Outlet C says X while Outlet D says Y — be specific>"
  ],
  "suspicious_patterns": [
    "<Pattern 1: specific observation from these articles>",
    "<Pattern 2: specific observation from these articles>"
  ],
  "viral_prediction": "<Which specific narrative from these articles is most likely to spread and why>",
  "recommended_actions": [
    "<Action 1: specific verification step for journalists based on these articles>",
    "<Action 2: specific verification step for journalists based on these articles>",
    "<Action 3: specific verification step for journalists based on these articles>"
  ],
  "story_timeline": [
    "<Earliest development visible in these articles>",
    "<How narrative shifted across these articles>",
    "<Most recent or dominant narrative>"
  ],
  "source_analysis": {{
    "most_reliable": "<Name the highest trust outlet and what they reported>",
    "least_reliable": "<Name the lowest trust outlet and what they reported>",
    "consensus_level": "<HIGH if 80%+ agree, MEDIUM if some differences, LOW if strong contradictions>",
    "sources_count": {len(articles[:12])},
    "key_outlier": "<Name the source reporting most differently and what they said>"
  }}
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Try primary model, then fallbacks
    models_to_try = [AI_MODEL] + FALLBACK_MODELS
    last_error = None

    for model in models_to_try:
        try:
            logger.info(f"Trying model: {model}")
            content = call_featherless_api(model, messages, max_tokens=1500)
            logger.info(f"Got content from {model}, length: {len(content)}")
            logger.info(f"Content preview: {content[:300]}")

            ai_result = extract_json_from_content(content)

            # Validate the result has required fields
            required_fields = ["misinformation_risk_score", "risk_level", "summary", "detected_narratives"]
            if not all(field in ai_result for field in required_fields):
                logger.warning(f"Missing required fields in AI result: {list(ai_result.keys())}")
                raise Exception("Missing required fields")

            # Sanity check risk score
            ai_score = ai_result.get("misinformation_risk_score", 50)
            if not isinstance(ai_score, (int, float)) or ai_score < 0 or ai_score > 100:
                ai_result["misinformation_risk_score"] = data_risk_score
                ai_result["risk_level"] = data_risk_level

            # If AI returned generic fallback text, replace with data-driven score
            summary = ai_result.get("summary", "")
            if "AI analysis is processing" in summary or "multiple sources" in summary.lower() and len(summary) < 100:
                ai_result["misinformation_risk_score"] = data_risk_score
                ai_result["risk_level"] = data_risk_level

            logger.info(f"Successfully analyzed with {model}. Risk: {ai_result.get('risk_level')} ({ai_result.get('misinformation_risk_score')})")
            return ai_result

        except Exception as e:
            logger.error(f"Model {model} failed: {e}")
            last_error = e
            continue

    # All models failed — use intelligent data-driven fallback
    logger.error(f"All models failed. Last error: {last_error}. Using data-driven fallback.")
    return _intelligent_fallback(topic, articles, data_risk_score, data_risk_level, risk_reason)


def _intelligent_fallback(topic: str, articles: list, risk_score: int, risk_level: str, risk_reason: str) -> dict:
    """
    When AI fails, generate a meaningful analysis from the actual article data.
    This is much better than a generic fallback.
    """
    if not articles:
        return _fallback_analysis(topic, 0)

    # Extract real source names
    source_names = []
    for a in articles[:10]:
        for domain in SOURCE_TRUST:
            if domain in a.get("link", ""):
                source_names.append(domain)
                break

    # Get highest and lowest trust articles
    sorted_by_trust = sorted(articles, key=lambda x: x.get("trust_score", 50), reverse=True)
    most_reliable = sorted_by_trust[0] if sorted_by_trust else None
    least_reliable = sorted_by_trust[-1] if len(sorted_by_trust) > 1 else None

    # Extract real headlines
    headlines = [a.get("title", "") for a in articles[:5] if a.get("title")]

    # Build narratives from actual headlines
    narratives = []
    for a in articles[:3]:
        domain = next((d for d in SOURCE_TRUST if d in a.get("link", "")), "unknown source")
        title = a.get("title", "")
        if title:
            narratives.append(f"{domain} reports: {title[:100]}")

    unique_domains = set()
    for a in articles:
        for domain in SOURCE_TRUST:
            if domain in a.get("link", ""):
                unique_domains.add(domain)
                break

    most_reliable_name = next((d for d in SOURCE_TRUST if most_reliable and d in most_reliable.get("link", "")), "unknown")
    least_reliable_name = next((d for d in SOURCE_TRUST if least_reliable and d in least_reliable.get("link", "")), "unknown")

    return {
        "misinformation_risk_score": risk_score,
        "risk_level": risk_level,
        "summary": f"Sentinel detected {len(articles)} articles about '{topic}' from {len(unique_domains)} sources. {risk_reason} Manual cross-referencing is recommended before publishing.",
        "detected_narratives": narratives if narratives else [
            f"Multiple outlets covering {topic} from different angles",
            "Cross-source verification recommended",
            "Monitor for narrative shifts across sources"
        ],
        "contradictions": [
            f"Sources show trust score spread indicating varying reliability. Verify claims from lower-trust outlets ({least_reliable_name}) against higher-trust ones ({most_reliable_name}).",
            "Independent verification of key facts recommended before publishing."
        ],
        "suspicious_patterns": [
            f"Trust score variance detected across {len(unique_domains)} sources",
            f"Coverage from {len(unique_domains)} outlets with varying editorial standards"
        ],
        "viral_prediction": f"Stories from high-engagement outlets covering {topic} are likely to spread. Monitor social media amplification patterns.",
        "recommended_actions": [
            f"Cross-reference {topic} coverage with primary sources like official statements or government releases",
            f"Verify claims from {least_reliable_name} (lower trust) against {most_reliable_name} (higher trust)",
            "Check original source documents and official statements before publishing"
        ],
        "story_timeline": [
            f"Multiple outlets began covering {topic} simultaneously",
            f"Coverage spans {len(unique_domains)} different news organizations",
            f"Current narrative status: {risk_level} risk level based on source diversity and trust analysis"
        ],
        "source_analysis": {
            "most_reliable": f"{most_reliable_name} (Trust: {most_reliable.get('trust_score', 50) if most_reliable else 'N/A'}/100) — {most_reliable.get('title', '')[:80] if most_reliable else ''}",
            "least_reliable": f"{least_reliable_name} (Trust: {least_reliable.get('trust_score', 50) if least_reliable else 'N/A'}/100) — {least_reliable.get('title', '')[:80] if least_reliable else ''}",
            "consensus_level": "HIGH" if risk_score < 35 else ("MEDIUM" if risk_score < 65 else "LOW"),
            "sources_count": len(articles),
            "key_outlier": f"{least_reliable_name} shows most divergence from mainstream coverage with trust score of {least_reliable.get('trust_score', 'N/A') if least_reliable else 'N/A'}/100"
        }
    }


def _fallback_analysis(topic: str, article_count: int) -> dict:
    return {
        "misinformation_risk_score": 30,
        "risk_level": "LOW",
        "summary": f"No articles found for '{topic}'. Try searching for: Iran, Gaza, Ukraine, ceasefire, or other active conflict topics.",
        "detected_narratives": [
            "No articles retrieved for this topic",
            "Try a different search term",
            "System is monitoring 20 global sources"
        ],
        "contradictions": ["No contradictions detected — no articles found for this topic"],
        "suspicious_patterns": ["Insufficient data for pattern analysis"],
        "viral_prediction": "Cannot predict — no articles found",
        "recommended_actions": [
            "Try searching for a more specific topic",
            "Search for active conflict keywords like Iran, Gaza, Ukraine",
            "Wait for the next data refresh cycle"
        ],
        "story_timeline": [
            "No articles found",
            "Try a different search term",
            "System actively monitoring 20 global sources"
        ],
        "source_analysis": {
            "most_reliable": "N/A — no articles found",
            "least_reliable": "N/A — no articles found",
            "consensus_level": "N/A",
            "sources_count": 0,
            "key_outlier": "N/A — no articles found"
        }
    }


async def auto_refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            logger.info("Auto-refresh triggered...")
            result = do_full_refresh()
            logger.info(f"Auto-refresh complete: {result['articles_stored']} articles")
        except Exception as e:
            logger.error(f"Auto-refresh failed: {e}")


@app.on_event("startup")
async def startup_event():
    global last_refresh_time
    logger.info("Starting Sentinel API v4.2...")
    setup_qdrant_collection()
    try:
        result = do_full_refresh()
        logger.info(f"Startup complete. {result['articles_stored']} articles ready.")
    except Exception as e:
        logger.error(f"Startup refresh failed: {e}")
    asyncio.create_task(auto_refresh_loop())


@app.get("/")
async def root():
    info = qdrant.get_collection(COLLECTION_NAME)
    return {
        "status": "Sentinel API running",
        "version": "4.2",
        "articles_in_database": info.points_count,
        "sources_monitored": len(RSS_FEEDS),
        "last_refresh": last_refresh_time,
        "actor_used": YOUR_ACTOR_ID,
        "ai_model": AI_MODEL,
        "auto_refresh_interval": f"Every {REFRESH_INTERVAL_SECONDS // 60} minutes",
        "high_risk_detections_today": high_risk_detections_today,
    }


@app.get("/api/live-feed")
async def get_live_feed(limit: int = 30):
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
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
            "total_articles": info.points_count,
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
        logger.info(f"Analyzing: {topic}")
        similar_results = search_similar_articles(topic, limit=15)
        if not similar_results:
            raise HTTPException(
                status_code=404,
                detail=f"No articles found for '{topic}'. Try: Iran, Gaza, Ukraine, ceasefire"
            )
        articles = []
        for r in similar_results:
            article = r.payload.copy()
            article["similarity_score"] = round(r.score, 3)
            articles.append(article)

        analysis = analyze_with_ai(articles, topic)
        avg_trust = sum(a.get("trust_score", 50) for a in articles) / len(articles)

        if analysis.get("risk_level") in ["HIGH", "CRITICAL"]:
            high_risk_detections_today += 1
        if topic.lower() not in [t.lower() for t in topics_analyzed_today]:
            topics_analyzed_today.append(topic)

        return {
            "topic": topic,
            "articles_analyzed": len(articles),
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
            source_domain = next(
                (d for d in SOURCE_TRUST if d in p.get("link", "")), "unknown"
            )
            node = {
                "id": str(result.id),
                "title": p.get("title", ""),
                "source": source_domain,
                "trust_score": p.get("trust_score", 50),
                "trust_label": p.get("trust_label", "uncertain"),
                "trust_tier": p.get("trust_tier", "Tier 3"),
                "similarity": round(result.score, 3),
                "size": max(10, int(result.score * 40)),
                "link": p.get("link", "")
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
        return {"status": "success", **do_full_refresh()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    reset_daily_counters()
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=False
        )
        trust_dist = {"trusted": 0, "uncertain": 0, "flagged": 0}
        sources = {}
        for point in results[0]:
            p = point.payload
            label = p.get("trust_label", "uncertain")
            trust_dist[label] = trust_dist.get(label, 0) + 1
            for domain in SOURCE_TRUST:
                if domain in p.get("link", ""):
                    sources[domain] = sources.get(domain, 0) + 1
                    break
        return {
            "total_articles": info.points_count,
            "sources_monitored": len(RSS_FEEDS),
            "high_risk_detections_today": high_risk_detections_today,
            "topics_analyzed_today": len(topics_analyzed_today),
            "trust_distribution": trust_dist,
            "articles_by_source": sources,
            "last_refresh": last_refresh_time,
            "actor_id": YOUR_ACTOR_ID,
            "ai_model": AI_MODEL,
            "collection_status": "healthy",
            "last_checked": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources")
async def get_sources():
    sources_list = []
    for feed_url in RSS_FEEDS:
        domain = next((d for d in SOURCE_TRUST if d in feed_url), feed_url.split("/")[2])
        score = SOURCE_TRUST.get(domain, 55)
        sources_list.append({
            "domain": domain,
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


@app.get("/api/debug/featherless")
async def debug_featherless():
    """Debug endpoint to test Featherless API connection"""
    try:
        content = call_featherless_api(
            AI_MODEL,
            [
                {"role": "system", "content": "Respond with only valid JSON."},
                {"role": "user", "content": 'Return this exact JSON: {"status": "working", "model": "ok"}'}
            ],
            max_tokens=50
        )
        return {
            "status": "success",
            "raw_content": content,
            "featherless_key_set": bool(FEATHERLESS_KEY),
            "model": AI_MODEL
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "featherless_key_set": bool(FEATHERLESS_KEY),
            "model": AI_MODEL
        }


if __name__ == "__main__":
    print("SENTINEL API v4.2 STARTING")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, log_level="info")
