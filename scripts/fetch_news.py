import requests
import json
import time
import re
from pathlib import Path
from datetime import datetime, timedelta

API_KEY = "8b1050127596431c9af1437165f2c7eb"
URL = "https://newsapi.org/v2/everything"

# Expanded domain list for better coverage
DOMAINS = ",".join([
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "reuters.com",
    "seekingalpha.com",
])

# More targeted query terms
QUERY = '("stock market" OR "equity market" OR earnings OR "shares" OR volatility OR IPO OR "price target") -recipe -coupon -deal'

# Categorized market terms with weights
STRONG_MARKET_INDICATORS = {
    "stock", "stocks", "equity", "equities", "shares", "market",
    "s&p 500", "s&p", "dow jones", "dow", "nasdaq", "russell",
    "earnings", "eps", "revenue", "guidance", "forecast",
    "price target", "analyst rating", "upgrade", "downgrade",
    "ipo", "stock split", "dividend", "buyback",
    "volatility", "vix", "options", "futures",
    "bull market", "bear market", "correction", "crash",
}

MODERATE_MARKET_INDICATORS = {
    "index", "indexes", "indices", "bond", "bonds", "yield",
    "treasury", "treasuries", "fed", "federal reserve",
    "rate cut", "rate hike", "interest rate", "inflation",
    "gdp", "cpi", "ppi", "unemployment",
    "rally", "selloff", "decline", "surge", "plunge",
    "trading", "investor", "investors", "valuation",
    "sector", "hedge fund", "mutual fund", "etf",
}

# Company/ticker patterns
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b(?:\s+stock|\s+shares)?')
COMPANY_SUFFIXES = ["inc", "corp", "ltd", "plc", "llc"]

# Hard exclusions
EXCLUDE_PATTERNS = [
    # Shopping/consumer
    r'\b(deal|deals|coupon|discount|sale|black friday|cyber monday|gift guide|shopping)\b',
    # Lifestyle
    r'\b(recipe|recipes|fashion|lifestyle|beauty|travel|hotel|restaurant)\b',
    # HR/compensation (unless about public disclosure)
    r'\b(salary|compensation|ceo pay|benefits|hiring)\b(?!.*\b(disclose|report|filing)\b)',
    # Reviews/guides (unless financial product)
    r'\b(review|guide|how to|tips for)\b(?!.*\b(invest|stock|trading)\b)',
    # Entertainment
    r'\b(movie|film|game|gaming|sports|music|celebrity)\b(?!.*\b(stock|share|ipo)\b)',
]

def calculate_relevance_score(title, desc):
    """Score article relevance based on market terminology density."""
    text = (title + " " + desc).lower()
    score = 0
    
    # Check for hard exclusions first
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return -1  # Immediate rejection
    
    # Strong indicators worth 3 points each
    for term in STRONG_MARKET_INDICATORS:
        if term in text:
            score += 3
            if term in title.lower():  # Bonus if in title
                score += 2
    
    # Moderate indicators worth 1 point
    for term in MODERATE_MARKET_INDICATORS:
        if term in text:
            score += 1
    
    # Ticker symbols worth 2 points
    if TICKER_PATTERN.search(title):
        score += 2
    
    # Company name indicators
    for suffix in COMPANY_SUFFIXES:
        if suffix in text:
            score += 1
            break
    
    return score

def is_market_article(title, desc, threshold=3):
    """Determine if article is market-related based on score threshold."""
    if not title.strip():
        return False
    
    score = calculate_relevance_score(title, desc)
    return score >= threshold

def fetch_market_news(days_back=7, min_score=3, save_all=False):
    """
    Fetch market news with configurable parameters.
    
    Args:
        days_back: Number of days to look back
        min_score: Minimum relevance score (default 3)
        save_all: If True, save both filtered and raw results
    """
    print(f"[FETCH] Market news from last {days_back} days (min_score={min_score})")
    
    # Calculate date range
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days_back)
    
    params = {
        "q": QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "domains": DOMAINS,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
    }
    
    headers = {"X-Api-Key": API_KEY}
    
    try:
        response = requests.get(URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        raw = response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed - {e}")
        return
    
    if "articles" not in raw:
        print(f"ERROR: {raw.get('message', 'Unknown error')}")
        return
    
    print(f"[RAW] Retrieved {len(raw['articles'])} articles")
    
    # Score and filter articles
    scored_articles = []
    for art in raw["articles"]:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        
        score = calculate_relevance_score(title, desc)
        if score >= min_score:
            art["relevance_score"] = score
            scored_articles.append(art)
    
    # Sort by relevance score
    scored_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    print(f"[FILTERED] {len(scored_articles)} articles passed threshold")
    
    # Score distribution
    if scored_articles:
        scores = [a["relevance_score"] for a in scored_articles]
        print(f"[SCORES] Min: {min(scores)}, Max: {max(scores)}, Avg: {sum(scores)/len(scores):.1f}")
    
    # Save results
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    output_file = f"data/raw/news_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(scored_articles, f, indent=2)
    print(f"[SAVED] {output_file}")
    
    if save_all:
        raw_file = f"data/raw/news_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(raw_file, "w") as f:
            json.dump(raw["articles"], f, indent=2)
        print(f"[SAVED] {raw_file}")
    
    # Sample output for review
    print("\n[SAMPLE] Top 3 articles:")
    for i, art in enumerate(scored_articles[:3], 1):
        print(f"{i}. [{art['relevance_score']}] {art['title']}")
    
    return scored_articles

if __name__ == "__main__":
    # Fetch with default settings
    fetch_market_news(days_back=7, min_score=3, save_all=False)
    