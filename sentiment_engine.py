import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
import re
import logging

logger = logging.getLogger('SniperBot')

class SentimentEngine:
    def __init__(self):
        self.sentiment_score = 0.0  # -1.0 (Bearish) to 1.0 (Bullish)
        self.last_update = 0
        self.news_headlines = []
        self.lock = threading.Lock()
        
        # Sources: Efficient RSS feeds (No API key required)
        self.rss_sources = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cryptopotato.com/feed/"
        ]
        
        # Rigorous Financial Lexicon (Weighted)
        self.lexicon = {
            # BULLISH (+1 to +3)
            'surge': 2, 'record': 2, 'high': 1, 'bull': 2, 'bullish': 2,
            'adoption': 2, 'approved': 3, 'etf': 2, 'launch': 1, 'partnership': 2,
            'gain': 1, 'rally': 2, 'soar': 2, 'breakout': 2, 'upgrade': 1,
            'accumulate': 1, 'buy': 1, 'support': 1, 'long': 1, 'growth': 1,
            'regulatory approval': 3, 'institutional': 2, 'stimulus': 2,
            
            # BEARISH (-1 to -3)
            'crash': -3, 'slump': -2, 'bear': -2, 'bearish': -2, 'plunge': -2,
            'ban': -3, 'regulation': -1, 'lawsuit': -2, 'hack': -3, 'scam': -2,
            'sec': -1, 'sell-off': -2, 'drop': -1, 'low': -1, 'resistance': -1,
            'liquidated': -1, 'down': -1, 'collapse': -3, 'fail': -2, 'risk': -1,
            'inflation': -1, 'recession': -2, 'warning': -1, 'investigation': -2
        }

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)

    def start(self):
        """Starts the background sentiment analysis thread."""
        logger.info("🧠 Sentiment Engine: Starting background thread...")
        self.thread.start()

    def get_sentiment(self):
        """Thread-safe getter for sentiment score."""
        with self.lock:
            return self.sentiment_score, self.news_headlines[:5]

    def _update_loop(self):
        """Background loop to fetch and analyze news every 15 minutes."""
        while self.running:
            try:
                self._analyze_feeds()
                # Sleep for 15 minutes (900 seconds)
                for _ in range(900):
                    if not self.running: break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Sentiment Engine Error: {e}")
                time.sleep(60)

    def _analyze_feeds(self):
        total_score = 0
        article_count = 0
        headlines = []
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

        for url in self.rss_sources:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    xml_data = response.read()
                    root = ET.fromstring(xml_data)
                    
                    # Parse standard RSS items
                    for item in root.findall('.//item')[:10]: # Analyze top 10 per feed
                        title = item.find('title').text
                        if title:
                            score = self._score_text(title)
                            total_score += score
                            article_count += 1
                            headlines.append({'title': title, 'score': score})
            except Exception as e:
                logger.warning(f"Error fetching RSS {url}: {e}")

        # Normalize score
        # Assumes extreme market sentiment ~ +/- 20 total points from ~30 headlines represents strong bias
        if article_count > 0:
            raw_avg = total_score / article_count # Average score per headline
            # Scale: A raw average of 0.5 is very high. Map 0.5 -> 1.0 (Clamp at 1.0)
            normalized = max(min(raw_avg * 2.0, 1.0), -1.0)
        else:
            normalized = 0.0

        with self.lock:
            self.sentiment_score = normalized
            self.news_headlines = headlines
            self.last_update = time.time()
            
        logger.info(f"📊 Market Sentiment Updated: {self.sentiment_score:.2f} based on {article_count} articles.")

    def _score_text(self, text):
        """Calculates a score for a text based on the weighted lexicon."""
        text = text.lower()
        score = 0
        words = re.findall(r'\w+', text)
        
        for word in words:
            if word in self.lexicon:
                score += self.lexicon[word]
                
        return score

# Singleton instance for easy import
sentiment_engine = SentimentEngine()
