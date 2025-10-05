import cohere
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List
from dotenv import load_dotenv

load_dotenv()

class ScreenedStockList:
    def __init__(self, cache_file: str = "data/stock_cache.json", cache_days: int = 7):
        self.cache_file = cache_file
        self.cache_days = cache_days
        self.banned_stocks = {
            "TSLA", "GME", "AMC", "BABA", "JD", "BIDU", "RIVN", "LCID", 
            "HOOD", "PTON", "NKLA", "SPCE", "BBBY", "WISH", "CLOV"
        }
    
    def get_stocks(self) -> List[str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                
                cache_date = datetime.fromisoformat(cache['generated_date'])
                if datetime.now() - cache_date < timedelta(days=self.cache_days):
                    logging.info(f"Using cached list from {cache_date.strftime('%Y-%m-%d')}")
                    return self._filter_banned(cache['stocks'])
            except Exception as e:
                logging.warning(f"Cache read failed: {e}")
        
        try:
            stocks = self._generate_list()
            
            cache = {
                'stocks': stocks,
                'generated_date': datetime.now().isoformat(),
                'source': 'cohere_api'
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            logging.info(f"Generated {len(stocks)} stocks via Cohere")
            return self._filter_banned(stocks)
            
        except Exception as e:
            logging.error(f"Cohere failed: {e}")
            logging.info("Using fallback list")
            return self._filter_banned(self._get_defaults())

    def _generate_list(self) -> List[str]:
        key = os.getenv('COHERE_API_KEY')
        if not key:
            raise ValueError("COHERE_API_KEY not set")
        
        co = cohere.Client(key)
        
        prompt = """Generate exactly 50 US stock tickers for algo trading. 
        Requirements:
        - Large/mid-cap (>$1B market cap)
        - High volume (>500k shares/day)
        - No earnings in next 5 days
        - Diversified sectors: tech, healthcare, finance, consumer, energy, utilities, industrial, real estate
        - Mix of growth and value
        - Avoid: penny stocks, recent IPOs (<2yrs), meme stocks, crypto stocks
        - Banned: TSLA, GME, AMC, BABA, JD, BIDU, RIVN, LCID, HOOD, PTON, NKLA, SPCE

        Return ONLY ticker symbols separated by commas, no other text.

        Example: AAPL, MSFT, GOOGL, AMZN, META

        Your 50 tickers:"""

        try:
            resp = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=300,
                temperature=0.3,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
            
            text = resp.generations[0].text.strip()
            stocks = [s.strip().upper() for s in text.split(',')]
            stocks = [s for s in stocks if s and len(s) <= 5 and s.isalpha()]
            
            if len(stocks) < 40:
                raise ValueError(f"Only got {len(stocks)} stocks")
            
            stocks = stocks[:50]
            logging.info(f"Got {len(stocks)} from Cohere")
            return stocks
            
        except Exception as e:
            logging.error(f"API call failed: {e}")
            raise
    
    def _get_defaults(self) -> List[str]:
        print("Using default list")
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "UBER",
            "NFLX", "INTC", "AMD", "PYPL", "SHOP", "SQ", "ZOOM", "DOCU", "SNOW", "PLTR",
            "JNJ", "PFE", "UNH", "ABBV", "TMO", "DHR", "ABT", "ISRG", "GILD", "AMGN",
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
            "WMT", "HD", "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "DIS", "COST",
            "XOM", "CVX", "COP", "EOG", "SLB", "NEE", "DUK", "SO", "AEP", "EXC",
            "BA", "CAT", "GE", "MMM", "UPS", "FDX", "LMT", "RTX", "HON", "DE",
            "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "MAA", "ESS"
        ]
    
    def _filter_banned(self, stocks: List[str]) -> List[str]:
        filtered = [s for s in stocks if s not in self.banned_stocks]
        
        if len(filtered) < len(stocks):
            banned = [s for s in stocks if s in self.banned_stocks]
            logging.info(f"Filtered {len(banned)} banned: {', '.join(banned)}")
        
        return filtered