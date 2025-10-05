# ML Portfolio Manager

Weekly automated trading system using XGBoost models for stock prediction.

## Setup

1. Install dependencies:
   pip install -r requirements.txt --break-system-packages

2. Create a `.env` file with your Cohere API key:
   COHERE_API_KEY=your_key_here

3. Run it:
   python main.py

## How it works

- Trains XGBoost models on 50 stocks (or uses cached models)
- Trades Monday mornings, closes everything Friday afternoon
- 10% position sizing, $12 profit target, $10 stop loss
- Models saved in `models/`, state saved in `data/`

## Quick commands

```bash
# List positions
python main.py --list

# Sell a position
python main.py --sell=AAPL
```

## Strategy

- Weekly hold period (Mon-Fri)
- Target 2%+ weekly returns
- Max 10 positions
- Auto-close all positions Friday 3:45 PM

## Files

- `main.py` - Entry point and CLI
- `portfolio_manager.py` - Trading logic
- `stock_agent.py` - ML model per stock
- `stock_screening.py` - Stock universe (Cohere API)
- `ui.py` - Menu display

## Notes

Models are cached for 7 days. Weekend runs skip stock loading and use existing models only.
