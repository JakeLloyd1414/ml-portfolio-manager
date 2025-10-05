from datetime import datetime

class Position:
    def __init__(self, symbol: str, shares: int, entry_price: float, entry_date: datetime):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        
    def update_price(self, current_price: float):
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.shares
        
    def get_market_value(self) -> float:
        return self.current_price * self.shares
        
    def get_return_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price