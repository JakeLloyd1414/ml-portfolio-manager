from datetime import datetime

class Trade:
    def __init__(self, symbol: str, action: str, shares: int, price: float, timestamp: datetime, reason: str = ""):
        self.symbol = symbol
        self.action = action
        self.shares = shares
        self.price = price
        self.timestamp = timestamp
        self.reason = reason
        self.value = shares * price