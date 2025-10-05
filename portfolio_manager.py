import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import warnings
import time
import schedule
import threading
import signal
import sys
import select
import os
from src.stock_agent import StockPredictionAgent
from src.position import Position
from src.trade import Trade

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

trade_logger = logging.getLogger('trades')
trade_logger.setLevel(logging.INFO)

os.makedirs('data', exist_ok=True)

trade_handler = logging.FileHandler('data/portfolio_manager.log')
trade_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
trade_logger.addHandler(trade_handler)
trade_logger.propagate = False


class PortfolioManager:
    def __init__(self, initial_capital: float = 10000, max_positions: int = 10, 
                 commission: float = 0.0, max_position_size: float = 0.05):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.commission = commission
        self.max_position_size = 0.10
        self.running = True
        
        self.positions: Dict[str, Position] = {}
        self.prediction_agents: Dict[str, StockPredictionAgent] = {}
        self.trade_history: List[Trade] = []
        self.daily_portfolio_values: List[Dict] = []
        
        # risk params for weekly strategy
        self.stop_loss_amount = 10.00
        self.max_daily_loss_pct = 0.02
        self.min_confidence = 0.75
        self.profit_target = 12.00
        
        self.performance_metrics = {}
        self.model_performance = {}
        
        logging.info(f"Portfolio initialized: ${initial_capital:,.2f}")
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logging.info("\nShutting down...")
        self.running = False
        self.save_state()
        sys.exit(0)
    
    def add_stock_to_universe(self, symbol: str, period: str = '1y'):
        try:
            self.prediction_agents[symbol] = StockPredictionAgent(symbol, period)
            logging.info(f"Added {symbol}")
        except Exception as e:
            logging.error(f"Couldn't add {symbol}: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            return data['Close'].iloc[-1] if not data.empty else None
        except Exception as e:
            logging.error(f"Price fetch failed for {symbol}: {e}")
            return None
    
    def get_portfolio_value(self) -> float:
        total = self.cash
        for pos in self.positions.values():
            price = self.get_current_price(pos.symbol)
            if price:
                pos.update_price(price)
                total += pos.get_market_value()
        return total
    
    def calculate_position_size(self, symbol: str, prediction_prob: float, 
                              current_price: float) -> int:
        try:
            pval = self.get_portfolio_value()
            target = pval * self.max_position_size
            shares = int(target / current_price)
            
            # make sure we have cash
            needed = shares * current_price * (1 + self.commission)
            if needed > self.cash:
                shares = int(self.cash / (current_price * (1 + self.commission)))
            
            # stay under 10%
            actual = shares * current_price
            if actual > target:
                shares = int(target / current_price)
            
            logging.info(f"{symbol} sizing: ${actual:.2f} ({actual/pval:.1%})")
            return max(shares, 0)
            
        except Exception as e:
            logging.error(f"Sizing error for {symbol}: {e}")
            return 0
    
    def get_historical_accuracy(self, symbol: str) -> Optional[float]:
        if symbol in self.model_performance:
            return self.model_performance[symbol].get('accuracy', None)
        return None
    
    def train_all_models(self, force_retrain=False):
        logging.info("=== Training Models ===")
        
        for sym, agent in self.prediction_agents.items():
            try:
                if agent.model is not None and not force_retrain:
                    info = agent.get_model_info()
                    if info:
                        logging.info(f"{sym} - existing (acc: {info.get('accuracy', 0):.4f})")
                        self.model_performance[sym] = {
                            'accuracy': info.get('accuracy', 0),
                            'last_trained': datetime.fromisoformat(info.get('training_date', datetime.now().isoformat())),
                            'samples': info.get('samples', 0)
                        }
                        continue
                
                logging.info(f"Training {sym}")
                data = agent.fetch_data()
                features = agent.create_features(data)
                X, y = agent.prepare_data(features)
                results = agent.train_model(X, y)
                
                self.model_performance[sym] = {
                    'accuracy': results['accuracy'],
                    'last_trained': datetime.now(),
                    'samples': len(X)
                }
                
                logging.info(f"{sym} done - acc: {results['accuracy']:.4f} ({len(X)} samples)")
                
            except Exception as e:
                logging.error(f"{sym} training failed: {e}")
                continue
        
        logging.info("=== Training Complete ===")
    
    def generate_predictions_only(self) -> Dict[str, Dict]:
        predictions = {}
        for sym, agent in self.prediction_agents.items():
            try:
                if agent.model is None:
                    logging.warning(f"{sym} not trained - skipping")
                    continue
                predictions[sym] = agent.predict_next_day()
            except Exception as e:
                logging.error(f"{sym} prediction failed: {e}")
        return predictions
    
    def generate_trading_signals(self, predictions: Dict[str, Dict]) -> List[Dict]:
        signals = []
        for sym, pred in predictions.items():
            if pred['probability'] <= self.min_confidence:
                continue
                
            price = self.get_current_price(sym)
            if not price:
                continue
            
            action = None
            if pred['prediction'] == 1 and sym not in self.positions:
                action = 'BUY'
            elif pred['prediction'] == 0 and sym in self.positions:
                action = 'SELL'
            
            if action:
                signals.append({
                    'symbol': sym,
                    'action': action,
                    'confidence': pred['probability'],
                    'current_price': price,
                    'expected_direction': pred['interpretation']
                })
        
        return sorted(signals, key=lambda x: x['confidence'], reverse=True)
    
    def execute_trade(self, symbol: str, action: str, shares: int, price: float, reason: str = ""):
        try:
            val = shares * price
            comm = val * self.commission
            
            if action == 'BUY':
                cost = val + comm
                if cost > self.cash:
                    logging.warning(f"Not enough cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
                    return False
                
                self.cash -= cost
                
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    total_shares = pos.shares + shares
                    avg = ((pos.shares * pos.entry_price) + (shares * price)) / total_shares
                    pos.shares = total_shares
                    pos.entry_price = avg
                else:
                    self.positions[symbol] = Position(symbol, shares, price, datetime.now())
                
                trade_logger.info(f"BUY {shares} {symbol} @ ${price:.2f} - {reason}")
                
            elif action == 'SELL':
                if symbol not in self.positions:
                    logging.warning(f"Can't sell {symbol}: no position")
                    return False
                
                pos = self.positions[symbol]
                if shares > pos.shares:
                    shares = pos.shares
                
                proceeds = (shares * price) - comm
                self.cash += proceeds
                
                pos.shares -= shares
                if pos.shares == 0:
                    pnl = (price - pos.entry_price) * shares
                    del self.positions[symbol]
                    trade_logger.info(f"CLOSED {symbol}: ${pnl:.2f}")
                
                trade_logger.info(f"SELL {shares} {symbol} @ ${price:.2f} - {reason}")
            
            trade = Trade(symbol, action, shares, price, datetime.now(), reason)
            self.trade_history.append(trade)
            return True
            
        except Exception as e:
            logging.error(f"Trade failed for {symbol}: {e}")
            return False
    
    def apply_stop_losses(self):
        to_close = []
        for sym, pos in self.positions.items():
            price = self.get_current_price(sym)
            if not price:
                continue
            
            pos.update_price(price)
            loss = (pos.entry_price - price) * pos.shares
            
            if loss >= self.stop_loss_amount:
                to_close.append(sym)
                trade_logger.info(f"Stop loss: {sym} -${loss:.2f}")
        
        for sym in to_close:
            pos = self.positions[sym]
            price = self.get_current_price(sym)
            if price:
                self.execute_trade(sym, 'SELL', pos.shares, price, "stop_loss")
    
    def apply_profit_targets(self):
        to_close = []
        for sym, pos in self.positions.items():
            price = self.get_current_price(sym)
            if not price:
                continue
            
            pos.update_price(price)
            profit = (price - pos.entry_price) * pos.shares
            
            if profit >= self.profit_target:
                to_close.append(sym)
                trade_logger.info(f"Profit target: {sym} +${profit:.2f}")
        
        for sym in to_close:
            pos = self.positions[sym]
            price = self.get_current_price(sym)
            if price:
                self.execute_trade(sym, 'SELL', pos.shares, price, "profit_target")
    
    def check_daily_loss_limit(self) -> bool:
        val = self.get_portfolio_value()
        loss = (self.initial_capital - val) / self.initial_capital
        if loss > self.max_daily_loss_pct:
            logging.warning(f"Daily loss limit: {loss:.2%}")
            return True
        return False
    
    def execute_live_trading_startup(self):
        logging.info("=== Live Trading Startup ===")
        
        try:
            self.apply_stop_losses()
            self.apply_profit_targets()
            
            preds = self.generate_predictions_only()
            signals = self.generate_trading_signals(preds)
            logging.info(f"Generated {len(signals)} signals")
            
            executed = 0
            for sig in signals:
                sym = sig['symbol']
                action = sig['action']
                price = sig['current_price']
                
                if action == 'BUY' and len(self.positions) < self.max_positions:
                    shares = self.calculate_position_size(sym, sig['confidence'], price)
                    if shares > 0:
                        if self.execute_trade(sym, action, shares, price, 
                                           f"startup_conf_{sig['confidence']:.3f}"):
                            executed += 1
                
                elif action == 'SELL' and sym in self.positions:
                    pos = self.positions[sym]
                    if self.execute_trade(sym, action, pos.shares, price,
                                       f"startup_conf_{sig['confidence']:.3f}"):
                        executed += 1
            
            logging.info(f"Startup complete - {executed} trades")
            self.log_portfolio_status()
            
        except Exception as e:
            logging.error(f"Startup error: {e}")
    
    def execute_daily_strategy(self):
        logging.info("=== Daily Strategy ===")
        
        try:
            if self.check_daily_loss_limit():
                logging.info("Loss limit hit - skipping trades")
                return
            
            self.apply_stop_losses()
            self.apply_profit_targets()
            
            preds = self.generate_predictions_only()
            signals = self.generate_trading_signals(preds)
            logging.info(f"{len(signals)} signals")
            
            executed = 0
            for sig in signals:
                if executed >= 3:
                    break
                
                sym = sig['symbol']
                action = sig['action']
                price = sig['current_price']
                
                if action == 'BUY' and len(self.positions) < self.max_positions:
                    shares = self.calculate_position_size(sym, sig['confidence'], price)
                    if shares > 0:
                        if self.execute_trade(sym, action, shares, price, 
                                           f"daily_conf_{sig['confidence']:.3f}"):
                            executed += 1
                
                elif action == 'SELL' and sym in self.positions:
                    pos = self.positions[sym]
                    if self.execute_trade(sym, action, pos.shares, price,
                                       f"daily_conf_{sig['confidence']:.3f}"):
                        executed += 1
            
            self.log_portfolio_status()
            
        except Exception as e:
            logging.error(f"Daily strategy error: {e}")
    
    def execute_weekly_strategy(self):
        logging.info("=== Monday Weekly Strategy ===")
        
        try:
            # clear stale positions from state
            if len(self.positions) > 0:
                logging.warning(f"Clearing {len(self.positions)} stale positions from memory")
                self.positions.clear()
                self.save_state()
            
            if self.check_daily_loss_limit():
                logging.info("Loss limit - skipping")
                return
            
            preds = self.generate_predictions_only()
            signals = self.generate_trading_signals(preds)
            logging.info(f"{len(signals)} weekly signals")
            
            executed = 0
            for sig in signals:
                sym = sig['symbol']
                action = sig['action']
                price = sig['current_price']
                
                if action == 'BUY' and len(self.positions) < self.max_positions:
                    shares = self.calculate_position_size(sym, sig['confidence'], price)
                    if shares > 0:
                        if self.execute_trade(sym, action, shares, price, 
                                           f"weekly_conf_{sig['confidence']:.3f}"):
                            executed += 1
            
            logging.info(f"Week started - {executed} positions")
            self.log_portfolio_status()
            
        except Exception as e:
            logging.error(f"Weekly strategy error: {e}")
    
    def log_portfolio_status(self):
        val = self.get_portfolio_value()
        ret = (val - self.initial_capital) / self.initial_capital
        
        logging.info(f"Value: ${val:,.2f}")
        logging.info(f"Cash: ${self.cash:,.2f}")
        logging.info(f"Return: {ret:.2%}")
        logging.info(f"Positions: {len(self.positions)}")
        
        for sym, pos in self.positions.items():
            price = self.get_current_price(sym)
            if price:
                pos.update_price(price)
                logging.info(f"  {sym}: {pos.shares} @ ${pos.current_price:.2f} "
                           f"(entry ${pos.entry_price:.2f}, pnl ${pos.unrealized_pnl:.2f})")
        
        self.daily_portfolio_values.append({
            'date': datetime.now(),
            'portfolio_value': val,
            'cash': self.cash,
            'total_return': ret,
            'num_positions': len(self.positions)
        })
        
        if len(self.daily_portfolio_values) > 100:
            self.daily_portfolio_values = self.daily_portfolio_values[-100:]
    
    def calculate_performance_metrics(self) -> Dict:
        if len(self.daily_portfolio_values) < 2:
            return {}
        
        df = pd.DataFrame(self.daily_portfolio_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        total_ret = (df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        daily_rets = df['daily_return'].dropna()
        if len(daily_rets) > 0:
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
            max_dd = (df['portfolio_value'] / df['portfolio_value'].cummax() - 1).min()
        else:
            sharpe = 0
            max_dd = 0
        
        wins = [t for t in self.trade_history if t.action == 'SELL' and self._calculate_trade_pnl(t) > 0]
        sells = [t for t in self.trade_history if t.action == 'SELL']
        win_rate = len(wins) / len(sells) if len(sells) > 0 else 0
        
        return {
            'total_return': total_ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'current_positions': len(self.positions),
            'days_active': len(df)
        }
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        if trade.action.lower() == 'buy':
            return -trade.value  # Cash outflow
        elif trade.action.lower() == 'sell':
            return trade.value   # Cash inflow
        else:
            return 0.0
    
    def retrain_models_weekly(self):
        logging.info("Weekly retraining...")
        
        for sym, agent in self.prediction_agents.items():
            try:
                data = agent.fetch_data()
                features = agent.create_features(data)
                X, y = agent.prepare_data(features)
                results = agent.train_model(X, y)
                
                self.model_performance[sym] = {
                    'accuracy': results['accuracy'],
                    'last_trained': datetime.now(),
                    'samples': len(X)
                }
                
                logging.info(f"{sym} retrained - acc: {results['accuracy']:.4f}")
                
            except Exception as e:
                logging.error(f"{sym} retrain failed: {e}")
    
    def save_state(self, filename: str = "data/portfolio_state.json"):
        limited = self.daily_portfolio_values[-100:] if len(self.daily_portfolio_values) > 100 else self.daily_portfolio_values
        
        state = {
            'cash': self.cash,
            'initial_capital': self.initial_capital,
            'positions': {sym: {
                'shares': pos.shares,
                'entry_price': pos.entry_price,
                'entry_date': pos.entry_date.isoformat()
            } for sym, pos in self.positions.items()},
            'performance_metrics': self.performance_metrics,
            'daily_portfolio_values': [{
                'date': rec['date'].isoformat() if isinstance(rec['date'], datetime) else rec['date'],
                'portfolio_value': rec['portfolio_value'],
                'cash': rec['cash'],
                'total_return': rec['total_return'],
                'num_positions': rec['num_positions']
            } for rec in limited]
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logging.info(f"State saved ({len(limited)} entries)")
    
    def load_state(self, filename: str = "data/portfolio_state.json"):
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.cash = state['cash']
            self.initial_capital = state['initial_capital']
            
            for sym, data in state['positions'].items():
                self.positions[sym] = Position(
                    sym, 
                    data['shares'],
                    data['entry_price'],
                    datetime.fromisoformat(data['entry_date'])
                )
            
            self.daily_portfolio_values = []
            for rec in state['daily_portfolio_values']:
                rec['date'] = datetime.fromisoformat(rec['date'])
                self.daily_portfolio_values.append(rec)
            
            logging.info(f"State loaded from {filename}")
            
        except FileNotFoundError:
            logging.info("No state file found - starting fresh")
        except Exception as e:
            logging.error(f"Load error: {e}")

    def monitor_positions_continuously(self):
        while self.running:
            try:
                now = datetime.now()
                if self.is_market_hours(now):
                    if self.is_friday_close_time(now):
                        self.close_all_positions_friday()
                    elif len(self.positions) > 0:
                        logging.info(f"Monitoring {len(self.positions)} positions")
                        self.apply_stop_losses()
                        self.apply_profit_targets()
                        self.log_portfolio_status()
                
                for _ in range(60):
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logging.error(f"Monitor error: {e}")
                time.sleep(60)
    
    def is_market_hours(self, current_time: datetime) -> bool:
        if current_time.weekday() >= 5:
            return False
        
        open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= current_time <= close_time
    
    def is_friday_close_time(self, current_time: datetime) -> bool:
        if current_time.weekday() != 4:
            return False
        close = current_time.replace(hour=15, minute=45, second=0, microsecond=0)
        return current_time >= close
    
    def close_all_positions_friday(self):
        if not self.positions:
            return
        
        logging.info("Friday 3:45 PM - closing all positions")
        
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            price = self.get_current_price(sym)
            if price:
                pnl = (price - pos.entry_price) * pos.shares
                trade_logger.info(f"Closing {sym}: ${pnl:.2f}")
                self.execute_trade(sym, 'SELL', pos.shares, price, "friday_close")
    
    def run_live_monitoring(self):
        logging.info("=== Live Monitoring (No Buying) ===")
        
        self.apply_stop_losses()
        self.apply_profit_targets()
        
        schedule.every().friday.at("15:45").do(self.close_all_positions_friday)
        
        monitor_thread = threading.Thread(target=self.monitor_positions_continuously, daemon=True)
        monitor_thread.start()
        
        logging.info("Monitoring active...")
        logging.info("- NO NEW BUYS")
        logging.info("- Friday close: 3:45 PM")
        logging.info("- Profit: $10/position | Stop: $8")
        logging.info("- Type 's' + Enter to sell")
        logging.info("- Ctrl+C to stop")
        
        while self.running:
            try:
                schedule.run_pending()
                
                for _ in range(60):
                    if not self.running:
                        break
                    
                    try:
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            user_in = input().strip().lower()
                            if user_in == 's':
                                self.show_manual_sell_menu()
                                logging.info("Resumed monitoring")
                    except (ImportError, OSError):
                        pass
                    
                    time.sleep(1)
                    
            except Exception as e:
                logging.error(f"Monitor loop error: {e}")
                time.sleep(60)
        
        logging.info("Monitoring stopped")

    def run_live_trading(self):
        logging.info("=== Live Trading ===")
        
        self.execute_live_trading_startup()
        
        schedule.every().monday.at("10:00").do(self.execute_weekly_strategy)
        schedule.every().friday.at("15:45").do(self.close_all_positions_friday)
        
        monitor_thread = threading.Thread(target=self.monitor_positions_continuously, daemon=True)
        monitor_thread.start()
        
        logging.info("Weekly trading active...")
        logging.info("- Monday 10am: new trades")
        logging.info("- Friday 3:45pm: close all")
        logging.info("- Monitor: every minute")
        logging.info("- Profit: $10 | Stop: $8")
        
        while self.running:
            try:
                schedule.run_pending()
                for _ in range(60):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                time.sleep(60)
        
        logging.info("Trading stopped")
    
    def manual_sell_position(self, symbol: str) -> bool:
        symbol = symbol.upper()
        
        if symbol not in self.positions:
            logging.warning(f"No position in {symbol}")
            return False
        
        pos = self.positions[symbol]
        price = self.get_current_price(symbol)
        
        if not price:
            logging.error(f"Couldn't get price for {symbol}")
            return False
        
        pnl = (price - pos.entry_price) * pos.shares
        success = self.execute_trade(symbol, 'SELL', pos.shares, price, "manual_sell")
        
        if success:
            logging.info(f"Manual sell: {symbol} - ${pnl:.2f}")
        
        return success
    
    def list_current_positions(self) -> List[str]:
        return list(self.positions.keys())
    
    def show_manual_sell_menu(self):
        if not self.positions:
            print("\nNo positions")
            return
        
        print("\n" + "="*40)
        print("MANUAL SELL")
        print("="*40)
        
        pos_data = []
        for i, (sym, pos) in enumerate(self.positions.items(), 1):
            price = self.get_current_price(sym)
            if price:
                pos.update_price(price)
                sign = "+" if pos.unrealized_pnl >= 0 else ""
                pos_data.append({
                    'num': i,
                    'symbol': sym,
                    'shares': pos.shares,
                    'entry_price': pos.entry_price,
                    'current_price': price,
                    'pnl': pos.unrealized_pnl,
                    'pnl_sign': sign
                })
                
                print(f"{i}. {sym}: {pos.shares} @ ${price:.2f} "
                      f"(entry ${pos.entry_price:.2f}, {sign}${pos.unrealized_pnl:.2f})")
        
        print(f"{len(pos_data) + 1}. Cancel")
        print("-"*40)
        
        try:
            choice = input("Select (number): ").strip()
            num = int(choice)
            
            if num == len(pos_data) + 1:
                print("Cancelled")
                return
            
            if 1 <= num <= len(pos_data):
                sel = pos_data[num - 1]
                sym = sel['symbol']
                pnl = sel['pnl']
                
                if input(f"\nSell {sym} for ${pnl:.2f}? (y/n): ").lower() == 'y':
                    if self.manual_sell_position(sym):
                        print(f"{sym} sold!")
                    else:
                        print("Sell failed")
                else:
                    print("Cancelled")
            else:
                print("Invalid selection")
                
        except ValueError:
            print("Invalid input")
        except Exception as e:
            print(f"Error: {e}")