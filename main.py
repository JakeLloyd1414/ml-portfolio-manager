import sys
import os
from datetime import datetime
from src.portfolio_manager import PortfolioManager
from src.stock_screening import ScreenedStockList
from src.ui import display_menu

def main():
    os.makedirs("data", exist_ok=True)
    
    portfolio = PortfolioManager(initial_capital=10000, max_positions=10)
    
    # skip loading stocks on weekends
    if datetime.now().weekday() < 5:
        print("Loading stocks")
        try:
            screener = ScreenedStockList()
            for stock in screener.get_stocks():
                portfolio.add_stock_to_universe(stock)
        except Exception as e:
            print(f"Couldn't load stocks: {e}")
            print("Using existing models instead")
    else:
        print("Weekend. Loading saved models only")
        if os.path.exists("models"):
            models = [f.replace("_model.pkl", "") for f in os.listdir("models") 
                     if f.endswith("_model.pkl")]
            print(f"Found {len(models)} models")
            for sym in models:
                portfolio.add_stock_to_universe(sym)
    
    portfolio.load_state()
    
    # handle cli args
    if len(sys.argv) > 1:
        if sys.argv[1].startswith('--sell='):
            sym = sys.argv[1].split('=')[1].upper()
            print(f"\nSelling {sym}")
            
            if sym not in portfolio.positions:
                print(f"No position in {sym}")
                if portfolio.positions:
                    print("Holdings:", list(portfolio.positions.keys()))
                return
            
            pos = portfolio.positions[sym]
            price = portfolio.get_current_price(sym)
            if not price:
                print("Couldn't get price")
                return
                
            pnl = (price - pos.entry_price) * pos.shares
            print(f"{pos.shares} shares @ ${price:.2f} | P&L: ${pnl:.2f}")
            
            if input("Confirm? (y/n): ").lower() == 'y':
                if portfolio.manual_sell_position(sym):
                    print("Sold!")
                    portfolio.save_state()
            return
        
        elif sys.argv[1] == '--list':
            print("\nPositions:")
            if not portfolio.positions:
                print("  None")
                return
                
            for sym, pos in portfolio.positions.items():
                price = portfolio.get_current_price(sym)
                if price:
                    pnl = (price - pos.entry_price) * pos.shares
                    sign = "+" if pnl >= 0 else ""
                    print(f"  {sym}: {pos.shares} @ ${price:.2f} ({sign}${pnl:.2f})")
            return
    
    # main loop
    while True:
        display_menu()
        choice = input("Choice (1-7): ").strip()
        
        try:
            if choice == '1':
                print("\nTraining models")
                portfolio.train_all_models(force_retrain=False)
                portfolio.save_state()
                input("\nPress Enter...")
            
            elif choice == '2':
                print("\nForce retraining")
                portfolio.train_all_models(force_retrain=True)
                portfolio.save_state()
                input("\nPress Enter")
            
            elif choice == '3':
                print("\n=== LIVE TRADING ===")
                print("• Analyze stocks now")
                print("• Buy high-confidence signals")
                print("• Hold Mon-Fri, close Friday 3:45pm")
                print("• Exit at +$12 profit or -$10 loss")
                
                if input("\nStart? (y/n): ").lower() == 'y':
                    print("Starting (Ctrl+C to stop)")
                    portfolio.run_live_trading()

            elif choice == '4':
                print("\n=== MONITORING MODE ===")
                print("• Watch existing positions only")
                print("• No new buys")
                print("• Exit at +$12 profit or -$10 loss")
                
                if input("\nStart? (y/n): ").lower() == 'y':
                    print("Monitoring (Ctrl+C to stop)")
                    portfolio.run_live_monitoring()

            elif choice == '5':
                portfolio.show_manual_sell_menu()
                input("\nPress Enter...")
            
            elif choice == '6':
                print("\n" + "="*40)
                print("PORTFOLIO")
                print("="*40)
                
                val = portfolio.get_portfolio_value()
                ret = (val - portfolio.initial_capital) / portfolio.initial_capital
                
                print(f"Value: ${val:,.2f}")
                print(f"Cash: ${portfolio.cash:,.2f}")
                print(f"Return: {ret:.2%}")
                print(f"Positions: {len(portfolio.positions)}")
                
                if portfolio.positions:
                    print(f"\nHoldings:")
                    for sym, pos in portfolio.positions.items():
                        price = portfolio.get_current_price(sym)
                        if price:
                            pos.update_price(price)
                            sign = "+" if pos.unrealized_pnl >= 0 else ""
                            print(f"  {sym}: {pos.shares} @ ${pos.current_price:.2f} "
                                  f"({sign}${pos.unrealized_pnl:.2f})")
                
                metrics = portfolio.calculate_performance_metrics()
                if metrics:
                    print(f"\nMetrics:")
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            fmt = f"{v:.2%}" if 'return' in k or 'drawdown' in k else f"{v:.4f}"
                            print(f"  {k}: {fmt}")
                        else:
                            print(f"  {k}: {v}")
                
                if portfolio.positions:
                    if input("\nSell anything? (y/n): ").lower() == 'y':
                        portfolio.show_manual_sell_menu()
                
                input("\nPress Enter...")
            
            elif choice == '7':
                portfolio.save_state()
                break
            else:
                print("Invalid choice")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            portfolio.save_state()
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("Press Enter")

if __name__ == "__main__":
    main()