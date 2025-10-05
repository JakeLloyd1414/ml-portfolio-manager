import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockPredictionAgent:
    def __init__(self, symbol, period):
        self.symbol = symbol
        self.period = period
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = f"models/{symbol}_model.pkl"
        self.scaler_path = f"models/{symbol}_scaler.pkl"
        self.metadata_path = f"models/{symbol}_metadata.json"
        
        self.load_model()

    def fetch_data(self):
        ticker = yf.Ticker(self.symbol)
        return ticker.history(period=self.period)
    
    def create_features(self, data):
        df = data.copy()
        
        df['price_change'] = df['Close'] - df['Open']
        df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open']
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Open']
        df['open_prev_close_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_5']
        
        for period in [10, 20, 50, 100]:
            df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}']
            
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=20).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=24).mean()
        exp2 = df['Close'].ewm(span=52).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=18).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df['volatility'] = df['Close'].rolling(window=10).std()
        
        for lag in [1, 5, 10]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # weekly target
        df['monday_open'] = df['Open'].shift(-4)
        df['friday_close'] = df['Close']
        df['weekly_return'] = (df['friday_close'] - df['monday_open']) / df['monday_open']
        df['target'] = (df['weekly_return'] > 0.02).astype(int)
        df['daily_target'] = (df['Close'] > df['Open']).astype(int)
        
        return df
    
    def save_model(self, accuracy: float, samples: int):
        try:
            os.makedirs("models", exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            metadata = {
                'symbol': self.symbol,
                'training_date': datetime.now().isoformat(),
                'accuracy': accuracy,
                'samples': samples,
                'feature_columns': self.feature_columns,
                'period': self.period,
                'model_type': 'XGBoost'
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved {self.symbol} model (acc: {accuracy:.4f})")
            return True
            
        except Exception as e:
            print(f"Save failed for {self.symbol}: {e}")
            return False
    
    def load_model(self):
        try:
            if not all(os.path.exists(p) for p in [self.model_path, self.scaler_path, self.metadata_path]):
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata.get('feature_columns', [])
            
            print(f"Loaded {self.symbol} (trained: {metadata.get('training_date', 'Unknown')}, acc: {metadata.get('accuracy', 0):.4f})")
            return True
            
        except Exception as e:
            print(f"Couldn't load {self.symbol}: {e}")
            return False
    
    def get_model_info(self):
        if not os.path.exists(self.metadata_path):
            return None
        
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Metadata read error for {self.symbol}: {e}")
            return None
    
    def prepare_data(self, df):
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 
                   'target', 'price_change', 'monday_open', 'friday_close', 'weekly_return', 'daily_target']
        features = [col for col in df.columns if col not in exclude]
        
        clean = df[features + ['target']].dropna()
        X = clean[features]
        y = clean['target']
        
        self.feature_columns = features
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        tscv = TimeSeriesSplit(n_splits=5)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, stratify=None
        )
        
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train_sc, y_train, eval_set=[(X_test_sc, y_test)], verbose=False)
        
        preds = self.model.predict(X_test_sc)
        proba = self.model.predict_proba(X_test_sc)[:, 1]
        acc = accuracy_score(y_test, preds)
        
        self.save_model(acc, len(X))
        
        return {
            'accuracy': acc,
            'y_test': y_test,
            'y_pred': preds,
            'y_pred_proba': proba,
            'X_test': X_test,
            'X_train': X_train
        }
    
    def get_feature_importance(self):
        if self.model is None:
            return None
            
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def predict_next_day(self, latest_data=None):
        if self.model is None:
            raise ValueError("Model not trained")
        
        if latest_data is None:
            latest_data = self.fetch_data()
        
        features = self.create_features(latest_data)
        latest = features[self.feature_columns].iloc[-1:]
        
        if latest.isnull().any().any():
            latest = latest.ffill().fillna(0)
        
        scaled = self.scaler.transform(latest)
        pred = self.model.predict(scaled)[0]
        prob = self.model.predict_proba(scaled)[0, 1]
        
        return {
            'prediction': pred,
            'probability': prob,
            'interpretation': 'Close > Open' if pred == 1 else 'Close <= Open'
        }
    
    def backtest(self, df, start_date=None, end_date=None):
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        features = self.create_features(df)
        X, y = self.prepare_data(features)
        
        predictions = []
        returns = []
        window = 252
        
        for i in range(window, len(X)):
            X_train = X.iloc[i-window:i]
            y_train = y.iloc[i-window:i]
            
            scaler_temp = StandardScaler()
            X_train_sc = scaler_temp.fit_transform(X_train)
            
            temp_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            temp_model.fit(X_train_sc, y_train)
            
            X_pred = scaler_temp.transform(X.iloc[i:i+1])
            prob = temp_model.predict_proba(X_pred)[0, 1]
            
            predictions.append(prob)
            
            actual = (features['Close'].iloc[i] - features['Open'].iloc[i]) / features['Open'].iloc[i]
            strategy = actual if prob > 0.5 else -actual
            returns.append(strategy)
        
        return {
            'total_return': np.sum(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'avg_return': np.mean(returns)
        }