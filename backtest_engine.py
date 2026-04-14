# backtest_engine.py - Backtesting Engine (FIXED)
# =========================================================
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)

class CryptoBacktester:
    """
    Motor de backtesting robusto para evaluar estrategias LSTM.
    👈 Ahora maneja automáticamente la cantidad de datos disponible.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        position_size: float = 1.0,
        transaction_fee: float = 0.001,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        min_confidence: float = 0.0
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_fee = transaction_fee
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
        self.reset()
    
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.dates: List[pd.Timestamp] = []
    
    def fetch_historical_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Descarga datos históricos con manejo robusto de períodos."""
        try:
            # 👈 ESTRATEGIA: Si es intervalo diario, usar período amplio para asegurar datos
            if interval == "1d":
                df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False)
            else:
                # Para intervalos intradía, yfinance tiene límites; usar period en su lugar
                days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                period_map = {
                    (0, 7): "5d", (7, 30): "1mo", (30, 90): "3mo", 
                    (90, 180): "6mo", (180, 365): "1y", (365, 730): "2y"
                }
                period = "2y"  # Default amplio
                for (min_d, max_d), p in period_map.items():
                    if min_d <= days < max_d:
                        period = p
                        break
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            
            if df.empty:
                return pd.DataFrame()
            
            # Manejar MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [c.lower().strip() for c in df.columns]
            required = ["open", "high", "low", "close", "volume"]
            
            if not all(col in df.columns for col in required):
                return pd.DataFrame()
            
            df = df[required].copy()
            df.rename(columns={
                "open": "Open", "high": "High", "low": "Low", 
                "close": "Close", "volume": "Volume"
            }, inplace=True)
            
            # 👈 FILTRAR por fechas exactas (por si yfinance devolvió más datos)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
        except Exception as e:
            logging.error(f"❌ Error descargando {ticker}: {e}")
            return pd.DataFrame()
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores con manejo seguro de NaN."""
        if df.empty:
            return df
        df = df.copy()
        
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df["returns"] = df["Close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        df["volume_ma"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_ma"]
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return df
    
    def generate_signals(self, predictions: np.ndarray, actual_prices: np.ndarray, threshold_pct: float = 1.5) -> np.ndarray:
        signals = np.zeros(len(predictions))
        for i in range(len(predictions)):
            if np.isnan(predictions[i]) or np.isnan(actual_prices[i]):
                continue
            change_pct = (predictions[i] / actual_prices[i] - 1) * 100
            if change_pct >= threshold_pct:
                signals[i] = 1
            elif change_pct <= -threshold_pct:
                signals[i] = -1
        return signals
    
    def execute_trade(self, signal: int, price: float, date: pd.Timestamp, confidence: float = 0.0) -> Optional[Dict]:
        if signal == 0 or abs(confidence) < self.min_confidence:
            return None
        
        trade_capital = self.capital * self.position_size
        quantity = trade_capital / price
        fee = trade_capital * self.transaction_fee
        net_invested = trade_capital - fee
        
        trade = {
            'date': date, 'type': 'BUY' if signal > 0 else 'SELL', 'signal': signal,
            'entry_price': price, 'quantity': quantity, 'invested': net_invested,
            'fee': fee, 'capital_before': self.capital, 'confidence': confidence,
            'exit_price': None, 'exit_date': None, 'pnl': 0, 'pnl_pct': 0, 'exit_reason': 'open'
        }
        
        self.position = signal
        self.entry_price = price
        self.capital -= net_invested
        return trade
    
    def close_trade(self, trade: Dict, exit_price: float, exit_date: pd.Timestamp, reason: str = 'signal') -> Dict:
        quantity = trade['quantity']
        gross_value = quantity * exit_price
        exit_fee = gross_value * self.transaction_fee
        net_value = gross_value - exit_fee
        pnl = net_value - trade['invested']
        pnl_pct = (pnl / trade['invested']) * 100 if trade['invested'] > 0 else 0
        
        trade.update({
            'exit_price': exit_price, 'exit_date': exit_date, 'pnl': pnl, 'pnl_pct': pnl_pct,
            'exit_fee': exit_fee, 'exit_reason': reason, 'gross_value': gross_value, 'net_value': net_value
        })
        
        self.capital += net_value
        self.position = 0
        self.entry_price = 0
        return trade
    
    def check_stop_loss_take_profit(self, current_price: float, entry_price: float, signal: int) -> Optional[str]:
        if signal == 0 or entry_price == 0:
            return None
        change_pct = (current_price / entry_price - 1) * 100
        
        if signal > 0:  # Long
            if self.stop_loss_pct and change_pct <= -self.stop_loss_pct:
                return 'stop_loss'
            if self.take_profit_pct and change_pct >= self.take_profit_pct:
                return 'take_profit'
        else:  # Short
            if self.stop_loss_pct and change_pct >= self.stop_loss_pct:
                return 'stop_loss'
            if self.take_profit_pct and change_pct <= -self.take_profit_pct:
                return 'take_profit'
        return None
    
    def run_backtest(
        self,
        ticker: str,
        model,
        scaler,
        feature_cols: List[str],
        close_idx: int,
        seq_len: int,
        start_date: str,
        end_date: str,
        threshold_pct: float = 1.5,
        interval: str = "1d"
    ) -> Dict:
        """
        Ejecuta backtesting con manejo adaptativo de datos.
        👈 Ahora ajusta automáticamente los requisitos según datos disponibles.
        """
        from tensorflow.keras.models import load_model
        
        self.reset()
        
        # 👈 ESTRATEGIA: Descargar MÁS datos de los solicitados para tener buffer post-indicadores
        buffer_days = 100 if interval == "1d" else 30
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=buffer_days)
        
        logging.info(f"📥 Descargando datos para {ticker} (con buffer: {start_dt.date()} a {end_date})...")
        df = self.fetch_historical_data(ticker, start_dt.strftime("%Y-%m-%d"), end_date, interval)
        
        if df.empty:
            return {'error': f'No se pudieron descargar datos para {ticker}'}
        
        # Añadir indicadores
        df = self.add_indicators(df)
        
        # 👈 VALIDACIÓN ADAPTATIVA: Calcular mínimo real necesario
        min_required = seq_len + 10  # 👈 Reducido de 50 a 10 para ser más flexible
        
        if len(df) < min_required:
            # 👈 FALLBACK: Intentar con datos diarios si falló con horarios
            if interval != "1d":
                logging.warning(f"⚠️ Pocos datos con {interval}, intentando con 1d...")
                df = self.fetch_historical_data(ticker, start_dt.strftime("%Y-%m-%d"), end_date, "1d")
                df = self.add_indicators(df)
                interval = "1d"  # Actualizar para consistencia
            
            if len(df) < min_required:
                return {
                    'error': f'Datos insuficientes: {len(df)} filas (mínimo: {min_required}). '
                            f'Prueba ampliando el período o usando interval="1d".'
                }
        
        # Escalar datos
        scaled_data = scaler.transform(df[feature_cols])
        
        # Generar predicciones walk-forward
        predictions, actual_prices, dates_list = [], [], []
        
        logging.info(f"🔮 Generando {len(scaled_data) - seq_len} predicciones...")
        
        for i in range(seq_len, len(scaled_data)):
            X_seq = scaled_data[i-seq_len:i].reshape(1, seq_len, len(feature_cols))
            pred_scaled = model.predict(X_seq, verbose=0)[0, 0]
            
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, close_idx] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, close_idx]
            
            predictions.append(pred_price)
            actual_prices.append(df['Close'].iloc[i])
            dates_list.append(df.index[i])
        
        if len(predictions) == 0:
            return {'error': 'No se pudieron generar predicciones válidas'}
        
        predictions = np.array(predictions)
        actual_prices = np.array(actual_prices)
        dates = pd.DatetimeIndex(dates_list)
        
        # Generar señales
        raw_signals = self.generate_signals(predictions, actual_prices, threshold_pct)
        confidences = np.abs((predictions / actual_prices - 1) * 100)
        
        # Ejecutar simulación
        logging.info("💼 Ejecutando simulación de trading...")
        open_trade = None
        
        for idx in range(len(raw_signals)):
            date = dates[idx]
            price = actual_prices[idx]
            signal = int(raw_signals[idx])
            confidence = confidences[idx]
            
            # Registrar equity
            if open_trade:
                current_value = open_trade['quantity'] * price
                current_equity = self.capital + current_value
            else:
                current_equity = self.capital
            
            self.equity_curve.append(current_equity)
            self.dates.append(date)
            
            # Gestionar posición abierta
            if open_trade:
                exit_reason = self.check_stop_loss_take_profit(price, open_trade['entry_price'], open_trade['signal'])
                if exit_reason:
                    open_trade = self.close_trade(open_trade, price, date, reason=exit_reason)
                    self.trades.append(open_trade)
                    open_trade = None
                    continue
            
            # Nueva señal
            if open_trade is None and signal != 0:
                open_trade = self.execute_trade(signal, price, date, confidence)
                if open_trade:
                    self.trades.append(open_trade)
        
        # Cerrar posición final
        if open_trade:
            open_trade = self.close_trade(open_trade, actual_prices[-1], dates[-1], reason='end_of_period')
            self.trades.append(open_trade)
        
        return self._calculate_metrics(ticker, df, predictions, actual_prices, dates, raw_signals)
    
    def _calculate_metrics(self, ticker: str, df: pd.DataFrame, predictions: np.ndarray, 
                          actual_prices: np.ndarray, dates: pd.DatetimeIndex, signals: np.ndarray) -> Dict:
        final_capital = self.capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        equity_df = pd.DataFrame({'date': self.dates, 'equity': self.equity_curve})
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        if len(equity_df) > 1:
            equity_df['daily_return'] = equity_df['equity'].pct_change()
            sharpe = (equity_df['daily_return'].mean() / equity_df['daily_return'].std()) * np.sqrt(252) if equity_df['daily_return'].std() > 0 else 0
        else:
            sharpe = 0
        
        closed_trades = [t for t in self.trades if t['exit_price'] is not None]
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in closed_trades if t['pnl'] < 0]
        
        win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        profit_factor = (abs(sum(t['pnl'] for t in winning_trades)) / abs(sum(t['pnl'] for t in losing_trades))) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else 999
        
        # Accuracy de dirección
        actual_direction = np.sign(np.diff(actual_prices))
        predicted_direction = np.sign(np.diff(predictions))
        min_len = min(len(actual_direction), len(predicted_direction))
        direction_accuracy = (np.sum(actual_direction[:min_len] == predicted_direction[:min_len]) / min_len * 100) if min_len > 0 else 0
        
        # Correlación
        valid_mask = ~np.isnan(predictions) & ~np.isnan(actual_prices)
        correlation = np.corrcoef(predictions[valid_mask], actual_prices[valid_mask])[0, 1] if np.sum(valid_mask) > 2 else np.nan
        
        return {
            'ticker': ticker, 'period': f"{dates[0].date()} a {dates[-1].date()}",
            'initial_capital': self.initial_capital, 'final_capital': final_capital,
            'total_return_pct': total_return, 'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe, 'win_rate_pct': win_rate,
            'avg_win_pct': avg_win, 'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor, 'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades), 'losing_trades': len(losing_trades),
            'direction_accuracy_pct': direction_accuracy, 'prediction_correlation': correlation,
            'equity_curve': equity_df, 'trades': closed_trades,
            'predictions': predictions, 'actual_prices': actual_prices, 'dates': dates, 'signals': signals
        }