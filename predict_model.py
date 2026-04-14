# predict_model.py - Inference and signals with yfinance (CORRECTED)
# =========================================================
import numpy as np
import pandas as pd
import time
import logging
import pickle
import os
import glob
import yfinance as yf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Configuración
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODEL_DIR = "models"
SEQ_LEN = 180
THRESHOLD_PCT = 0.015  # 👈 1.5% umbral para señales

TICKERS = [
    "BTC-USD", "ETH-USD", "TRX-USD", "LINK-USD", "DOGE-USD",
    "SHIB-USD", "XRP-USD", "LTC-USD", "SOL-USD", "ADA-USD", "DOT-USD"
]

FEATURE_COLS = [
    "Close", "RSI", "MACD", "MACD_signal", "Volume",
    "MA5", "MA10", "MA20", "EMA50", "EMA100",
    "returns", "volatility", "volume_ratio"
]
CLOSE_IDX = FEATURE_COLS.index("Close")

# ==============================
# OBTENER DATOS CON YFINANCE (CORREGIDO)
# ==============================
def get_recent_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Descarga datos recientes desde Yahoo Finance.
    👈 Cambiado a '1y' y añadido auto_adjust=False para evitar warnings y garantizar datos suficientes.
    """
    try:
        # 👈 auto_adjust=False silencia el FutureWarning y mantiene precios reales
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [c.lower().strip() for c in df.columns]
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
            
        df = df[required_cols].copy()
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", 
                           "close": "Close", "volume": "Volume"}, inplace=True)
        df.dropna(inplace=True)
        return df
        
    except Exception as e:
        logging.error(f"❌ Error al descargar {ticker}: {e}")
        return pd.DataFrame()

# ==============================
# INDICADORES TÉCNICOS (IDÉNTICO A TRAIN)
# ==============================
def add_indicators(df):
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
    df.dropna(inplace=True)  # 👈 Aquí se pierden ~100 filas iniciales por EMA100
    return df

# ==============================
# INFERENCIA Y SEÑALES
# ==============================
def predict_and_signal():
    model_files = glob.glob(os.path.join(MODEL_DIR, "best_lstm_*.keras"))
    if not model_files:
        logging.error("❌ No se encontraron modelos. Ejecuta train_model.py primero.")
        return

    results = []
    
    for model_path in tqdm(model_files, desc="Generando predicciones"):
        ticker = os.path.basename(model_path).replace("best_lstm_", "").replace(".keras", "")
        if ticker not in TICKERS: continue
            
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl")
        if not os.path.exists(scaler_path): continue

        try:
            # 👈 Descarga 1 año para asegurar que tras dropna() queden >= 180 filas
            df = get_recent_data(ticker, period="1y", interval="1d")
            if df.empty: continue

            df = add_indicators(df)
            
            # 👈 Validación real POST-indicadores
            if len(df) < SEQ_LEN:
                logging.warning(f"⚠️ {ticker}: Solo {len(df)} filas válidas tras indicadores (necesarias: {SEQ_LEN})")
                continue

            current_price = df['Close'].iloc[-1]
            current_volume = df['Volume'].iloc[-1]

            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            model = load_model(model_path)

            scaled_data = scaler.transform(df[FEATURE_COLS])
            X_seq = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
            pred_scaled = model.predict(X_seq, verbose=0)[0][0]

            dummy = np.zeros((1, len(FEATURE_COLS)))
            dummy[0, CLOSE_IDX] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, CLOSE_IDX]

            change_pct = (pred_price / current_price) - 1
            
            if change_pct >= THRESHOLD_PCT:
                signal, confidence = "🟢 BUY", min((change_pct / THRESHOLD_PCT) * 100, 999)
            elif change_pct <= -THRESHOLD_PCT:
                signal, confidence = "🔴 SELL", min((abs(change_pct) / THRESHOLD_PCT) * 100, 999)
            else:
                signal, confidence = "🟡 HOLD", 0

            results.append({
                "Ticker": ticker, "Current_Price": round(current_price, 6),
                "Predicted_24h": round(pred_price, 6), "Change_Pct": round(change_pct * 100, 2),
                "Signal": signal, "Confidence": round(confidence, 1),
                "Volume": f"{current_volume:,.0f}", "Model": os.path.basename(model_path)
            })
            time.sleep(0.5)  # Rate limit seguro

        except Exception as e:
            logging.error(f"⚠️ Error en {ticker}: {e}")
            continue

    if results:
        df_res = pd.DataFrame(results).sort_values("Change_Pct", ascending=False)
        print("\n" + "="*85)
        print(f"📊 SEÑALES DE TRADING - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"🎯 Umbral: ±{THRESHOLD_PCT*100}% | Ventana: {SEQ_LEN} días")
        print("="*85)
        
        display_df = df_res.copy()
        display_df["Current_Price"] = display_df["Current_Price"].apply(lambda x: f"${x:,.2f}")
        display_df["Predicted_24h"] = display_df["Predicted_24h"].apply(lambda x: f"${x:,.2f}")
        display_df["Change_Pct"] = display_df["Change_Pct"].apply(lambda x: f"{x:+.2f}%")
        
        print(display_df[["Ticker", "Current_Price", "Predicted_24h", "Change_Pct", "Signal", "Confidence"]].to_string(index=False))
        print("="*85 + "\n")
        
        buys = len(df_res[df_res["Signal"] == "🟢 BUY"])
        sells = len(df_res[df_res["Signal"] == "🔴 SELL"])
        holds = len(df_res[df_res["Signal"] == "🟡 HOLD"])
        logging.info(f"📈 Resumen: {buys} BUY | {sells} SELL | {holds} HOLD")
        
        output_path = os.path.join(MODEL_DIR, f"signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv")
        df_res.to_csv(output_path, index=False)
        logging.info(f"💾 CSV guardado: {output_path}")
    else:
        logging.warning("⚠️ No se generaron señales válidas.")

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia logs internos de TF
    logging.info("🚀 Iniciando generación de señales con yfinance...")
    predict_and_signal()