# train_model.py - Robust training with yfinance (real OHLCV)
# =========================================================
import numpy as np
import pandas as pd
import time
import logging
import pickle
import os
from tqdm import tqdm
import yfinance as yf  # 👈 Nueva dependencia
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ==============================
# CONFIGURACIÓN - TICKERS YFINANCE
# ==============================
# yfinance usa el formato "SYM-USD" para criptos
TICKERS = [
    "BTC-USD", "ETH-USD", "TRX-USD", "LINK-USD", "DOGE-USD",
    "SHIB-USD", "XRP-USD", "LTC-USD", "SOL-USD", "ADA-USD",
    # "NEXA-USD",  # 👈 No disponible en yfinance (comentado)
    # "NODL-USD",  # 👈 No disponible en yfinance (comentado)
    "DOT-USD"
]

SEQ_LEN = 180
EPOCHS = 100
BATCH_SIZE = 32
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# OBTENER DATOS CON YFINANCE (OHLCV REAL)
# ==============================
def get_crypto_data(ticker: str, period: str = "2y", interval: str = "1d"):
    """
    Descarga datos OHLCV reales desde Yahoo Finance.
    
    Args:
        ticker: Símbolo en formato yfinance (ej: "BTC-USD")
        period: Período de datos ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max")
        interval: Frecuencia ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
    
    Returns:
        DataFrame con columnas: Open, High, Low, Close, Volume
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty or len(df) < 100:
            logging.warning(f"⚠️ Datos insuficientes para {ticker} (filas: {len(df)})")
            return pd.DataFrame()
        
        # yfinance devuelve MultiIndex en columnas en algunas versiones
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Asegurar que las columnas estén en minúsculas y sin espacios
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Verificar columnas requeridas
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logging.warning(f"⚠️ Columnas faltantes en {ticker}: {set(required_cols) - set(df.columns)}")
            return pd.DataFrame()
        
        # Seleccionar y ordenar columnas
        df = df[required_cols].copy()
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", 
            "close": "Close", "volume": "Volume"
        }, inplace=True)
        
        # Eliminar filas con valores nulos
        df.dropna(inplace=True)
        
        logging.info(f"✅ {ticker}: {len(df)} filas | {df.index[0].date()} a {df.index[-1].date()}")
        return df
        
    except Exception as e:
        logging.error(f"❌ Error al descargar {ticker}: {e}")
        return pd.DataFrame()

# ==============================
# INDICADORES TÉCNICOS (robustos)
# ==============================
def add_indicators(df):
    df = df.copy()
    
    # Medias móviles simples
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    
    # Medias móviles exponenciales
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)  # 👈 Evita división por cero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Retornos y volatilidad
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    
    # Volumen relativo
    df["volume_ma"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"]
    
    # Limpieza explícita de infinitos y NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# ==============================
# CREAR SECUENCIAS
# ==============================
def create_sequences(data, seq_len, target_col_idx=0):
    """
    Crea secuencias para LSTM.
    
    Args:
        data: Array numpy escalado con todas las features
        seq_len: Longitud de la secuencia (ventana de tiempo)
        target_col_idx: Índice de la columna objetivo (0 = Close)
    
    Returns:
        X: Array de secuencias [samples, seq_len, features]
        y: Array de valores objetivo [samples]
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, target_col_idx])  # Predecir Close
    return np.array(X), np.array(y)

# ==============================
# ENTRENAMIENTO PRINCIPAL
# ==============================
for ticker in tqdm(TICKERS, desc="Entrenando monedas"):
    logging.info(f"\n📈 Iniciando entrenamiento: {ticker}")
    
    try:
        # 👇 Obtener datos OHLCV reales con yfinance
        df = get_crypto_data(ticker, period="2y", interval="1d")
        
        if df.empty or len(df) < 300:
            logging.warning(f"❌ Datos insuficientes para {ticker}. Saltando.")
            time.sleep(1)
            continue

        # Añadir indicadores técnicos
        df = add_indicators(df)
        if len(df) < SEQ_LEN + 100:
            logging.warning(f"❌ Pocos datos tras indicadores para {ticker}. Saltando.")
            time.sleep(1)
            continue

        # Columnas de features para el modelo
        feature_cols = [
            "Close", "RSI", "MACD", "MACD_signal", "Volume",
            "MA5", "MA10", "MA20", "EMA50", "EMA100",
            "returns", "volatility", "volume_ratio"
        ]

        # Escalado robusto (mejor para outliers en cripto)
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])

        # Guardar escalador para inferencia futura
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Crear secuencias para LSTM
        X, y = create_sequences(scaled_data, SEQ_LEN, target_col_idx=feature_cols.index("Close"))
        if len(X) < SEQ_LEN:
            logging.warning(f"❌ Secuencias insuficientes para {ticker}.")
            time.sleep(1)
            continue

        # Split train/test (80/20)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Arquitectura del modelo LSTM
        model = Sequential([
            Input(shape=(SEQ_LEN, len(feature_cols))),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        # Callbacks para entrenamiento eficiente
        early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, min_delta=1e-5)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
        best_model_path = os.path.join(MODEL_DIR, f"best_lstm_{ticker}.keras")
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_loss", mode="min")

        logging.info(f"🚀 Entrenando: {len(X_train)} secuencias | Val: {len(X_test)}")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=0
        )

        # Evaluación final
        val_loss = min(history.history['val_loss'])
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        logging.info(f"✅ {ticker} | Val Loss: {val_loss:.6f} | Test Loss: {test_loss:.6f} | Test MAE: {test_mae:.6f}")
        logging.info(f"   ┣ 📄 {best_model_path}")
        logging.info(f"   ┗ 📦 {scaler_path}")

    except Exception as e:
        logging.error(f"⚠️ Error crítico en {ticker}: {e}")
    finally:
        # Liberar memoria de Keras entre iteraciones
        tf.keras.backend.clear_session()
        time.sleep(1)  # Pequeña pausa para evitar sobrecarga

logging.info("\n🎉 Entrenamiento completado exitosamente.")