# =========================================================
# train_model.py - Entrenamiento con CoinGecko (funcional en Nepal)
# =========================================================
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os

# ==============================
# CONFIGURACIÓN CON IDs OFICIALES DE COINGECKO
# ==============================
COIN_IDS = [
    "bitcoin",          # BTC
    "ethereum",         # ETH
    "tron",             # MATIC/POL ✅ (ID correcto)
    "chainlink",        # LINK
    "dogecoin",         # DOGE
    "shiba-inu",        # SHIB
    "ripple",           # XRP
    "litecoin",         # LTC
    "solana",           # SOL
    "cardano",          # ADA
    "nexacoin",         # NEXA
    "nodle-network",    # NODL
    "polkadot"          # DOT
]

TICKER_MAP = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "tron": "TRX-USD",     
    "chainlink": "LINK-USD",
    "dogecoin": "DOGE-USD",
    "shiba-inu": "SHIB-USD",
    "ripple": "XRP-USD",
    "litecoin": "LTC-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD",
    "nexacoin": "NEXA-USD",
    "nodle-network": "NODL-USD",
    "polkadot": "DOT-USD"
}

SEQ_LEN = 120      # ~5 días de datos horarios
EPOCHS = 100
BATCH_SIZE = 32
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# OBTENER DATOS DE COINGECKO (SIN 'interval')
# ==============================
def get_crypto_data(coin_id: str, days: int = 89):
    """Obtiene datos horarios de CoinGecko (máx 90 días, sin usar 'interval')."""
    cg = CoinGeckoAPI()
    try:
        # ✅ SIN el parámetro 'interval' → devuelve datos horarios si 2 <= days <= 90
        data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
        df = pd.merge(prices, volumes, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Crear OHLC simple
        df['Open'] = df['Close'].shift(1)
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df.dropna(inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"❌ Error en CoinGecko para {coin_id}: {e}")
        return pd.DataFrame()

# ==============================
# INDICADORES TÉCNICOS
# ==============================
def add_indicators(df):
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df["volume_ma"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"]
    df.dropna(inplace=True)
    return df

# ==============================
# CREAR SECUENCIAS
# ==============================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ==============================
# ENTRENAMIENTO
# ==============================
for coin_id in COIN_IDS:
    ticker = TICKER_MAP[coin_id]
    print(f"\n📈 Entrenando {ticker} ({coin_id})...")
    try:
        df = get_crypto_data(coin_id, days=89)
        if df.empty or len(df) < 500:
            print(f"❌ Datos insuficientes para {ticker}.")
            continue

        df = add_indicators(df)
        if len(df) < 400:
            print(f"❌ Pocos datos tras indicadores para {ticker}.")
            continue

        feature_cols = [
            "Close", "RSI", "MACD", "MACD_signal", "Volume",
            "MA5", "MA10", "MA20", "EMA50", "EMA100",
            "returns", "volatility", "volume_ratio"
        ]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])

        scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        X, y = create_sequences(scaled_data, SEQ_LEN)
        # ✅ Cambio solicitado: de 200 a 1900 secuencias mínimas
        if len(X) < 1900:
            print(f"❌ Secuencias insuficientes para {ticker} (necesarias: 1900, disponibles: {len(X)}).")
            continue

        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(feature_cols))),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)
        best_model_path = os.path.join(MODEL_DIR, f"best_lstm_{ticker}.h5")
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_loss", mode="min")

        print(f"🚀 Entrenando con {len(X_train)} secuencias...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )

        model_path = os.path.join(MODEL_DIR, f"lstm_{ticker}.h5")
        model.save(model_path)

        val_loss = min(history.history['val_loss'])
        print(f"✅ Guardado. Val Loss: {val_loss:.6f}")
        print(f"   ┣ 📄 {model_path}")
        print(f"   ┗ 📦 {scaler_path}")

    except Exception as e:
        print(f"⚠️ Error en {ticker}: {e}")
        continue

print(f"\n🎉 Entrenamiento completado para {len(COIN_IDS)} monedas.")