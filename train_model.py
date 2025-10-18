# =========================================================
# train_model.py - Entrenamiento robusto y corregido
# =========================================================
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os

# ==============================
# CONFIGURACIÓN
# ==============================
TICKERS = [
    "BTC-USD", "ETH-USD",
    "MATIC-USD", "LINK-USD",
    "DOGE-USD", "SHIB-USD",
    "XRP-USD", "LTC-USD",
    "SOL-USD", "ADA-USD",
    "NEXA-USD", "NODL-USD",
    "DOT-USD"
]

SEQ_LEN = 120      # ~5 días de datos horarios
EPOCHS = 100
BATCH_SIZE = 32
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# FUNCIÓN DE INDICADORES
# ==============================
def add_indicators(df):
    df = df.copy()
    # Precios
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Volatilidad y retornos
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    # Volumen normalizado
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
for TICKER in TICKERS:
    print(f"\n📈 Entrenando {TICKER}...")
    try:
        # Descargar datos
        df = yf.download(TICKER, period="2y", interval="1h")
        
        # ✅ CORRECCIÓN CLAVE: Asegurar DataFrame plano
        if df.empty:
            print(f"❌ Sin datos para {TICKER}.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Si sigue siendo insuficiente, intentar 1 año
        if len(df) < 1000:
            print(f"⚠️ Pocos datos con 2y, intentando 1y...")
            df = yf.download(TICKER, period="1y", interval="1h")
            if df.empty:
                print(f"❌ Sin datos con 1y para {TICKER}.")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        if len(df) < 500:
            print(f"❌ Datos insuficientes para {TICKER}.")
            continue

        # Añadir indicadores
        df = add_indicators(df)
        if len(df) < 400:
            print(f"❌ Pocos datos tras indicadores para {TICKER}.")
            continue

        # Normalización
        feature_cols = [
            "Close", "RSI", "MACD", "MACD_signal", "Volume",
            "MA5", "MA10", "MA20", "EMA50", "EMA100",
            "returns", "volatility", "volume_ratio"
        ]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])

        # Guardar scaler
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{TICKER}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Crear secuencias
        X, y = create_sequences(scaled_data, SEQ_LEN)
        if len(X) < 200:
            print(f"❌ Secuencias insuficientes para {TICKER}.")
            continue

        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Modelo
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

        # Callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)
        best_model_path = os.path.join(MODEL_DIR, f"best_lstm_{TICKER}.h5")
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_loss", mode="min")

        # Entrenar
        print(f"🚀 Entrenando con {len(X_train)} secuencias...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )

        # Guardar modelo final
        model_path = os.path.join(MODEL_DIR, f"lstm_{TICKER}.h5")
        model.save(model_path)

        val_loss = min(history.history['val_loss'])
        print(f"✅ Guardado. Val Loss: {val_loss:.6f}")
        print(f"   ┣ 📄 {model_path}")
        print(f"   ┗ 📦 {scaler_path}")

    except Exception as e:
        print(f"⚠️ Error en {TICKER}: {e}")
        continue

print(f"\n🎉 Entrenamiento completado para {len(TICKERS)} monedas.")