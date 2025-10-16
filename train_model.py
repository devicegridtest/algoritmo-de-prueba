# =========================================================
# train_model.py - Entrenamiento múltiple de modelos LSTM
# =========================================================
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

# ==============================
# 1️⃣ CONFIGURACIÓN GENERAL
# ==============================
TICKERS = ["BTC-USD", "ETH-USD", "LTC-USD"]  # monedas a entrenar
SEQ_LEN = 60      # 60 pasos (1h x 60 = 2.5 días)
EPOCHS = 50
BATCH_SIZE = 64
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# 2️⃣ FUNCIÓN DE INDICADORES
# ==============================
def add_indicators(df):
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().add(1).rolling(14).apply(lambda x: (x[x > 0].mean() / abs(x[x < 0].mean())) if abs(x[x < 0].mean()) > 0 else 0)))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df.dropna(inplace=True)
    return df

# ==============================
# 3️⃣ FUNCIÓN PARA CREAR SECUENCIAS
# ==============================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])  # predecir el precio de cierre
    return np.array(X), np.array(y)

# ==============================
# 4️⃣ ENTRENAR MODELO POR MONEDA
# ==============================
for TICKER in TICKERS:
    print(f"\n📈 Entrenando modelo para {TICKER}...")
    try:
        # Descarga de datos históricos
        df = yf.download(TICKER, period="1y", interval="1h")
        df = add_indicators(df)

        # Normalización
        feature_cols = ["Close", "RSI", "MACD", "Volume", "MA5", "MA10", "MA20", "EMA50", "returns", "volatility"]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])

        # Guardar el scaler
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{TICKER}.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Crear secuencias
        X, y = create_sequences(scaled_data, SEQ_LEN)
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Construir modelo
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # Callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        best_model_path = os.path.join(MODEL_DIR, f"best_lstm_{TICKER}.h5")
        checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_loss", mode="min")

        # Entrenamiento
        print(f"🚀 Entrenando {TICKER}...")
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        # Guardar modelo final
        model_path = os.path.join(MODEL_DIR, f"lstm_{TICKER}.h5")
        model.save(model_path)

        print(f"✅ Modelo y scaler guardados para {TICKER}:")
        print(f"   ┣ 📄 {model_path}")
        print(f"   ┗ 📦 {scaler_path}")

    except Exception as e:
        print(f"⚠️ Error entrenando {TICKER}: {e}")
        continue

print("\n🎉 Entrenamiento completo para todas las monedas (BTC, ETH, LTC).")
