# =========================================================
# app_realtime.py - Streamlit Dashboard (VERSIÓN FINAL - SIN ERRORES)
# =========================================================
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
import pickle
import requests
import threading
import time
import traceback

# ✅ IMPORTS DE FECHA - ÚNICA FORMA CORRECTA (sin conflictos)
from datetime import datetime, timedelta

from tensorflow.keras.models import load_model
from tensorflow.keras import losses


# ==============================
# CONFIGURACIÓN GLOBAL (DEBE COINCIDIR CON train_model.py)
# ==============================
SEQ_LEN = 90  # 👈 ¡CRÍTICO! Debe ser igual al usado en entrenamiento
MODEL_DIR = "models"
FEATURE_COLS = [
    "Close", "RSI", "MACD", "MACD_signal", "Volume",
    "MA5", "MA10", "MA20", "EMA50", "EMA100",
    "returns", "volatility", "volume_ratio"
]
CLOSE_IDX = FEATURE_COLS.index("Close")

# Tickers disponibles en yfinance (NEXA/NODL no disponibles)
ALL_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", 
    "TRX-USD", "LINK-USD", "DOGE-USD", "SHIB-USD", 
    "XRP-USD", "LTC-USD"
    # "NEXA-USD",  # ❌ No disponible en yfinance
    # "NODL-USD",  # ❌ No disponible en yfinance
]

# ==============================
# LOGGING CONFIG
# ==============================
logging.basicConfig(
    filename="alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True  # 👈 Evita warnings de re-configuración en Streamlit
)

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def format_price_dynamic(price: float) -> str:
    """Formatea precios según magnitud para visualización limpia."""
    if pd.isna(price) or price is None:
        return "$N/A"
    if price >= 1:
        return f"${price:,.2f}"
    elif price >= 1e-4:
        return f"${price:.6f}".rstrip('0').rstrip('.')
    elif price >= 1e-8:
        return f"${price:.8f}".rstrip('0').rstrip('.')
    else:
        return f"${price:.2e}"

def add_indicators(df):
    """Añade indicadores técnicos (IDÉNTICO a train_model.py para consistencia)."""
    if df.empty:
        return df
    df = df.copy()
    
    # Medias móviles
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    
    # RSI con protección contra división por cero 👈 CRÍTICO
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)  # 👈 Evita inf
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
    
    # Limpieza explícita de valores inválidos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

@st.cache_resource(ttl=3600)  # 👈 Cache por 1 hora para evitar recargas constantes
def load_trained_model_and_scaler(ticker):
    """Carga modelo y escalador pre-entrenados con manejo robusto de errores."""
    # Buscar modelo con extensión .keras (nuevo formato) o .h5 (legacy)
    model_path = os.path.join(MODEL_DIR, f"best_lstm_{ticker}.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, f"best_lstm_{ticker}.h5")  # Fallback
    
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl")
    
    if not os.path.exists(model_path):
        logging.warning(f"⚠️ Modelo no encontrado para {ticker}: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        logging.warning(f"⚠️ Escaler no encontrado para {ticker}: {scaler_path}")
        return None, None

    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss=losses.MeanSquaredError(), metrics=["mae"])
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        logging.info(f"✅ Modelo y escalador cargados para {ticker}")
        return model, scaler
    except Exception as e:
        logging.error(f"❌ Error cargando modelo/escalador para {ticker}: {e}")
        return None, None

def send_telegram_message(message: str):
    """Envía mensaje a Telegram usando secrets de Streamlit."""
    try:
        # Verificar que existen los secrets
        if "telegram" not in st.secrets:
            logging.debug("⚠️ Secrets de Telegram no configurados")
            return
            
        bot_token = st.secrets["telegram"]["BOT_TOKEN"]
        chat_id = st.secrets["telegram"]["CHAT_ID"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            logging.info(f"📤 Alerta enviada a Telegram: {message[:50]}...")
        else:
            logging.warning(f"⚠️ Telegram API error {response.status_code}: {response.text}")
    except Exception as e:
        logging.debug(f"⚠️ Error enviando Telegram (esperado en local): {e}")

def fetch_cryptopanic_news(ticker_symbol: str, limit: int = 3):
    """Obtiene noticias desde CryptoPanic API."""
    try:
        if "cryptopanic" not in st.secrets:
            return []
            
        api_token = st.secrets["cryptopanic"]["API_TOKEN"]
        symbol_map = {
            "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL",
            "ADA-USD": "ADA", "DOT-USD": "DOT", "TRX-USD": "TRX",
            "LINK-USD": "LINK", "DOGE-USD": "DOGE", "SHIB-USD": "SHIB",
            "XRP-USD": "XRP", "LTC-USD": "LTC"
        }
        symbol = symbol_map.get(ticker_symbol, "BTC")
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_token}&currencies={symbol}&public=true"
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []

        data = response.json()
        news_list = []
        for item in data.get("results", [])[:limit]:
            news_list.append({
                "title": item.get("title", "No title"),
                "published_at": item.get("published_at", "")[:10] if item.get("published_at") else "",
                "url": item.get("url", "#")
            })
        return news_list
    except Exception:
        return []

# ==============================
# ALERTAS EN SEGUNDO PLANO (Robusto para Cloud)
# ==============================
def hourly_alerts_background():
    """Hilo que ejecuta alertas globales cada hora sin bloquear Streamlit."""
    logging.info("🟢 Servicio de alertas iniciado (background thread).")
    
    while True:
        try:
            now = datetime.datetime.now()
            next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            seconds_until_next = max(60, (next_hour - now).total_seconds())  # Mínimo 60s

            logging.info("🚀 Ejecutando revisión global de alertas...")
            send_global_alerts()
            logging.info("✅ Revisión de alertas completada.")

        except Exception as e:
            logging.error(f"❌ Error en background alert thread: {e}")
            logging.error(traceback.format_exc())

        logging.info(f"⏰ Próxima ejecución en {int(seconds_until_next)}s ({next_hour.strftime('%H:%M')})")
        time.sleep(seconds_until_next)

def send_global_alerts():
    """Revisa todas las monedas y envía alertas RSI/Predicción a Telegram."""
    for ticker in ALL_TICKERS:
        try:
            # Datos para RSI (1 día, 1h)
            data = yf.download(ticker, period="1d", interval="1h", progress=False, auto_adjust=False)
            if data.empty or len(data) < 20:
                continue
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower().strip() for c in data.columns]
            
            if not all(col in data.columns for col in ["close", "volume"]):
                continue
                
            data = data[["open", "high", "low", "close", "volume"]].copy()
            data.rename(columns={"open": "Open", "high": "High", "low": "Low", 
                               "close": "Close", "volume": "Volume"}, inplace=True)
            
            data = add_indicators(data)
            if data.empty or 'RSI' not in data.columns:
                continue

            rsi_last = data['RSI'].iloc[-1]
            if pd.isna(rsi_last):
                continue
                
            if rsi_last > 70:
                send_telegram_message(f"⚠️ <b>{ticker}</b> OVERBOUGHT\nRSI: {rsi_last:.2f} (>70)")
            elif rsi_last < 30:
                send_telegram_message(f"✅ <b>{ticker}</b> OVERSOLD\nRSI: {rsi_last:.2f} (<30)")

            # Predicción LSTM (solo si hay modelo)
            model, scaler = load_trained_model_and_scaler(ticker)
            if model is None or scaler is None:
                continue

            # Datos para predicción (más historia para secuencia)
            data_pred = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=False)
            if data_pred.empty:
                continue
                
            if isinstance(data_pred.columns, pd.MultiIndex):
                data_pred.columns = data_pred.columns.get_level_values(0)
            data_pred.columns = [c.lower().strip() for c in data_pred.columns]
            data_pred = data_pred[["open", "high", "low", "close", "volume"]].copy()
            data_pred.rename(columns={"open": "Open", "high": "High", "low": "Low", 
                                    "close": "Close", "volume": "Volume"}, inplace=True)
            
            data_pred = add_indicators(data_pred)
            if len(data_pred) < SEQ_LEN:
                continue

            scaled_data = scaler.transform(data_pred[FEATURE_COLS])
            last_sequence = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
            pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            
            # Inversa del escalador (solo Close)
            dummy = np.zeros((1, len(FEATURE_COLS)))
            dummy[0, CLOSE_IDX] = pred_scaled
            pred_price = scaler.inverse_transform(dummy)[0, CLOSE_IDX]

            current_price = data['Close'].iloc[-1]
            if pd.isna(current_price) or current_price == 0:
                continue
                
            change_pct = ((pred_price - current_price) / current_price) * 100
            
            # Solo alertar si cambio significativo (>2%)
            if abs(change_pct) > 2.0:
                direction = "📈" if change_pct > 0 else "📉"
                send_telegram_message(
                    f"{direction} <b>{ticker}</b> Prediction\n"
                    f"Expected: {change_pct:+.2f}%\n"
                    f"Target: {format_price_dynamic(pred_price)}"
                )

        except Exception as e:
            logging.debug(f"⚠️ Error procesando {ticker} en alertas: {e}")
            continue  # Continuar con siguiente ticker

# ==============================
# INICIALIZACIÓN DE STREAMLIT
# ==============================
st.set_page_config(
    page_title="📈 Crypto Tracker — Real-Time + AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lanzar hilo de alertas SOLO UNA VEZ
if "alerts_thread_started" not in st.session_state:
    st.session_state.alerts_thread_started = True
    thread = threading.Thread(target=hourly_alerts_background, daemon=True)
    thread.start()
    logging.info("🧵 Background alert thread started")

# Auto-refresh cada 60 segundos
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, limit=None, key="datarefresh")
except ImportError:
    st.warning("⚠️ Instalar: `pip install streamlit-autorefresh` para auto-refresh")

# ==============================
# CSS FUTURISTA
# ==============================
st.markdown("""
<style>
    /* Fondo animado */
    .stApp { 
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29); 
        background-size: 400% 400%; 
        animation: gradientBG 20s ease infinite; 
    }
    @keyframes gradientBG { 
        0% { background-position: 0% 50%; } 
        50% { background-position: 100% 50%; } 
        100% { background-position: 0% 50%; } 
    }
    
    /* Tarjetas de métricas */
    .metric-card { 
        background: rgba(0, 40, 60, 0.4); 
        border-radius: 12px; 
        padding: 15px; 
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2); 
        margin-bottom: 20px; 
    }
    
    /* Botones neon */
    .stButton > button { 
        background: linear-gradient(45deg, #00f2fe, #4facfe); 
        color: #000; 
        font-weight: bold; 
        border: none; 
        border-radius: 8px; 
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        transition: all 0.3s ease;
    }
    .stButton > button:hover { 
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.9); 
        transform: scale(1.02); 
    }
    
    /* Títulos con glow */
    h1, h2, h3 { 
        text-shadow: 0 0 10px rgba(0, 170, 255, 0.8); 
        color: #0af !important; 
    }
    
    /* Noticias */
    .news-card { 
        padding: 12px; 
        background: rgba(0, 255, 255, 0.08); 
        border-radius: 10px; 
        margin-bottom: 12px; 
        border-left: 3px solid #0af; 
        transition: transform 0.2s;
    }
    .news-card:hover { 
        transform: translateX(5px); 
        background: rgba(0, 255, 255, 0.12); 
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER CON LOGO
# ==============================
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 15px; 
            margin-bottom: 20px; padding: 15px; 
            background: rgba(0, 255, 255, 0.1); 
            border-radius: 12px; 
            border: 1px solid rgba(0, 255, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);">
    <div style="font-size: 2.5rem;">🚀</div>
    <h1 style="margin: 0; font-size: 1.8rem; color: #0af; letter-spacing: 0.5px;">
        Crypto Tracker — Real-Time Monitoring + AI Prediction
    </h1>
</div>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_ticker = st.selectbox("Cryptocurrency", ALL_TICKERS, index=0)
    
    # Validación de intervalos según período
    valid_intervals = {
        "1d": ["1m", "5m", "15m", "30m", "1h"],
        "5d": ["15m", "30m", "1h"],
        "1mo": ["1h", "4h", "1d"]
    }
    period = st.selectbox("Period", ["1d", "5d", "1mo"], index=0)
    allowed_intervals = valid_intervals[period]
    interval = st.selectbox("Interval", allowed_intervals, index=0)
    
    st.divider()
    
    enable_alerts = st.checkbox("🔔 RSI Alerts", value=True, help="Alertas cuando RSI >70 o <30")
    enable_news = st.checkbox("📰 Crypto News", value=False, help="Noticias desde CryptoPanic")
    enable_predictions = st.checkbox("🤖 AI Predictions", value=True, help="Mostrar predicciones LSTM")
    
    st.divider()
    
    # Botón Telegram
    st.markdown("### 🔔 Telegram Bot")
    st.markdown(
        f"""
        <a href="https://t.me/LTCAlertaBot" target="_blank" style="text-decoration: none;">
            <button style="
                width: 100%;
                background: linear-gradient(45deg, #0088cc, #00aadd);
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: 12px;
                cursor: pointer;
                box-shadow: 0 0 15px rgba(0, 136, 204, 0.5);
            ">
                📲 Add Bot to Telegram
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
    
    st.caption("🔄 Auto-refresh: 60s")
    
    # Estado del sistema
    st.divider()
    st.markdown("### 🟢 System Status")
    st.markdown(f"- Models loaded: `{len([f for f in os.listdir(MODEL_DIR) if f.startswith('best_lstm_')])}`")
    st.markdown(f"- Last update: `{datetime.now().strftime('%H:%M:%S')}`")

ticker = selected_ticker

# ==============================
# CARGA DE DATOS
# ==============================
@st.cache_data(ttl=30)  # 👈 Cache de datos por 30 segundos
def fetch_crypto_data(ticker, period, interval):
    """Descarga y procesa datos con manejo de errores."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if data.empty:
            return None
            
        # Manejar MultiIndex de columnas
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Normalizar nombres
        data.columns = [c.lower().strip() for c in data.columns]
        
        # Verificar columnas requeridas
        required = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required):
            return None
            
        # Seleccionar y renombrar
        data = data[required].copy()
        data.rename(columns={
            "open": "Open", "high": "High", "low": "Low", 
            "close": "Close", "volume": "Volume"
        }, inplace=True)
        
        # Añadir indicadores
        data = add_indicators(data)
        
        return data if not data.empty else None
        
    except Exception as e:
        logging.error(f"❌ Error descargando {ticker}: {e}")
        return None

# Cargar datos
with st.spinner(f"📡 Loading {ticker} data..."):
    data = fetch_crypto_data(ticker, period, interval)

if data is None or data.empty:
    st.error(f"❌ No se pudieron cargar datos para **{ticker}**.\n\nPosibles causas:\n- Ticker no disponible en Yahoo Finance\n- Período/intervalo inválido\n- Problema de conexión")
    st.stop()

# ==============================
# MÉTRICAS PRINCIPALES
# ==============================
current_price = float(data['Close'].iloc[-1])
price_change = current_price - float(data['Open'].iloc[0])
price_change_pct = (price_change / float(data['Open'].iloc[0])) * 100

with st.container():
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "💰 Current Price", 
        format_price_dynamic(current_price),
        delta=f"{price_change_pct:+.2f}%" if not pd.isna(price_change_pct) else None,
        delta_color="normal"
    )
    col2.metric("📊 24h Volume", f"{data['Volume'].iloc[-1]:,.0f}")
    
    # RSI con color condicional
    rsi_val = data['RSI'].iloc[-1] if 'RSI' in data.columns and len(data) > 0 else None
    if rsi_val and not pd.isna(rsi_val):
        rsi_color = "🔴" if rsi_val > 70 else "🟢" if rsi_val < 30 else "🟡"
        col3.metric(f"{rsi_color} RSI", f"{rsi_val:.1f}")
    else:
        col3.metric("🟡 RSI", "N/A")
    
    # Estado del modelo
    model, scaler = load_trained_model_and_scaler(ticker)
    col4.metric("🤖 AI Model", "✅ Active" if model else "❌ Missing")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# ALERTAS RSI EN TIEMPO REAL
# ==============================
if enable_alerts and 'RSI' in data.columns and not pd.isna(rsi_val):
    if rsi_val > 70:
        st.warning(f"⚠️ **Overbought Signal**: RSI = {rsi_val:.2f} (>70)\n\nConsider taking profits or waiting for pullback.")
        if st.session_state.get(f"rsi_alert_{ticker}_overbought", True):
            send_telegram_message(f"⚠️ <b>{ticker}</b> OVERBOUGHT\nRSI: {rsi_val:.2f}\nPrice: {format_price_dynamic(current_price)}")
            st.session_state[f"rsi_alert_{ticker}_overbought"] = False
            st.session_state[f"rsi_alert_{ticker}_oversold"] = True  # Reset opposite
    elif rsi_val < 30:
        st.success(f"✅ **Oversold Signal**: RSI = {rsi_val:.2f} (<30)\n\nPotential buying opportunity (confirm with other indicators).")
        if st.session_state.get(f"rsi_alert_{ticker}_oversold", True):
            send_telegram_message(f"✅ <b>{ticker}</b> OVERSOLD\nRSI: {rsi_val:.2f}\nPrice: {format_price_dynamic(current_price)}")
            st.session_state[f"rsi_alert_{ticker}_oversold"] = False
            st.session_state[f"rsi_alert_{ticker}_overbought"] = True  # Reset opposite

# ==============================
# EXPORTAR DATOS
# ==============================
csv = data.to_csv(index=True)
st.download_button(
    label="📥 Export Data (CSV)",
    data=csv,
    file_name=f"{ticker}_{period}_{interval}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    use_container_width=True
)

# ==============================
# PREDICCIONES LSTM
# ==============================
if enable_predictions:
    with st.expander("🤖 AI Price Prediction (LSTM)", expanded=False):
        if model is None or scaler is None:
            st.warning(f"❌ Modelo no disponible para **{ticker}**.\n\nPara habilitar predicciones:\n1. Ejecuta `train_model.py`\n2. Verifica que los archivos estén en `/models/`")
        else:
            try:
                # Descargar datos para predicción (más historia para secuencia completa)
                data_pred = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
                
                if not data_pred.empty and isinstance(data_pred.columns, pd.MultiIndex):
                    data_pred.columns = data_pred.columns.get_level_values(0)
                data_pred.columns = [c.lower().strip() for c in data_pred.columns]
                data_pred = data_pred[["open", "high", "low", "close", "volume"]].copy()
                data_pred.rename(columns={"open": "Open", "high": "High", "low": "Low", 
                                         "close": "Close", "volume": "Volume"}, inplace=True)
                
                data_pred = add_indicators(data_pred)
                
                if len(data_pred) < SEQ_LEN:
                    st.warning(f"⚠️ Se necesitan al menos {SEQ_LEN} puntos de datos para predecir. Actualmente: {len(data_pred)}")
                else:
                    # Preparar secuencia
                    scaled_data = scaler.transform(data_pred[FEATURE_COLS])
                    last_sequence = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, len(FEATURE_COLS))
                    
                    # Predecir
                    pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
                    
                    # Inversa del escalador (solo columna Close)
                    dummy = np.zeros((1, len(FEATURE_COLS)))
                    dummy[0, CLOSE_IDX] = pred_scaled
                    pred_price = scaler.inverse_transform(dummy)[0, CLOSE_IDX]
                    
                    # Calcular cambio
                    change_pct = ((pred_price - current_price) / current_price) * 100 if current_price != 0 else 0
                    
                    # Mostrar resultados
                    col1, col2, col3 = st.columns(3)
                    col1.metric("🎯 Predicted Price", format_price_dynamic(pred_price))
                    col2.metric("📈 Expected Change", f"{change_pct:+.2f}%", delta=change_pct, delta_color="normal")
                    
                    # Señal basada en umbral
                    threshold = 1.5  # 1.5%
                    if change_pct >= threshold:
                        col3.markdown("### 🟢 BUY Signal")
                        st.success(f"Predicción sugiere **compra**: +{change_pct:.2f}% esperado")
                    elif change_pct <= -threshold:
                        col3.markdown("### 🔴 SELL Signal") 
                        st.warning(f"Predicción sugiere **venta**: {change_pct:.2f}% esperado")
                    else:
                        col3.markdown("### 🟡 HOLD Signal")
                        st.info(f"Sin señal clara: cambio esperado < ±{threshold}%")
                    
                    # Guardar para comparación en siguiente refresh
                    if 'last_prediction' not in st.session_state:
                        st.session_state.last_prediction = {}
                    st.session_state.last_prediction[ticker] = {
                        'price': float(pred_price),
                        'change': float(change_pct),
                        'timestamp': datetime.datetime.now()
                    }
                    
            except Exception as e:
                st.error(f"🚨 Error en predicción: `{str(e)}`")
                logging.error(f"Prediction error for {ticker}: {e}\n{traceback.format_exc()}")

# ==============================
# NOTICIAS CRYPTOPANIC
# ==============================
if enable_news:
    st.divider()
    st.subheader("📰 Latest Crypto News")
    
    with st.spinner("Fetching news..."):
        news_items = fetch_cryptopanic_news(ticker)
        
        if not news_items:
            st.info("ℹ️ No news available or API limit reached.")
        else:
            for i, item in enumerate(news_items, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="news-card">
                        <small style="color: #0af;">{item['published_at']}</small><br>
                        <strong>{i}. <a href="{item['url']}" target="_blank" style="color: #fff; text-decoration: none;">
                            {item['title']}
                        </a></strong>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================
# GRÁFICO INTERACTIVO
# ==============================
st.divider()
st.subheader("📊 Interactive Chart")

# Crear subplots: Precio, Volumen, RSI
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    specs=[[{"secondary_y": True}], [{}], [{}]]
)

# Candlestick principal
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'], high=data['High'], 
    low=data['Low'], close=data['Close'],
    name='OHLC',
    increasing_line_color='#00ff88',
    decreasing_line_color='#ff4444',
    showlegend=False
), row=1, col=1)

# Medias móviles
for ma, color, name in [
    ('MA5', '#ffa500', 'MA5'),
    ('MA10', '#4169e1', 'MA10'), 
    ('MA20', '#9370db', 'MA20'),
    ('EMA50', '#00ffff', 'EMA50')
]:
    if ma in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[ma], 
            name=name, line=dict(color=color, width=1),
            showlegend=True
        ), row=1, col=1)

# Línea de precio actual
fig.add_hline(
    y=current_price, 
    line_dash="dot", line_color="#ffff00", line_width=1,
    annotation_text=f"Current: {format_price_dynamic(current_price)}",
    annotation_position="right",
    row=1, col=1
)

# Predicción (si existe)
if enable_predictions and model and 'last_prediction' in st.session_state and ticker in st.session_state.last_prediction:
    pred = st.session_state.last_prediction[ticker]
    future_x = data.index[-1] + pd.Timedelta(days=1)  # Próximo día
    
    # Línea de predicción
    fig.add_trace(go.Scatter(
        x=[data.index[-1], future_x],
        y=[current_price, pred['price']],
        mode='lines',
        line=dict(color='#ff0066', dash='dash', width=2),
        name='AI Prediction',
        showlegend=True
    ), row=1, col=1)
    
    # Punto de predicción
    fig.add_trace(go.Scatter(
        x=[future_x], y=[pred['price']],
        mode='markers+text',
        marker=dict(color='#ff0066', size=12, symbol='star'),
        text=[format_price_dynamic(pred['price'])],
        textposition="top center",
        name='Target',
        showlegend=False
    ), row=1, col=1)

# Volumen
fig.add_trace(go.Bar(
    x=data.index, y=data['Volume'],
    name='Volume',
    marker_color='rgba(0, 255, 255, 0.5)',
    showlegend=False
), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(
    x=data.index, y=data['RSI'],
    name='RSI',
    line=dict(color='#00ffff', width=2),
    showlegend=False
), row=3, col=1)

# Líneas de sobrecompra/sobreventa en RSI
fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=3, col=1)
fig.add_annotation(x=data.index[-1], y=70, text="Overbought", 
                   showarrow=False, font=dict(size=9, color="#ff4444"), row=3, col=1)
fig.add_annotation(x=data.index[-1], y=30, text="Oversold", 
                   showarrow=False, font=dict(size=9, color="#00ff88"), row=3, col=1)

# Layout del gráfico
fig.update_layout(
    title=f"{ticker} — Technical Analysis",
    xaxis_rangeslider_visible=False,
    height=700,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    font_color="#ffffff",
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1,
        bgcolor="rgba(0,0,0,0.5)"
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified"
)

# Estilos de ejes
fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", row=1, col=1)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", title="Price (USD)", row=1, col=1)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", title="Volume", row=2, col=1)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", title="RSI", range=[0, 100], row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SECCIÓN DE BACKTESTING (CORREGIDO - PERSISTENTE)
# =========================================================

# 👇 1. IMPORT SEGURO (no rompe la app si no existe el módulo)
BACKTEST_AVAILABLE = False
try:
    from backtest_engine import CryptoBacktester
    BACKTEST_AVAILABLE = True
except ImportError:
    pass

st.divider()
st.subheader("🧪 Backtesting de Estrategia")

# 👇 2. INICIALIZAR SESSION STATE (solo primera vez)
if "bt_results" not in st.session_state:
    st.session_state.bt_results = None
if "bt_params" not in st.session_state:
    st.session_state.bt_params = {}

with st.expander("🔬 Ejecutar Backtest Histórico", expanded=False):
    st.markdown("""
    **Valida tu estrategia** simulando cómo hubiera rendido el modelo en datos históricos.
    
    ⚙️ **Parámetros configurables:**
    - Capital inicial, tamaño de posición, comisiones
    - Stop Loss / Take Profit opcionales
    - Umbral de señal para filtrar operaciones débiles
    """)
    
    # Configuración del backtest
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bt_start = st.date_input(
            "📅 Fecha Inicio", 
            value=datetime.now() - timedelta(days=180),
            max_value=datetime.now() - timedelta(days=30),
            key="bt_start_input"
        )
        bt_end = st.date_input(
            "📅 Fecha Fin",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now() - timedelta(days=1),
            key="bt_end_input"
        )
        bt_interval = st.selectbox("⏱️ Intervalo", ["1d", "1h"], index=0, key="bt_interval_input")
    
    with col2:
        bt_capital = st.number_input("💰 Capital Inicial ($)", min_value=100, value=10000, step=100, key="bt_capital_input")
        bt_position = st.slider("📊 Tamaño de Posición (%)", 10, 100, 100, key="bt_position_input") / 100
        bt_fee = st.number_input("🧾 Comisión (%)", 0.0, 1.0, 0.1, step=0.01, key="bt_fee_input") / 100
    
    with col3:
        bt_threshold = st.slider("🎯 Umbral de Señal (%)", 0.5, 5.0, 1.5, step=0.1, key="bt_threshold_input")
        bt_sl = st.number_input("🛑 Stop Loss (%)", 0.0, 50.0, 0.0, step=0.5, key="bt_sl_input") / 100 or None
        bt_tp = st.number_input("🎯 Take Profit (%)", 0.0, 100.0, 0.0, step=1.0, key="bt_tp_input") / 100 or None
    
    # 👇 3. BOTÓN DE EJECUCIÓN (solo corre al hacer clic)
    if st.button("🚀 Ejecutar Backtest", use_container_width=True, type="primary", key="bt_run_button"):
        if bt_start >= bt_end:
            st.error("❌ La fecha de inicio debe ser anterior a la fecha de fin")
        else:
            with st.spinner(f"🔄 Ejecutando backtest para {ticker}..."):
                if not BACKTEST_AVAILABLE:
                    st.error("❌ Módulo `backtest_engine.py` no encontrado.")
                else:
                    model, scaler = load_trained_model_and_scaler(ticker)
                    
                    if model is None or scaler is None:
                        st.error(f"❌ Modelo no disponible para {ticker}. Ejecuta `train_model.py` primero.")
                    else:
                        try:
                            backtester = CryptoBacktester(
                                initial_capital=bt_capital,
                                position_size=bt_position,
                                transaction_fee=bt_fee,
                                stop_loss_pct=bt_sl,
                                take_profit_pct=bt_tp,
                                min_confidence=0
                            )
                            
                            results = backtester.run_backtest(
                                ticker=ticker,
                                model=model,
                                scaler=scaler,
                                feature_cols=FEATURE_COLS,
                                close_idx=CLOSE_IDX,
                                seq_len=SEQ_LEN,
                                start_date=bt_start.strftime("%Y-%m-%d"),
                                end_date=bt_end.strftime("%Y-%m-%d"),
                                threshold_pct=bt_threshold,
                                interval=bt_interval
                            )
                            
                            if 'error' in results:
                                st.error(f"❌ {results['error']}")
                            else:
                                # 💾 GUARDAR RESULTADOS EN SESSION STATE (PERSISTE ENTRE REFRESHES)
                                st.session_state.bt_results = results
                                st.session_state.bt_params = {
                                    'ticker': ticker,
                                    'start': bt_start,
                                    'end': bt_end,
                                    'interval': bt_interval,
                                    'capital': bt_capital
                                }
                                st.success(f"✅ Backtest completado: {results['total_trades']} operaciones")
                                
                        except Exception as e:
                            st.error(f"🚨 Error en backtest: {e}")
                            logging.error(f"Backtest error: {e}\n{traceback.format_exc()}")

# 👇 4. VISUALIZE RESULTS (IS EXECUTED IN EACH REFRESH IF THERE IS SAVED DATA)
if st.session_state.bt_results is not None:
    results = st.session_state.bt_results
    params = st.session_state.bt_params
    
    # Header informativo
    st.info(f"📊 Resultados para **{params.get('ticker', ticker)}** | {params.get('start')} → {params.get('end')} | Intervalo: {params.get('interval', '1d')}")
    
    # === MÉTRICAS PRINCIPALES (SIN key= en st.metric) ===
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "📈 Retorno Total", 
        f"{results['total_return_pct']:+.2f}%",
        delta=f"${results['final_capital'] - results['initial_capital']:+,.2f}"
        # 👈 SIN key= aquí (st.metric no lo soporta)
    )
    m2.metric(
        "📉 Max Drawdown", 
        f"{results['max_drawdown_pct']:.2f}%",
        delta_color="inverse"
    )
    m3.metric(
        "⚡ Sharpe Ratio", 
        f"{results['sharpe_ratio']:.2f}",
        help=">1 = Bueno, >2 = Excelente"
    )
    m4.metric(
        "🎯 Win Rate", 
        f"{results['win_rate_pct']:.1f}%",
        help=f"{results['winning_trades']} wins / {results['losing_trades']} losses"
    )
    
    # === GRÁFICO: EQUITY CURVE + DRAWDOWN ===
    fig_bt = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=["💰 Equity Curve", "📉 Drawdown"]
    )
    
    equity_df = results['equity_curve']
    
    # Buy & Hold para comparar
    prices_bt = pd.Series(results['actual_prices'], index=results['dates'])
    bah_returns = prices_bt.pct_change().fillna(0)
    bah_equity = params.get('capital', 10000) * (1 + bah_returns.cumsum())
    
    fig_bt.add_trace(go.Scatter(
        x=equity_df['date'], y=equity_df['equity'],
        name='🤖 Strategy Equity',
        line=dict(color='#00ff88', width=2)
    ), row=1, col=1)
    
    fig_bt.add_trace(go.Scatter(
        x=prices_bt.index, y=bah_equity,
        name='📊 Buy & Hold',
        line=dict(color='#8888ff', width=1, dash='dot')
    ), row=1, col=1)
    
    # Drawdown
    fig_bt.add_trace(go.Scatter(
        x=equity_df['date'], y=equity_df['drawdown'],
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255, 50, 50, 0.3)',
        line=dict(color='#ff4444', width=1)
    ), row=2, col=1)
    
    fig_bt.update_layout(
        title=f"{params.get('ticker', ticker)} — Backtest Performance",
        height=500,
        template="plotly_dark",
        showlegend=True,
        hovermode="x unified"
    )
    fig_bt.update_yaxes(title="Capital ($)", row=1, col=1)
    fig_bt.update_yaxes(title="Drawdown (%)", row=2, col=1)
    
    # 👇 st.plotly_chart SÍ acepta key=
    st.plotly_chart(fig_bt, use_container_width=True, key="bt_plotly_chart")
    
    # === MÉTRICAS ADICIONALES (SIN key= en st.metric) ===
    with st.expander("📊 Métricas Detalladas", expanded=False):
        det_col1, det_col2 = st.columns(2)
        
        with det_col1:
            st.markdown("##### 🎯 Calidad de Predicción")
            st.metric("Accuracy Dirección", f"{results['direction_accuracy_pct']:.1f}%")
            st.metric("Correlación Precio", f"{results['prediction_correlation']:.3f}" if not np.isnan(results['prediction_correlation']) else "N/A")
            st.metric("Profit Factor", f"{results['profit_factor']:.2f}" if results['profit_factor'] != 999 else ">999")
        
        with det_col2:
            st.markdown("##### 💼 Gestión de Trades")
            st.metric("Avg Win", f"{results['avg_win_pct']:+.2f}%")
            st.metric("Avg Loss", f"{results['avg_loss_pct']:+.2f}%")
            ratio = abs(results['avg_win_pct'] / results['avg_loss_pct']) if results['avg_loss_pct'] != 0 else 0
            st.metric("Ratio Win/Loss", f"{ratio:.2f}x")
    
    # === TABLA DE OPERACIONES ===
    if results['trades']:
        with st.expander(f"📋 Historial de Trades ({len(results['trades'])} operaciones)", expanded=False):
            trades_df = pd.DataFrame([{
                'Fecha': t['date'].strftime("%Y-%m-%d"),
                'Tipo': t['type'],
                'Entrada': f"${t['entry_price']:.4f}",
                'Salida': f"${t['exit_price']:.4f}" if t['exit_price'] else 'Abierta',
                'P&L ($)': f"{t['pnl']:+.2f}",
                'P&L (%)': f"{t['pnl_pct']:+.2f}%",
                'Razón': t['exit_reason']
            } for t in results['trades']])
            
            # Colorear P&L
            def color_pnl(val):
                if isinstance(val, str) and '+' in val:
                    return 'color: #00ff88'
                elif isinstance(val, str) and '-' in val:
                    return 'color: #ff4444'
                return ''
            
            # 👇 st.dataframe SÍ acepta key=
            st.dataframe(
                trades_df.style.map(color_pnl, subset=['P&L ($)', 'P&L (%)']),
                use_container_width=True,
                hide_index=True,
                key="bt_trades_df"
            )
            
            # Exportar trades a CSV
            csv_trades = pd.DataFrame(results['trades']).to_csv(index=False)
            # 👇 st.download_button SÍ acepta key=
            st.download_button(
                label="📥 Exportar Trades (CSV)",
                data=csv_trades,
                file_name=f"{params.get('ticker', ticker)}_backtest_trades_{params.get('start', datetime.now().date()).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="bt_download_csv"
            )
    
    # === RECOMENDACIÓN FINAL ===
    st.divider()
    st.markdown("### 🎯 Conclusión")
    
    if results['total_return_pct'] > 10 and results['sharpe_ratio'] > 1 and results['max_drawdown_pct'] > -30:
        st.success("✅ **Estrategia VIABLE**: Retorno atractivo con riesgo controlado. Considera operar con capital real (con precaución).")
    elif results['total_return_pct'] > 0 and results['win_rate_pct'] > 50:
        st.warning("⚠️ **Estrategia PROMETEDORA**: Resultados positivos pero con margen de mejora. Prueba ajustando parámetros.")
    else:
        st.error("❌ **Estrategia NO RENTABLE**: El modelo no genera señales consistentes. Revisa: threshold, indicadores, o entrena con más datos.")
    
    st.info("💡 **Tip**: Un Sharpe >1 y Drawdown <25% son buenos indicadores de robustez. La dirección accuracy >55% sugiere que el modelo captura tendencias.")
    
    # 👇 Botón para limpiar (SÍ acepta key=)
    if st.button("🗑️ Limpiar Resultados", key="bt_clear_button"):
        st.session_state.bt_results = None
        st.session_state.bt_params = {}
        st.rerun()

elif not BACKTEST_AVAILABLE:
    st.warning("Backtesting⚠️ module not available. Create `backtest_engine.py` to enable this feature.")
    
# ==============================
# DISCLAIMER
# ==============================
st.divider()
with st.expander("🧾 Legal Disclaimer", expanded=False):
    st.markdown("""
    ### ⚠️ Important Notice
    
    This application is for **educational and research purposes only**. 
    
    - 🔹 **Not Financial Advice**: All predictions, signals, and analyses are generated by algorithms and should NOT be used as the sole basis for financial decisions.
    
    - 🔹 **No Guarantees**: Cryptocurrency markets are highly volatile. Past performance does not guarantee future results.
    
    - 🔹 **Risk Warning**: Trading cryptocurrencies carries substantial risk of loss. Only invest what you can afford to lose.
    
    - 🔹 **Data Accuracy**: Prices and indicators depend on third-party APIs (Yahoo Finance, CryptoPanic) which may have delays or errors.
    
    - 🔹 **Model Limitations**: AI predictions are probabilistic estimates, not certainties. Always conduct your own research (DYOR).
    
    *By using this application, you acknowledge and accept these terms.*
    """)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.6); font-size: 0.9rem;">
    🚀 Crypto Tracker v2.0 • Built with Streamlit + TensorFlow + yfinance • 
    <a href="#" style="color: #0af;">GitHub</a> • 
    <a href="#" style="color: #0af;">Documentation</a>
</div>
""", unsafe_allow_html=True)