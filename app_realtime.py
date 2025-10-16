# app_realtime.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
import pickle
import requests  # <-- Para Telegram

# --- Auto refresh every 60s ---
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# --- For Telegram alerts ---
def send_telegram_message(message: str):
    """Envía un mensaje a tu chat privado en Telegram."""
    try:
        bot_token = st.secrets["telegram"]["BOT_TOKEN"]
        chat_id = st.secrets["telegram"]["CHAT_ID"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, data=payload)
    except Exception:
        # Silencioso si no hay secrets o hay error
        pass

# --- For news/sentiment ---
try:
    from newsapi import NewsApiClient
    NEWSAPI_ENABLED = True
except ImportError:
    NEWSAPI_ENABLED = False

# --- Global variables ---
CURRENT_PRICE = 0.0
RSI_LAST = 0.0
DATA_UPDATED = False

# --- FUNCTION TO FORMAT PRICES DYNAMICALLY ---
def format_price_dynamic(price: float) -> str:
    """Format price with adaptive decimals for cryptocurrencies."""
    if price >= 1:
        return f"${price:.2f}"
    elif price >= 1e-4:  # >= 0.0001
        return f"${price:.6f}".rstrip('0').rstrip('.')
    elif price >= 1e-8:  # >= 0.00000001
        return f"${price:.8f}".rstrip('0').rstrip('.')
    else:
        return f"${price:.2e}"

st.set_page_config(
    page_title="📈 Crypto Tracker — Real-Time Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUTURISTIC CSS (including mobile keyboard fix + darker title) ---
st.markdown("""
<style>
/* Animated dark background */
body {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: #ffffff;
    margin: 0;
    padding: 0;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
/* Remove top margin */
.css-18ni7ap, header, .stApp {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
/* Cards and styles */
.metric-card {
    background: rgba(0, 40, 60, 0.3);
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
    margin-bottom: 20px;
}
.stMetric {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    transition: all 0.3s ease;
}
.stMetric:hover {
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
    transform: scale(1.02);
}

/* 🎨 Custom style for the hamburger button (sidebar toggle) */
/* We'll replace it via JS, but keep this for fallback */
.stButton > button {
    background: linear-gradient(45deg, #00f2fe, #4facfe);
    color: #000;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.7);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(45deg, #4facfe, #00f2fe);
    box-shadow: 0 0 25px rgba(0, 255, 255, 1);
    transform: scale(1.05);
}

h1, h2, h3 {
    /* 🎨 Título ligeramente más oscuro: de #0ff a #0af */
    text-shadow: 0 0 10px rgba(0, 170, 255, 0.8);
    color: #0af;
}
.plotly-graph-div {
    background: #0f0c29 !important;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}
.news-card {
    padding: 12px;
    background: rgba(0, 255, 255, 0.05);
    border-radius: 10px;
    margin-bottom: 12px;
    border-left: 3px solid #0af;
}
.stExpander {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}

/* 🔑 FIX: Prevent keyboard from opening on mobile selectbox */
.stSelectbox > div > div > input {
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
    user-select: none !important;
    pointer-events: none !important;
}
.stSelectbox {
    -webkit-appearance: none;
    -moz-appearance: none;
    appearance: none;
}
</style>
""", unsafe_allow_html=True)

# --- DGT LOGO ---
logo_base64 = "data:image/png;base64,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"

st.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 20px; padding: 10px; background: rgba(0, 255, 255, 0.05); border-radius: 10px; box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);">
    <img src="{logo_base64}" width="60" style="filter: drop-shadow(0 0 8px #0af); border-radius: 5px;" />
    <h1 style="margin: 0; font-size: 2rem; text-shadow: 0 0 15px rgba(0, 170, 255, 0.8); color: #0af; letter-spacing: 1px;">📈 Crypto Tracker — Real-Time Monitoring + AI Prediction</h1>
</div>
""", unsafe_allow_html=True)

# --- VALIDATION OF PERIOD/INTERVAL ---
valid_intervals = {
    "1d": ["1m", "5m", "15m", "30m", "1h"],
    "5d": ["1m", "5m", "15m", "30m", "1h"],
    "1mo": ["5m", "15m", "30m", "1h", "1d"]
}

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    tickers = [
        # Major
        "BTC-USD", "ETH-USD",
        # Layer 1 & 2
        "SOL-USD", "ADA-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
        # Memecoins
        "DOGE-USD", "SHIB-USD", 
        # Others
        "XRP-USD", "LTC-USD", "NEXA-USD", "NODL-USD"
    ]
    selected_ticker = st.selectbox("Cryptocurrency", tickers)
    period = st.selectbox("Period", ["1d", "5d", "1mo"], index=0)
    allowed_intervals = valid_intervals[period]
    default_interval = "1m" if "1m" in allowed_intervals else allowed_intervals[0]
    interval = st.selectbox("Interval", allowed_intervals, index=allowed_intervals.index(default_interval))
    enable_alerts = st.checkbox("🔔 RSI Alerts", value=True)
    enable_telegram = st.checkbox("📲 Telegram Alerts", value=False)
    enable_news = st.checkbox("📰 News", value=False)
    st.markdown("---")
    st.caption("🔄 Auto-refresh every 60s")

ticker = selected_ticker

# --- Helper functions ---
def add_indicators(df):
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
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
    # Returns y volatility (igual que en train_model.py)
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()
    df.dropna(inplace=True)
    return df

# --- Load PRE-TRAINED model and scaler ---
@st.cache_resource
def load_trained_model_and_scaler(ticker):
    model_dir = "models"
    model_path = os.path.join(model_dir, f"best_lstm_{ticker}.h5")
    scaler_path = os.path.join(model_dir, f"scaler_{ticker}.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras import losses

        # Cargar el modelo SIN compilar para evitar errores de serialización
        model = load_model(model_path, compile=False)
        
        # Recompilar manualmente con la misma configuración usada en train_model.py
        model.compile(optimizer="adam", loss=losses.MeanSquaredError())

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.warning(f"⚠️ Error loading model/scaler for {ticker}: {e}")
        return None, None

# --- State ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'last_best_prediction' not in st.session_state:
    st.session_state.last_best_prediction = 0.0

# --- Update data (for display) ---
def update_data():
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty or len(data) < 10:
            if st.session_state.data is not None and len(st.session_state.data) > 0:
                return st.session_state.data
            else:
                st.error("❌ No data available.")
                st.stop()
        data = add_indicators(data)
        data.dropna(inplace=True)
        if data.empty:
            if st.session_state.data is not None and len(st.session_state.data) > 0:
                return st.session_state.data
            else:
                st.error("⚠️ Insufficient data after indicator calculation.")
                st.stop()
        return data
    except Exception as e:
        st.error(f"🚨 Error downloading data: {e}")
        if st.session_state.data is not None and len(st.session_state.data) > 0:
            return st.session_state.data
        else:
            st.stop()

# --- Force refresh (original button, unchanged) ---
if st.button("🔄 Force Refresh", use_container_width=True):
    st.session_state.data = update_data()
    st.rerun()

# --- Load data for display ---
with st.spinner("🌐 Loading real-time data..."):
    data = update_data()

if data is None or data.empty:
    if st.session_state.data is not None and len(st.session_state.data) > 0:
        data = st.session_state.data
    else:
        st.warning("Waiting for initial data...")
        st.stop()

st.session_state.data = data

# --- Metrics ---
current_price = float(data['Close'].iloc[-1])
first_open = float(data['Open'].iloc[0])
last_close = float(data['Close'].iloc[-1])
change_pct = ((last_close - first_open) / first_open) * 100

with st.container():
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current Price", format_price_dynamic(current_price))
    col2.metric("📊 Today's Change", f"{change_pct:.2f}%", delta=change_pct, delta_color="normal")
    col3.metric("🤖 Model", "Pre-trained LSTM" if load_trained_model_and_scaler(ticker)[0] else "Not available")
    st.markdown('</div>', unsafe_allow_html=True)

# --- RSI Alerts + Telegram ---
rsi_last = float(data['RSI'].iloc[-1])
alert_message = None

if enable_alerts:
    if rsi_last > 70:
        st.warning("⚠️ **RSI Alert**: Overbought (RSI > 70)")
        st.toast("⚠️ Overbought detected!", icon="⚠️")
        alert_message = f"⚠️ RSI Alert: {ticker} is OVERBOUGHT (RSI = {rsi_last:.2f})"
    elif rsi_last < 30:
        st.success("✅ **RSI Alert**: Oversold (RSI < 30)")
        st.balloons()
        alert_message = f"✅ RSI Alert: {ticker} is OVERSOLD (RSI = {rsi_last:.2f})"

# Enviar alerta de RSI por Telegram
if alert_message and enable_telegram:
    send_telegram_message(alert_message)

# --- Export CSV ---
csv = data.to_csv(index=True)
st.download_button(
    label="📥 Export Data to CSV",
    data=csv,
    file_name=f"{ticker}_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=True
)

# --- Short-term Predictions (using pre-trained model) ---
with st.expander("🤖 AI Prediction (Pre-trained LSTM)", expanded=False):
    model_lstm, scaler = load_trained_model_and_scaler(ticker)

    if model_lstm is None or scaler is None:
        st.warning(f"❌ No pre-trained model found for {ticker}. Please run `train_model.py` first.")
    else:
        # Descargar datos recientes (usamos 1h como en train_model.py)
        try:
            data_pred = yf.download(ticker, period="5d", interval="1h")
            data_pred = add_indicators(data_pred)
            data_pred.dropna(inplace=True)
            
            if len(data_pred) < 70:
                st.warning("⚠️ Not enough data for prediction (need ≥70 hourly points).")
            else:
                feature_cols = ["Close", "RSI", "MACD", "Volume", "MA5", "MA10", "MA20", "EMA50", "returns", "volatility"]
                scaled_data = scaler.transform(data_pred[feature_cols])
                seq_len = 60
                
                if len(scaled_data) >= seq_len:
                    last_sequence = scaled_data[-seq_len:].reshape(1, seq_len, len(feature_cols))
                    pred_scaled = model_lstm.predict(last_sequence, verbose=0)[0, 0]
                    
                    # Desescalar solo 'Close' (asumimos que es la columna 0)
                    dummy = np.zeros((1, len(feature_cols)))
                    dummy[0, 0] = pred_scaled
                    pred_price = scaler.inverse_transform(dummy)[0, 0]
                    
                    # Mostrar
                    change_pct_pred = ((pred_price - current_price) / current_price) * 100
                    st.subheader("Next Hour Prediction")
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Price", format_price_dynamic(pred_price))
                    col2.metric("Change", f"{change_pct_pred:.2f}%", delta=change_pct_pred, delta_color="normal")
                    
                    st.session_state.best_prediction = float(pred_price)
                    
                    # Alerta si cambia significativamente
                    if 'last_best_prediction' in st.session_state and st.session_state.last_best_prediction != 0:
                        last_pred = st.session_state.last_best_prediction
                        change = ((pred_price - last_pred) / last_pred) * 100
                        if abs(change) > 1.0:
                            if change > 0:
                                st.success(f"📈 Prediction ↑ {change:.2f}% → {format_price_dynamic(pred_price)}")
                                st.balloons()
                                # Enviar alerta de predicción por Telegram
                                if enable_telegram:
                                    pred_msg = f"📈 Prediction Alert: {ticker} ↑ {change:.2f}%\nNew prediction: {format_price_dynamic(pred_price)}"
                                    send_telegram_message(pred_msg)
                            else:
                                st.warning(f"📉 Prediction ↓ {change:.2f}% → {format_price_dynamic(pred_price)}")
                                if enable_telegram:
                                    pred_msg = f"📉 Prediction Alert: {ticker} ↓ {change:.2f}%\nNew prediction: {format_price_dynamic(pred_price)}"
                                    send_telegram_message(pred_msg)
                    st.session_state.last_best_prediction = pred_price
                else:
                    st.warning("⚠️ Not enough data to form a 60-step sequence.")
        except Exception as e:
            st.error(f"🚨 Error during prediction: {e}")

# --- Future Prediction (30-day history, hourly) ---
with st.expander("📈 Future Prediction (3-day history)", expanded=False):
    try:
        from prophet import Prophet
        model_available = True
    except Exception as e:
        st.warning(f"⚠️ Prophet not available: {e}")
        model_available = False

    if model_available:
        try:
            with st.spinner("⏳ Loading historical data for prediction..."):
                data_long = yf.download(ticker, period="30d", interval="1h")
                if data_long.empty or len(data_long) < 20:
                    st.warning("⚠️ Not enough historical data to predict.")
                else:
                    df_prophet = data_long[['Close']].reset_index()
                    df_prophet.columns = ['ds', 'y']
                    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)

                    prophet_model = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    prophet_model.fit(df_prophet)

                    future_periods = {"6 hours": 6, "1 day": 24, "3 days": 72}
                    for name, hours in future_periods.items():
                        future = prophet_model.make_future_dataframe(periods=hours, freq='H')
                        forecast = prophet_model.predict(future)
                        next_price = forecast['yhat'].iloc[-1]
                        st.write(f"**{name}**: {format_price_dynamic(next_price)}")

        except Exception as e:
            st.error(f"❌ Error in future prediction: {str(e)}")
    else:
        st.info("ℹ️ Prophet model not available.")

# --- News ---
if enable_news and NEWSAPI_ENABLED:
    st.markdown("---")
    st.subheader("📰 Relevant News")
    with st.spinner("🔍 Loading news..."):
        api_key = ""
        try:
            if hasattr(st, 'secrets') and "NEWSAPI_KEY" in st.secrets:
                api_key = st.secrets["NEWSAPI_KEY"]
        except Exception:
            api_key = ""
        if not api_key:
            st.info("ℹ️ NewsAPI key not configured. Add it in Streamlit Cloud Secrets.")
        else:
            try:
                crypto_map = {
                    "BTC-USD": "Bitcoin",
                    "ETH-USD": "Ethereum",
                    "SOL-USD": "Solana",
                    "ADA-USD": "Cardano",
                    "DOT-USD": "Polkadot",
                    "MATIC-USD": "Polygon",
                    "LINK-USD": "Chainlink",
                    "DOGE-USD": "Dogecoin",
                    "SHIB-USD": "Shiba Inu",
                    "PEPE-USD": "Pepe",
                    "XRP-USD": "XRP",
                    "LTC-USD": "Litecoin",
                    "NEXA-USD": "Nexa",
                    "NODL-USD": "Nodle"
                }
                query = crypto_map.get(ticker, ticker.split("-")[0])
                newsapi = NewsApiClient(api_key=api_key)
                all_articles = newsapi.get_everything(
                    q=query,
                    language='en',
                    sort_by='publishedAt',
                    page_size=3
                )
                if all_articles.get('status') == 'ok' and len(all_articles['articles']) > 0:
                    for article in all_articles['articles']:
                        st.markdown(f"""
                        <div class="news-card">
                            <b>{article['title']}</b><br>
                            <span style="font-size:12px;">{article['source']['name']} — {article['publishedAt'][:10]}</span><br>
                            <a href="{article['url']}" target="_blank" style="color:#0af;">Read more</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ No recent news found.")
            except Exception as e:
                st.error(f"❌ Error loading news: {str(e)}")

# --- Interactive chart ---
st.subheader("📊 Interactive Chart (Zoom + Hover + Prediction)")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
)

candlestick = go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Price',
    increasing_line_color='lime',
    decreasing_line_color='red'
)
fig.add_trace(candlestick, row=1, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data['MA5'], name='MA5', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA10'], name='MA10', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='purple')), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['EMA50'], name='EMA50', line=dict(color='cyan', dash='dot')), row=1, col=1)

fig.add_hline(y=current_price, line_dash="dash", line_color="yellow", 
              annotation_text=f"Current: {format_price_dynamic(current_price)}", 
              annotation_position="top right", row=1, col=1)

last_date = data.index[-1]
fig.add_trace(go.Scatter(x=[last_date], y=[current_price], 
                         mode='markers+text',
                         marker=dict(color='yellow', size=12),
                         text=[format_price_dynamic(current_price)],
                         textposition="top center",
                         name='Current Price'), row=1, col=1)

if 'best_prediction' in st.session_state:
    best_pred = st.session_state.best_prediction
    future_date = data.index[-1] + pd.Timedelta(hours=1)
    fig.add_trace(go.Scatter(x=[last_date, future_date], 
                             y=[current_price, best_pred], 
                             mode='lines',
                             line=dict(color='red', dash='dash', width=3),
                             name=f'Prediction'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[best_pred], 
                             mode='markers+text',
                             marker=dict(color='red', size=15, symbol='star'),
                             text=[format_price_dynamic(best_pred)],
                             textposition="top center",
                             name='Prediction'), row=1, col=1)

fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='green'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='cyan')), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.update_layout(
    title_text=f"{ticker} — Real-Time Monitoring",
    xaxis_rangeslider_visible=False,
    height=600,
    template="plotly_dark",
    paper_bgcolor="#0f0c29",
    plot_bgcolor="#0f0c29",
    font_color="white",
    title_font_color="#0af",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔁 This app auto-refreshes every 60 seconds.")

# --- 💡 JavaScript to replace the sidebar toggle icon with a neon ">>" ---
st.markdown(
    """
    <style>
    .neon-arrow {
        font-size: 16px;
        font-weight: bold;
        color: #0ff;
        text-shadow: 0 0 8px #0ff, 0 0 12px #0ff;
        cursor: pointer;
        padding: 8px;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .neon-arrow:hover {
        text-shadow: 0 0 12px #0ff, 0 0 20px #0ff;
        transform: scale(1.1);
    }
    </style>

    <script>
    // Wait for DOM to be ready
    const observer = new MutationObserver(() => {
        const hamburger = document.querySelector('.css-1v0mbdj');
        if (hamburger && !hamburger.classList.contains('neon-replaced')) {
            // Hide the original hamburger
            hamburger.style.visibility = 'hidden';
            hamburger.style.position = 'absolute';
            hamburger.style.opacity = '0';

            // Create our custom neon >> icon
            const neonArrow = document.createElement('span');
            neonArrow.className = 'neon-arrow neon-replaced';
            neonArrow.innerHTML = '>>';
            neonArrow.onclick = () => {
                // Trigger the original toggle
                hamburger.click();
            };

            // Insert before the original hamburger
            hamburger.parentNode.insertBefore(neonArrow, hamburger);

            observer.disconnect();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """,
    unsafe_allow_html=True
)

# ==============================
# DISCLAIMER
# ==============================
st.markdown("---")
st.markdown("## 🧾 Disclaimer")

st.markdown("""
This disclaimer (“Disclaimer”) applies to the use of the service, algorithm, web application, or platform accessible at **[algoritmo-de-prueba-6kfppv5cggvmkxzkxvrr5s.streamlit.app]** (hereinafter referred to as the “Service”).

### 1. Informational / Testing Purpose
The Service is provided solely for **testing**, **demonstration**, **research**, or **informational purposes**, and **does not guarantee accuracy, completeness, or suitability** for any specific purpose. The results, predictions, suggestions, or analyses generated by the algorithm should be considered **indicative only** and not as definitive advice.

### 2. No Professional Advice
Nothing in the Service should be construed as **legal, medical, financial, accounting, or any other type of professional advice**. Users should not make significant decisions based solely on the output of the algorithm without consulting qualified professionals in the corresponding field.

### 3. No Warranties
The Service is provided “as is” and “as available,” without any warranties of any kind, either express or implied, including but not limited to **warranties of merchantability, fitness for a particular purpose, accuracy, or non-infringement** of third-party rights.

### 4. Limitation of Liability
In no event shall the providers of the Service (developers, owners, operators) be liable for **any direct, indirect, incidental, special, punitive, or consequential damages**, including but not limited to loss of profits, data, reputation, or any other loss or damage, even if advised of the possibility of such damages, arising from the use or inability to use the Service.

### 5. Data Reliability / User Inputs
The results depend on the **quality, completeness, and accuracy of the data** or parameters entered by the user. It is not guaranteed that the algorithm is immune to errors, inconsistencies, or biases inherent in the input data or the model used.

### 6. Changes, Interruptions, and Modifications
The Service provider reserves the right to **modify, suspend, or discontinue** the Service, in whole or in part, temporarily or permanently, with or without notice. The provider shall not be liable for any damages resulting from such modifications or interruptions.

### 7. Intellectual Property and Licenses
Unless expressly stated otherwise, all **rights, titles, patents, copyrights, trademarks, and other intellectual property rights** related to the algorithm, source code, documentation, and other components are the exclusive property of the Service provider or its licensors.

### 8. Privacy and Data Protection
Any data entered by users will be subject to the **privacy policy and terms of use** applicable to the Service. The user declares that they possess the necessary rights to the data they provide and assume full responsibility for its use.

### 9. Jurisdiction and Governing Law
This Disclaimer shall be governed by and construed in accordance with the laws of the **United States**, without regard to its conflict of law provisions. Any dispute related to this Disclaimer shall be submitted to the exclusive jurisdiction of the **competent courts of the State of California**.

### 10. User Acceptance
By using the Service, the user **expressly accepts the terms** of this Disclaimer. If the user does not agree with any of these terms, they must **refrain from using the Service**.
""")