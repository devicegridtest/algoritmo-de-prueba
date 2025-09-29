# app_realtime.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

# --- Auto refresh every 60s ---
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60000, limit=None, key="datarefresh")

# --- For Telegram alerts ---
TELEGRAM_ENABLED = False

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

# --- FUTURISTIC CSS ---
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
.stButton button {
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
.stButton button:hover {
    background: linear-gradient(45deg, #4facfe, #00f2fe);
    box-shadow: 0 0 25px rgba(0, 255, 255, 1);
    transform: scale(1.05);
}
h1, h2, h3 {
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
    color: #0ff;
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
    border-left: 3px solid #0ff;
}
.stExpander {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}
            
           /* Evitar que el teclado se abra en selectbox en móviles */
.stSelectbox > div > div > input {
    -webkit-user-select: none !important;
    -moz-user-select: none !important;
    -ms-user-select: none !important;
    user-select: none !important;
    pointer-events: none !important;
}

/* Opcional: mejorar apariencia en móviles */
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
    <img src="{logo_base64}" width="60" style="filter: drop-shadow(0 0 8px #0f0); border-radius: 5px;" />
    <h1 style="margin: 0; font-size: 2rem; text-shadow: 0 0 15px rgba(0, 255, 255, 0.8); color: #0ff; letter-spacing: 1px;">📈 Crypto Tracker — Real-Time Monitoring + AI Prediction</h1>
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
        "DOGE-USD", "SHIB-USD", "PEPE-USD",
        # Others
        "XRP-USD", "LTC-USD", "NEXA-USD", "NODL-USD"
    ]
    selected_ticker = st.selectbox("Cryptocurrency", tickers)
    period = st.selectbox("Period", ["1d", "5d", "1mo"], index=0)
    allowed_intervals = valid_intervals[period]
    default_interval = "1m" if "1m" in allowed_intervals else allowed_intervals[0]
    interval = st.selectbox("Interval", allowed_intervals, index=allowed_intervals.index(default_interval))
    enable_alerts = st.checkbox("🔔 RSI Alerts", value=True)
    enable_news = st.checkbox("📰 News", value=False)
    st.markdown("---")
    st.caption("🔄 Auto-refresh every 60s")

ticker = selected_ticker

# --- Helper functions ---
def add_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# --- Load models ---
@st.cache_resource
def load_models():
    models = {}
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 7)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        models["LSTM"] = model_lstm
    except Exception:
        models["LSTM"] = None

    from sklearn.ensemble import RandomForestRegressor
    model_rf = RandomForestRegressor(n_estimators=50)
    models["Random Forest"] = model_rf

    models["Prophet"] = None
    return models

models = load_models()

# --- State ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'last_best_prediction' not in st.session_state:
    st.session_state.last_best_prediction = 0.0

# --- Update data ---
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

# --- Force refresh ---
if st.button("🔄 Force Refresh", use_container_width=True):
    st.session_state.data = update_data()
    st.rerun()

# --- Load data ---
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
    active_models = ", ".join([k for k, v in models.items() if v is not None])
    col3.metric("🤖 Active Models", active_models if active_models else "None")
    st.markdown('</div>', unsafe_allow_html=True)

# --- RSI Alerts ---
rsi_last = float(data['RSI'].iloc[-1])
if enable_alerts:
    if rsi_last > 70:
        st.warning("⚠️ **RSI Alert**: Overbought (RSI > 70)")
        st.toast("⚠️ Overbought detected!", icon="⚠️")
    elif rsi_last < 30:
        st.success("✅ **RSI Alert**: Oversold (RSI < 30)")
        st.balloons()

# --- Export CSV ---
csv = data.to_csv(index=True)
st.download_button(
    label="📥 Export Data to CSV",
    data=csv,
    file_name=f"{ticker}_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    use_container_width=True
)

# --- Short-term Predictions ---
if len(data) >= 80:
    with st.expander("🤖 Predictions & Accuracy Comparison", expanded=False):
        features = ['Close', 'RSI', 'MACD', 'Volume', 'MA5', 'MA10', 'MA20']
        df_features = data[features].copy()
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features)
        sequence_length = 60
        X, y = create_sequences(scaled_data, sequence_length)
        
        if len(X) > 0:
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            predictions = {}
            metrics = {}
            
            # Random Forest
            def create_sequences_rf(data_array, seq_length):
                X_rf, y_rf = [], []
                for i in range(len(data_array) - seq_length):
                    X_rf.append(data_array[i + seq_length - 1])
                    y_rf.append(data_array[i + seq_length, 0])
                return np.array(X_rf), np.array(y_rf)
            
            X_rf_all, y_rf_all = create_sequences_rf(scaled_data, sequence_length)
            if len(X_rf_all) > 0:
                split_rf = int(0.8 * len(X_rf_all))
                X_train_rf, y_train_rf = X_rf_all[:split_rf], y_rf_all[:split_rf]
                X_test_rf, y_test_rf = X_rf_all[split_rf:], y_rf_all[split_rf:]
            else:
                X_train_rf = X_test_rf = y_train_rf = y_test_rf = np.array([])

            def inverse_scale_close(scaled_value):
                return scaler.data_min_[0] + scaled_value * scaler.data_range_[0]
            
            if models["Random Forest"] is not None and len(X_train_rf) > 0:
                models["Random Forest"].fit(X_train_rf, y_train_rf)
                last_sequence_rf = scaled_data[-1].reshape(1, -1)
                next_pred_rf_scaled = models["Random Forest"].predict(last_sequence_rf)[0]
                next_price_rf = inverse_scale_close(next_pred_rf_scaled)
                predictions["Random Forest"] = float(next_price_rf)
                y_pred_rf_scaled = models["Random Forest"].predict(X_test_rf)
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                metrics["Random Forest"] = {
                    "RMSE": np.sqrt(mean_squared_error(y_test_rf, y_pred_rf_scaled)),
                    "MAE": mean_absolute_error(y_test_rf, y_pred_rf_scaled)
                }

            # LSTM
            if models["LSTM"] is not None:
                try:
                    models["LSTM"].fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
                    last_sequence_lstm = scaled_data[-sequence_length:].reshape(1, sequence_length, len(features))
                    next_pred_lstm_scaled = models["LSTM"].predict(last_sequence_lstm, verbose=0)[0, 0]
                    next_price_lstm = inverse_scale_close(next_pred_lstm_scaled)
                    predictions["LSTM"] = float(next_price_lstm)
                    y_pred_lstm_scaled = models["LSTM"].predict(X_test, verbose=0).flatten()
                    metrics["LSTM"] = {
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lstm_scaled)),
                        "MAE": mean_absolute_error(y_test, y_pred_lstm_scaled)
                    }
                except Exception as e:
                    st.warning(f"LSTM error: {e}")

            if predictions:
                st.subheader("📊 Multi-Model Predictions")
                cols = st.columns(len(predictions))
                for i, (model_name, pred_price) in enumerate(predictions.items()):
                    change = ((pred_price - current_price) / current_price) * 100
                    cols[i].metric(model_name, format_price_dynamic(pred_price), delta=f"{change:.2f}%")

            if metrics:
                st.subheader("📈 Accuracy Comparison")
                metric_cols = st.columns(len(metrics))
                for i, (model_name, metric_dict) in enumerate(metrics.items()):
                    metric_cols[i].metric(model_name, f"RMSE: {metric_dict['RMSE']:.4f}", delta=f"MAE: {metric_dict['MAE']:.4f}")

            if metrics:
                best_model = min(metrics.keys(), key=lambda k: metrics[k]['RMSE'])
                best_prediction = predictions[best_model]
                st.session_state.best_prediction = best_prediction

                if 'last_best_prediction' not in st.session_state:
                    st.session_state.last_best_prediction = best_prediction
                else:
                    last_pred = st.session_state.last_best_prediction
                    if last_pred != 0:
                        change = ((best_prediction - last_pred) / last_pred) * 100
                        if abs(change) > 1.0:
                            if change > 0:
                                st.success(f"📈 Prediction ↑ {change:.2f}% → {format_price_dynamic(best_prediction)}")
                                st.balloons()
                            else:
                                st.warning(f"📉 Prediction ↓ {change:.2f}% → {format_price_dynamic(best_prediction)}")
                    st.session_state.last_best_prediction = best_prediction

# --- Future Prediction (CORRECTED & WITH DYNAMIC FORMAT) ---
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
        api_key = st.secrets.get("NEWSAPI_KEY", "")
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
                            <a href="{article['url']}" target="_blank" style="color:#0ff;">Read more</a>
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
    future_date = data.index[-1] + pd.Timedelta(minutes=1)
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
    title_font_color="#0ff",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔁 This app auto-refreshes every 60 seconds.")
