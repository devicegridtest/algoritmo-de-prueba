# algoritmo-de-prueba
# 📈 Crypto Tracker — Real-Time Cryptocurrency Monitoring & AI Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://algoritmo-de-prueba-6kfppv5cggvmkxzkxvrr5s.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-teal?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

**Crypto Tracker** is a real-time, AI-powered dashboard for monitoring cryptocurrency prices, technical indicators, and short-term predictions — all in a sleek, responsive, and mobile-friendly interface. Built with **Streamlit**, **yfinance**, **Prophet**, **LSTM**, and **Random Forest**, it delivers institutional-grade insights for traders, developers, and crypto enthusiasts.

---

## 🔍 Key Features

### 📊 Real-Time Market Data
- Live price tracking with **1-minute granularity** (configurable up to 1 month).
- Interactive **candlestick charts** with volume and RSI subplots.
- Dynamic price formatting (e.g., `$0.00000094` for PEPE, `$62,345.89` for BTC).

### 🤖 Multi-Model AI Predictions
- **Short-term forecasts** (next 1–5 minutes) using:
  - **LSTM Neural Networks**
  - **Random Forest Regressor**
  - Model comparison via **RMSE/MAE metrics**
- **Long-term outlook** (6h, 1d, 3d) powered by **Facebook Prophet**.
- Automatic selection of the **most accurate model** based on historical performance.

### 📈 Technical Analysis
- Real-time **RSI**, **MACD**, and **moving averages** (MA5/10/20, EMA50).
- **Overbought/Oversold alerts** (RSI > 70 / < 30) with visual notifications.

### 🌐 Global News Integration
- Fetches **breaking news** from 150,000+ sources via **NewsAPI**.
- Context-aware queries (e.g., "Shiba Inu", "Pepe", "Solana").
- Secure API key management using **Streamlit Secrets**.

### 📱 Mobile-Optimized UX
- Futuristic **cyberpunk UI** with animated dark gradient.
- **No keyboard pop-up** on mobile select menus.
- Fully responsive layout for **iOS & Android**.

### 🔧 Flexible Configuration
- Choose from **14+ cryptocurrencies**:
  - **Major**: BTC, ETH, SOL, ADA, DOT, MATIC, LINK
  - **Memecoins**: DOGE, SHIB, PEPE
  - **Emerging**: XRP, LTC, NEXA, NODL
- Adjustable **timeframes** (`1d`, `5d`, `1mo`) and **intervals** (`1m` to `1d`).

---

## 🚀 Live Demo

Try the app live on **Streamlit Community Cloud**:  
👉 [https://algoritmo-de-prueba-6kfppv5cggvmkxzkxvrr5s.streamlit.app/](https://algoritmo-de-prueba-6kfppv5cggvmkxzkxvrr5s.streamlit.app/)

> 💡 **Tip**: On mobile, tap the **three dots** → **"Add to Home Screen"** for an app-like experience.

---

## 🛠️ Installation & Deployment

### Prerequisites
- Python 3.9+
- `pip` package manager

### Local Setup
```bash
# Clone the repository
git clone https://github.com/your-username/crypto-tracker.git
cd crypto-tracker

# Install dependencies
pip install -r requirements.txt

# (Optional) Create .streamlit/secrets.toml for NewsAPI
echo 'NEWSAPI_KEY = "your_api_key_here"' > .streamlit/secrets.toml

# Run the app
streamlit run app_realtime.py
