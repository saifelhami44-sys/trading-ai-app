import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import json
import os
import logging
import queue
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Page Config ---
st.set_page_config(
    page_title="PRO AI RADAR v16.1",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Optional Libraries ---
try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
    st.error("pip install requests")

try:
    import websocket
    WEBSOCKET_OK = True
except ImportError:
    st.error("pip install websocket-client")

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("radar.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- Config ---
HISTORY_LIMIT = 500
PRICE_HISTORY_MAX = 30
SIGNAL_HISTORY_MAX = 50
VERSION = "16.1-SCALP-OPT-STREAMLIT"
API_DELAY_SEC = 0.06
ML_RETRAIN_EVERY_N_CANDLES = 30
ML_MIN_TRAIN_SAMPLES = 15
CANDLE_PATTERN_SCORE = 20
VP_BINS = 30

TF_HISTORY_MAP = {
    "1m": 200, "3m": 150, "5m": 100, "15m": 120,
    "30m": 150, "1h": 150, "2h": 200, "4h": 250,
    "6h": 300, "8h": 300, "12h": 350, "1d": 300,
    "3d": 250, "1w": 200,
}

_DEFAULT_MEME_COINS = {
    "HMSTR", "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BOME",
    "BRETT", "POPCAT", "MOG", "SLERF", "MEW", "TURBO", "MYRO", "PONKE",
    "WEN", "SNAP", "BABYDOGE", "ELON", "SAMO", "DOBO", "KISHU"
}

def _load_meme_coins() -> set:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meme_coins.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            coins = set(str(c).upper() for c in data.get("meme_coins", []))
            if coins:
                return coins
        except:
            pass
    return set(_DEFAULT_MEME_COINS)

MEME_COINS = _load_meme_coins()
MAX_LEVERAGE_MEME = 5
MAX_LEVERAGE_STANDARD = 20
MEME_MIN_MTF = 5
STANDARD_MIN_MTF = 4
FUNDING_WARNING = -0.01
FUNDING_DANGER = -0.05
OI_DROP_PCT = 5.0

@dataclass
class Candle:
    __slots__ = ("t", "o", "h", "l", "c", "v")
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float

@dataclass
class AnalysisResult:
    direction: str = "WAIT"
    strength: str = "NEUTRAL"
    confidence: float = 0.0
    confluence: int = 0
    reason: str = ""
    score: float = 0.0
    entry_low: float = 0.0
    entry_high: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    rr: float = 0.0
    ema20: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    vwap: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_sig: float = 0.0
    macd_hist: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_mid: float = 0.0
    bb_pct: float = 0.5
    bb_bandwidth: float = 0.0
    obv: float = 0.0
    obv_ema: float = 0.0
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    gator_val: float = 0.0
    vwap_dev: float = 0.0
    vol_ratio: float = 0.0
    atr: float = 0.0
    jaw: float = 0.0
    teeth: float = 0.0
    lips: float = 0.0
    alligator_state: str = "SLEEPING"
    m_pattern: bool = False
    w_pattern: bool = False
    double_top: bool = False
    double_bottom: bool = False
    rising_wedge: bool = False
    falling_wedge: bool = False
    pattern_label: str = ""
    price_action_bias: str = "NEUTRAL"
    swing_low: float = 0.0
    swing_high: float = 0.0
    fib_levels: Dict = field(default_factory=dict)
    candle_strength: float = 0.0
    candle_open_price: float = 0.0
    buy_pressure: float = 0.0
    sell_pressure: float = 0.0
    rsi_divergence: str = "NONE"
    macd_divergence: str = "NONE"
    market_regime: str = "UNKNOWN"
    stoch_rsi_k: float = 50.0
    stoch_rsi_d: float = 50.0
    ichi_above_cloud: bool = False
    ichi_bullish_cloud: bool = False
    cci: float = 0.0
    williams_r: float = -50.0
    smart_sl: float = 0.0
    smart_tp: float = 0.0
    position_size_pct: float = 1.0
    win_rate_est: float = 0.0
    candle_pattern: str = "NONE"
    candle_pattern_bias: str = "NEUTRAL"
    vp_poc: float = 0.0
    vp_vah: float = 0.0
    vp_val: float = 0.0
    vp_bias: str = "NEUTRAL"
    funding_signal: str = "NEUTRAL"

# --- CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f0b90b;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
        margin-bottom: 1rem;
    }
    .price-display {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        background: linear-gradient(135deg, #1a1d24 0%, #0b0e11 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2b3139;
    }
    .signal-buy { color: #02c076; font-weight: bold; font-size: 1.5rem; }
    .signal-sell { color: #cf304a; font-weight: bold; font-size: 1.5rem; }
    .signal-wait { color: #888888; font-weight: bold; font-size: 1.5rem; }
    .metric-card {
        background-color: #1a1d24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #f0b90b;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; }
    .metric-value { font-size: 1.1rem; font-weight: bold; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1a1d24;
        padding: 5px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2b3139;
        color: #aeb4bc;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0b90b !important;
        color: #000000 !important;
    }
    .warning-box {
        background-color: #1a0a0a;
        border: 1px solid #cf304a;
        color: #ff4d4d;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .info-box {
        background-color: #0a1a0f;
        border: 1px solid #02c076;
        color: #02c076;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .neutral-box {
        background-color: #1a1d24;
        border: 1px solid #2b3139;
        color: #888;
        padding: 0.5rem;
        border-radius: 5px;
    }
    hr { border-color: #2b3139; margin: 0.5rem 0; }
    .stButton>button {
        background-color: #f0b90b;
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover { background-color: #d4a017; }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
def init_session_state():
    defaults = {
        'symbol': 'BTCUSDT',
        'interval': '3m',
        'price': 0.0,
        'prev_price': 0.0,
        'mark_price': 0.0,
        'candle_open_price': 0.0,
        'price_history': deque(maxlen=PRICE_HISTORY_MAX),
        'last_update': "--:--:--",
        'error_msg': "",
        'ml_status': "ML: ACTIVE" if SKLEARN_OK else "ML: OFF",
        'ws_status': "WS: OFF",
        'current_analysis': AnalysisResult(),
        'candle_deque': deque(maxlen=500),
        'ws_connected': False,
        'ws_last_ping': 0,
        'signal_history': deque(maxlen=SIGNAL_HISTORY_MAX),
        'last_signal_direction': "WAIT",
        'signal_stats': {"buy": 0, "sell": 0, "wait": 0},
        'ml_clf': None,
        'ml_scaler': None,
        'ml_candles_at_train': 0,
        'ml_last_bias': 0,
        'ml_last_prob': 0.5,
        'mtf_master': "WAIT",
        'mtf_agree': 0,
        'mtf_cached_results': {},
        'mtf_coin_data': {},
        'coin_lookup': {},
        'current_coin_base': "BTC",
        'initial_coin_order': [],
        'intel_sr_levels': [],
        'intel_trend500': "WAIT",
        'intel_trend_str': "",
        'intel_ema_stack': "--",
        'intel_structure': "--",
        'intel_slope': "--",
        'intel_liq_buy': 0.0,
        'intel_liq_sell': 0.0,
        'intel_liq_dom': "--",
        'intel_liq_zones': "--",
        'intel_liq_sweep': "--",
        'intel_news': [],
        'intel_news_sentiment': "--",
        'intel_last_refresh': 0,
        'funding_rate': 0.0,
        'funding_next': 0,
        'oi_value': 0.0,
        'oi_change_pct': 0.0,
        'oi_prev': 0.0,
        'last_api_call': 0,
        'running': True,
        'last_price_update': 0,
        'bt_signals': [],
        'bt_wins': 0,
        'bt_losses': 0,
        'bt_open': None,
        'ui_queue': queue.Queue(),
        'new_data_event': threading.Event(),
        'df_lock': threading.Lock(),
        'analysis_lock': threading.Lock(),
        'intel_lock': threading.Lock(),
        'ml_lock': threading.Lock(),
        'bt_lock': threading.Lock(),
        'api_lock': threading.Lock(),
        'ws_lock': threading.Lock(),
        'bg_semaphore': threading.Semaphore(3),
        '_session': None,
        '_last_funding_update': 0,
        '_last_oi_update': 0,
        'threads_started': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Helper Functions ---
def _get_history_limit(tf=None) -> int:
    tf = tf or st.session_state.interval
    return TF_HISTORY_MAP.get(tf, 300)

def _init_session():
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=4,
        pool_maxsize=8,
        max_retries=requests.adapters.Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
    )
    s = requests.Session()
    s.mount("https://", adapter)
    s.headers.update({"Accept": "application/json"})
    return s

if st.session_state._session is None:
    st.session_state._session = _init_session()

def _api_get(url, timeout=8):
    with st.session_state.api_lock:
        elapsed = time.time() - st.session_state.last_api_call
        if elapsed < API_DELAY_SEC:
            time.sleep(API_DELAY_SEC - elapsed)
        try:
            r = st.session_state._session.get(url, timeout=timeout)
            st.session_state.last_api_call = time.time()
            return r
        except Exception as e:
            st.session_state.last_api_call = time.time()
            raise e

def _is_meme_coin():
    base = st.session_state.symbol.replace("USDT", "") if st.session_state.symbol else ""
    return base in MEME_COINS

def _get_max_leverage():
    return MAX_LEVERAGE_MEME if _is_meme_coin() else MAX_LEVERAGE_STANDARD

def _get_min_mtf():
    return MEME_MIN_MTF if _is_meme_coin() else STANDARD_MIN_MTF

def _check_funding_risk():
    if st.session_state.funding_rate <= FUNDING_DANGER:
        return False, f"FUNDING DANGER: {st.session_state.funding_rate*100:.4f}% -- Short squeeze likely!"
    if st.session_state.funding_rate <= FUNDING_WARNING:
        return False, f"FUNDING WARNING: {st.session_state.funding_rate*100:.4f}% -- Crowd is short"
    return True, ""

def _check_oi_risk(price_direction="BUY"):
    if st.session_state.oi_change_pct < -OI_DROP_PCT and price_direction == "BUY":
        return False, f"OI DROP: {st.session_state.oi_change_pct:.1f}% -- Fake pump, liquidity leaving"
    if st.session_state.oi_change_pct < -OI_DROP_PCT and price_direction == "SELL":
        return True, ""
    return True, ""

def _klines_to_df(klines) -> pd.DataFrame:
    df = pd.DataFrame(klines).iloc[:, :6]
    df.columns = ["t", "o", "h", "l", "c", "v"]
    df[["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].astype(float)
    return df

def _deque_to_df() -> pd.DataFrame:
    with st.session_state.df_lock:
        if len(st.session_state.candle_deque) == 0:
            return pd.DataFrame()
        data = [
            {"t": c.t, "o": c.o, "h": c.h, "l": c.l, "c": c.c, "v": c.v}
            for c in st.session_state.candle_deque
        ]
        return pd.DataFrame(data)

def set_error(msg):
    st.session_state.error_msg = msg[:100]
    logging.warning("Error: %s", msg)

# --- Indicator Math ---
def compute_smma(series, period):
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)
    alpha = 1.0 / period
    smma = series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smma.iloc[:period - 1] = np.nan
    return smma

def compute_alligator(df):
    median_price = (df["h"] + df["l"]) / 2.0
    raw_jaw = compute_smma(median_price, 13)
    raw_teeth = compute_smma(median_price, 8)
    raw_lips = compute_smma(median_price, 5)
    df["jaw"] = raw_jaw.shift(8)
    df["teeth"] = raw_teeth.shift(5)
    df["lips"] = raw_lips.shift(3)
    df["jaw_cur"] = raw_jaw
    df["teeth_cur"] = raw_teeth
    df["lips_cur"] = raw_lips
    df["gator"] = (raw_jaw - raw_teeth).abs()
    return df

def detect_m_pattern(df, tolerance=0.005):
    highs = df["h"].values
    closes = df["c"].values
    lows = df["l"].values
    vols = df["v"].values if "v" in df.columns else np.ones(len(highs))
    vol_avg = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
    n = len(highs)
    if n < 15:
        return False, None
    for i in range(3, n - 8):
        peak1 = highs[i]
        if highs[i] <= highs[i-1] or highs[i] <= highs[i+1]:
            continue
        for j in range(i + 4, min(i + 18, n - 3)):
            peak2 = highs[j]
            if highs[j] <= highs[j-1] or highs[j] <= highs[j+1]:
                continue
            if abs(peak1 - peak2) / max(peak1, 1e-9) < tolerance:
                valley = min(lows[i+1:j])
                if peak1 > valley and peak2 > valley:
                    if j + 2 < n:
                        breakdown = (highs[j+1] < valley or highs[j+2] < valley
                                     or closes[j+1] < valley)
                        vol_confirm = vols[j+1] > vol_avg * 0.8 if j+1 < len(vols) else True
                        if breakdown and vol_confirm:
                            return True, {
                                "peak1": round(float(peak1), 6),
                                "peak2": round(float(peak2), 6),
                                "neckline": round(float(valley), 6),
                                "peak1_idx": int(i),
                                "peak2_idx": int(j)
                            }
    return False, None

def detect_w_pattern(df, tolerance=0.005):
    lows = df["l"].values
    closes = df["c"].values
    vols = df["v"].values if "v" in df.columns else np.ones(len(lows))
    vol_avg = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
    n = len(lows)
    if n < 15:
        return False, None
    for i in range(3, n - 8):
        bottom1 = lows[i]
        if lows[i] >= lows[i-1] or lows[i] >= lows[i+1]:
            continue
        for j in range(i + 4, min(i + 18, n - 3)):
            bottom2 = lows[j]
            if lows[j] >= lows[j-1] or lows[j] >= lows[j+1]:
                continue
            if abs(bottom1 - bottom2) / max(bottom1, 1e-9) < tolerance:
                peak = max(lows[i+1:j])
                if bottom1 < peak and bottom2 < peak:
                    if j + 2 < n:
                        breakout = (lows[j+1] > peak or lows[j+2] > peak
                                    or closes[j+1] > peak)
                        vol_confirm = vols[j+1] > vol_avg * 0.8 if j+1 < len(vols) else True
                        if breakout and vol_confirm:
                            return True, {
                                "bottom1": round(float(bottom1), 6),
                                "bottom2": round(float(bottom2), 6),
                                "neckline": round(float(peak), 6),
                                "bottom1_idx": int(i),
                                "bottom2_idx": int(j)
                            }
    return False, None

def detect_double_top(df, tolerance=0.008):
    highs = df["h"].values
    closes = df["c"].values
    n = len(highs)
    if n < 20:
        return False, None
    for i in range(5, n - 10):
        if highs[i] <= highs[i-1] or highs[i] <= highs[i+1]:
            continue
        for j in range(i + 6, min(i + 25, n - 3)):
            if highs[j] <= highs[j-1] or highs[j] <= highs[j+1]:
                continue
            if abs(highs[i] - highs[j]) / max(highs[i], 1e-9) < tolerance:
                neckline = min(closes[i:j])
                if j + 2 < n and closes[j+1] < neckline:
                    return True, {
                        "peak1": round(float(highs[i]), 6),
                        "peak2": round(float(highs[j]), 6),
                        "neckline": round(float(neckline), 6),
                        "target": round(float(neckline - (highs[i] - neckline)), 6)
                    }
    return False, None

def detect_double_bottom(df, tolerance=0.008):
    lows = df["l"].values
    closes = df["c"].values
    n = len(lows)
    if n < 20:
        return False, None
    for i in range(5, n - 10):
        if lows[i] >= lows[i-1] or lows[i] >= lows[i+1]:
            continue
        for j in range(i + 6, min(i + 25, n - 3)):
            if lows[j] >= lows[j-1] or lows[j] >= lows[j+1]:
                continue
            if abs(lows[i] - lows[j]) / max(lows[i], 1e-9) < tolerance:
                neckline = max(closes[i:j])
                if j + 2 < n and closes[j+1] > neckline:
                    return True, {
                        "bottom1": round(float(lows[i]), 6),
                        "bottom2": round(float(lows[j]), 6),
                        "neckline": round(float(neckline), 6),
                        "target": round(float(neckline + (neckline - lows[i])), 6)
                    }
    return False, None

def detect_rising_wedge(df, lookback=30):
    if len(df) < lookback:
        return False, None
    recent = df.tail(lookback)
    highs = recent["h"].values
    lows = recent["l"].values
    closes = recent["c"].values
    n = len(highs)
    x = np.arange(n)
    try:
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope * 1.1:
            low_trend_end = lows[0] + low_slope * (n - 1)
            if closes[-1] < low_trend_end:
                return True, {
                    "high_slope": round(float(high_slope), 6),
                    "low_slope": round(float(low_slope), 6),
                    "bias": "BEARISH"
                }
    except Exception:
        pass
    return False, None

def detect_falling_wedge(df, lookback=30):
    if len(df) < lookback:
        return False, None
    recent = df.tail(lookback)
    highs = recent["h"].values
    lows = recent["l"].values
    closes = recent["c"].values
    n = len(highs)
    x = np.arange(n)
    try:
        high_slope = np.polyfit(x, highs, 1)[0]
        low_slope = np.polyfit(x, lows, 1)[0]
        if high_slope < 0 and low_slope < 0 and high_slope < low_slope * 1.1:
            high_trend_end = highs[0] + high_slope * (n - 1)
            if closes[-1] > high_trend_end:
                return True, {
                    "high_slope": round(float(high_slope), 6),
                    "low_slope": round(float(low_slope), 6),
                    "bias": "BULLISH"
                }
    except Exception:
        pass
    return False, None

def compute_price_action_bias(df, cp):
    if len(df) < 10:
        return "NEUTRAL"
    n = min(15, len(df))
    recent = df.tail(n)
    closes = recent["c"].values
    highs = recent["h"].values
    lows = recent["l"].values
    vols = recent["v"].values
    vol_total = max(vols.sum(), 1e-9)
    bull_score = 0.0
    bear_score = 0.0
    for i in range(1, len(highs)):
        w = vols[i] / vol_total
        if highs[i] > highs[i-1]:
            bull_score += w * 10
        else:
            bear_score += w * 10
        if lows[i] > lows[i-1]:
            bull_score += w * 10
        else:
            bear_score += w * 10
    bull_bodies = sum(1 for i in range(len(closes)) if closes[i] > recent["o"].values[i])
    bear_bodies = len(closes) - bull_bodies
    if bull_bodies > bear_bodies + 3:
        bull_score += 0.05
    elif bear_bodies > bull_bodies + 3:
        bear_score += 0.05
    if bull_score > bear_score * 1.4:
        return "UPWARD"
    elif bear_score > bull_score * 1.4:
        return "DOWNWARD"
    return "NEUTRAL"

def analyze_alligator_state(jaw, teeth, lips):
    if jaw <= 0 or teeth <= 0 or lips <= 0:
        return "SLEEPING", "WAIT", "#888888"
    if lips > teeth > jaw:
        return "UPTREND", "BUY", "#02c076"
    elif lips < teeth < jaw:
        return "DOWNTREND", "SELL", "#cf304a"
    elif abs(lips - teeth) < (teeth * 0.002) and abs(teeth - jaw) < (jaw * 0.002):
        return "SLEEPING", "WAIT", "#888888"
    else:
        return "WAKING", "WAIT", "#f0b90b"

def compute_indicators(df):
    delta = df["c"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["stoch_rsi_k"] = df["stoch_rsi"].rolling(3).mean() * 100
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    ema12 = df["c"].ewm(span=12, adjust=False).mean()
    ema26 = df["c"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["pc"] = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - df["pc"]).abs(), (df["l"] - df["pc"]).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, min_periods=14).mean()
    df["vol_avg20"] = df["v"].rolling(20).mean()
    df["vol_ratio"] = (df["v"] / df["vol_avg20"].replace(0, np.nan)) * 100
    df["roc"] = df["c"].pct_change(5)
    df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()

    df["_date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df["_pv"] = df["c"] * df["v"]
    df["cum_vol"] = df.groupby("_date")["v"].cumsum()
    df["cum_vol_price"] = df.groupby("_date")["_pv"].cumsum()
    df["vwap"] = df["cum_vol_price"] / df["cum_vol"].replace(0, np.nan)
    df["vwap_dev"] = (df["c"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    df.drop(columns=["_date", "_pv"], inplace=True)

    df["macd_hist"] = df["macd"] - df["macd_sig"]
    for lag in [1, 2, 3]:
        df["close_lag" + str(lag)] = df["c"].shift(lag)
    df["swing_low_10"] = df["l"].rolling(window=10, min_periods=5).min()
    df["swing_high_10"] = df["h"].rolling(window=10, min_periods=5).max()
    df["swing_low_20"] = df["l"].rolling(window=20, min_periods=10).min()
    df["swing_high_20"] = df["h"].rolling(window=20, min_periods=10).max()

    df = compute_alligator(df)

    bb_mid = df["c"].rolling(20).mean()
    bb_std = df["c"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_mid"] = bb_mid
    df["bb_pct"] = (df["c"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    closes_arr = df["c"].values
    vols_arr = df["v"].values
    direction = np.sign(np.diff(closes_arr, prepend=closes_arr[0]))
    obv_arr = np.cumsum(direction * vols_arr)
    df["obv"] = obv_arr
    df["obv_ema"] = pd.Series(obv_arr).ewm(span=20, adjust=False).mean().values

    df = _compute_adx(df)

    high9 = df["h"].rolling(9).max(); low9 = df["l"].rolling(9).min()
    high26 = df["h"].rolling(26).max(); low26 = df["l"].rolling(26).min()
    high52 = df["h"].rolling(52).max(); low52 = df["l"].rolling(52).min()
    df["ichi_tenkan"] = (high9 + low9) / 2
    df["ichi_kijun"] = (high26 + low26) / 2
    df["ichi_span_a"] = ((df["ichi_tenkan"] + df["ichi_kijun"]) / 2).shift(26)
    df["ichi_span_b"] = ((high52 + low52) / 2).shift(26)
    df["ichi_chikou"] = df["c"].shift(-26)

    typical = (df["h"] + df["l"] + df["c"]) / 3
    cci_ma = typical.rolling(20).mean()
    cci_md = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df["cci"] = (typical - cci_ma) / (0.015 * cci_md.replace(0, np.nan))

    high14 = df["h"].rolling(14).max()
    low14 = df["l"].rolling(14).min()
    df["williams_r"] = ((high14 - df["c"]) / (high14 - low14).replace(0, np.nan)) * -100

    df = _compute_volume_profile(df)

    return df

def _compute_adx(df, period=14):
    if len(df) < period + 1:
        df["adx"] = np.nan
        df["di_plus"] = np.nan
        df["di_minus"] = np.nan
        return df
    high = df["h"]
    low = df["l"]
    close = df["c"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    df["di_plus"] = plus_di
    df["di_minus"] = minus_di
    df["adx"] = adx
    return df

def _compute_volume_profile(df, lookback=100, bins=None):
    bins = bins or VP_BINS
    df["vp_poc"] = np.nan
    df["vp_vah"] = np.nan
    df["vp_val"] = np.nan
    try:
        n = min(lookback, len(df))
        recent = df.tail(n)
        price_min = recent["l"].min()
        price_max = recent["h"].max()
        if price_max <= price_min:
            return df
        edges = np.linspace(price_min, price_max, bins + 1)
        bin_vol = np.zeros(bins)
        lo_arr = recent["l"].values
        hi_arr = recent["h"].values
        vol_arr = recent["v"].values
        for b in range(bins):
            b_lo = edges[b]; b_hi = edges[b + 1]
            overlap = np.maximum(0.0, np.minimum(hi_arr, b_hi) - np.maximum(lo_arr, b_lo))
            candle_range = np.maximum(hi_arr - lo_arr, 1e-9)
            bin_vol[b] = np.sum(vol_arr * (overlap / candle_range))

        poc_idx = int(np.argmax(bin_vol))
        poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0

        total_vol = bin_vol.sum()
        target_vol = total_vol * 0.70
        va_vol = bin_vol[poc_idx]
        lo_idx = poc_idx; hi_idx = poc_idx
        while va_vol < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
            add_hi = bin_vol[hi_idx + 1] if hi_idx < bins - 1 else 0
            add_lo = bin_vol[lo_idx - 1] if lo_idx > 0 else 0
            if add_hi >= add_lo and hi_idx < bins - 1:
                hi_idx += 1; va_vol += bin_vol[hi_idx]
            elif lo_idx > 0:
                lo_idx -= 1; va_vol += bin_vol[lo_idx]
            else:
                hi_idx += 1; va_vol += bin_vol[hi_idx]

        vah = (edges[hi_idx] + edges[hi_idx + 1]) / 2.0
        val = (edges[lo_idx] + edges[lo_idx + 1]) / 2.0

        df.iloc[-1, df.columns.get_loc("vp_poc")] = round(poc_price, 6)
        df.iloc[-1, df.columns.get_loc("vp_vah")] = round(vah, 6)
        df.iloc[-1, df.columns.get_loc("vp_val")] = round(val, 6)
    except Exception as e:
        logging.debug("Volume Profile error: %s", str(e)[:60])
    return df

def detect_candle_patterns(df):
    if len(df) < 3:
        return "NONE", "NEUTRAL"
    try:
        c0 = df.iloc[-1]
        c1 = df.iloc[-2]
        c2 = df.iloc[-3]

        o0, h0, l0, cl0 = float(c0["o"]), float(c0["h"]), float(c0["l"]), float(c0["c"])
        o1, h1, l1, cl1 = float(c1["o"]), float(c1["h"]), float(c1["l"]), float(c1["c"])
        o2, h2, l2, cl2 = float(c2["o"]), float(c2["h"]), float(c2["l"]), float(c2["c"])

        body0 = abs(cl0 - o0)
        body1 = abs(cl1 - o1)
        range0 = max(h0 - l0, 1e-9)
        range1 = max(h1 - l1, 1e-9)
        upper_wick0 = h0 - max(cl0, o0)
        lower_wick0 = min(cl0, o0) - l0

        bull0 = cl0 > o0
        bull1 = cl1 > o1

        if bull0 and not bull1 and cl0 > o1 and o0 < cl1:
            return "BULL ENGULFING", "BULLISH"
        if not bull0 and bull1 and cl0 < o1 and o0 > cl1:
            return "BEAR ENGULFING", "BEARISH"

        if body0 > 0 and lower_wick0 >= body0 * 2 and upper_wick0 <= body0 * 0.3:
            if not bull1:
                return "HAMMER", "BULLISH"
            else:
                return "HANGING MAN", "BEARISH"

        if body0 > 0 and upper_wick0 >= body0 * 2 and lower_wick0 <= body0 * 0.3:
            if bull1:
                return "SHOOTING STAR", "BEARISH"
            else:
                return "INV HAMMER", "BULLISH"

        if body0 / range0 < 0.08:
            return "DOJI", "NEUTRAL"

        body2 = abs(cl2 - o2)
        if (not (cl2 > o2) and body2 > range0 * 0.4
                and body1 / max(h1 - l1, 1e-9) < 0.3
                and cl0 > o0
                and cl0 > (o2 + cl2) / 2):
            return "MORNING STAR", "BULLISH"

        if ((cl2 > o2) and body2 > range0 * 0.4
                and body1 / max(h1 - l1, 1e-9) < 0.3
                and cl0 < o0
                and cl0 < (o2 + cl2) / 2):
            return "EVENING STAR", "BEARISH"
    except Exception as e:
        logging.debug("Candle pattern error: %s", str(e)[:60])
    return "NONE", "NEUTRAL"

def compute_fibonacci(swing_low, swing_high):
    diff = swing_high - swing_low
    if diff <= 0:
        return {}
    return {
        "0.0": swing_low,
        "23.6": swing_low + diff * 0.236,
        "38.2": swing_low + diff * 0.382,
        "50.0": swing_low + diff * 0.5,
        "61.8": swing_low + diff * 0.618,
        "78.6": swing_low + diff * 0.786,
        "100.0": swing_high,
        "127.2": swing_high + diff * 0.272,
        "161.8": swing_high + diff * 0.618,
        "261.8": swing_high + diff * 1.618
    }

def _detect_rsi_divergence(df):
    if len(df) < 20:
        return "NONE"
    closes = df["c"].values
    rsi_vals = df["rsi"].fillna(50).values
    n = len(closes)

    highs_idx = []
    for i in range(2, n - 1):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            highs_idx.append(i)
    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if closes[i2] > closes[i1] and rsi_vals[i2] < rsi_vals[i1]:
            return "BEARISH"

    lows_idx = []
    for i in range(2, n - 1):
        if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
            lows_idx.append(i)
    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if closes[i2] < closes[i1] and rsi_vals[i2] > rsi_vals[i1]:
            return "BULLISH"
    return "NONE"

def _detect_macd_divergence(df):
    if len(df) < 20:
        return "NONE"
    closes = df["c"].values
    macd_vals = df["macd"].fillna(0).values
    n = len(closes)

    highs_idx = []
    for i in range(2, n - 1):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            highs_idx.append(i)
    if len(highs_idx) >= 2:
        i1, i2 = highs_idx[-2], highs_idx[-1]
        if closes[i2] > closes[i1] and macd_vals[i2] < macd_vals[i1]:
            return "BEARISH"

    lows_idx = []
    for i in range(2, n - 1):
        if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
            lows_idx.append(i)
    if len(lows_idx) >= 2:
        i1, i2 = lows_idx[-2], lows_idx[-1]
        if closes[i2] < closes[i1] and macd_vals[i2] > macd_vals[i1]:
            return "BULLISH"
    return "NONE"

def _detect_market_regime(df):
    if len(df) < 50:
        return "UNKNOWN"
    adx_val = df["adx"].iloc[-1] if "adx" in df.columns else 0
    bb_bw = df["bb_bandwidth"].iloc[-1] if "bb_bandwidth" in df.columns else 0
    if pd.isna(adx_val):
        return "UNKNOWN"

    ema20 = df["ema20"].dropna()
    ema_slope = 0.0
    if len(ema20) >= 5:
        slope = (ema20.iloc[-1] - ema20.iloc[-5]) / max(ema20.iloc[-5], 1e-9) * 100
        ema_slope = slope

    atr_series = df["atr"].dropna()
    atr_ratio = 0.0
    if len(atr_series) >= 20:
        atr_now = atr_series.iloc[-1]
        atr_avg = atr_series.iloc[-20:].mean()
        atr_ratio = atr_now / max(atr_avg, 1e-9)

    if adx_val > 25 and bb_bw > 0.02 and abs(ema_slope) > 0.1:
        return "TRENDING"
    elif adx_val < 20 and bb_bw < 0.015 and atr_ratio < 1.1:
        return "RANGING"
    elif adx_val > 30 and atr_ratio > 1.3:
        return "VOLATILE_TREND"
    elif atr_ratio > 1.5:
        return "VOLATILE"
    else:
        return "MIXED"

def compute_directional_forecast(cp: float, r: AnalysisResult, ml_bias: int = 0, interval: str = "3m") -> Tuple:
    score = 0
    reasons = []
    confluence = 0

    if ml_bias == 1:
        score += 15; reasons.append("ML: Next candle BULL ▲"); confluence += 1
    elif ml_bias == -1:
        score -= 15; reasons.append("ML: Next candle BEAR ▼"); confluence += 1

    regime = r.market_regime
    adx_val = r.adx
    is_ranging = (regime == "RANGING" or adx_val < 20)
    is_trending = (regime in ("TRENDING", "VOLATILE_TREND") and adx_val >= 25)
    is_volatile = regime in ("VOLATILE", "VOLATILE_TREND")

    candle_open = r.candle_open_price
    if candle_open > 0:
        candle_change_pct = (cp - candle_open) / candle_open * 100
        if candle_change_pct < -1.0 and r.rsi < 50:
            reasons.append(f"VETO: Bear candle {candle_change_pct:.2f}%")
            if score > 0:
                score = max(score - 40, 0)
        elif candle_change_pct > 1.0 and r.rsi > 50:
            reasons.append(f"VETO: Bull candle {candle_change_pct:.2f}%")
            if score < 0:
                score = min(score + 40, 0)

    if candle_open > 0:
        candle_change_pct = (cp - candle_open) / candle_open * 100
        if candle_change_pct > 0.5:
            score += 12; reasons.append(f"Candle +{candle_change_pct:.2f}%")
        elif candle_change_pct < -0.5:
            score -= 12; reasons.append(f"Candle {candle_change_pct:.2f}%")
        elif candle_change_pct > 0.1:
            score += 5
        elif candle_change_pct < -0.1:
            score -= 5

    ema_agree = False
    ema200_valid = r.ema200 > 0 and abs(r.ema200 - cp) / cp < 0.5
    if r.ema20 > r.ema50:
        score += 20; reasons.append("EMA20 > EMA50"); ema_agree = True
    if ema200_valid and r.ema50 > r.ema200:
        score += 20; reasons.append("EMA50 > EMA200"); ema_agree = True
    if r.ema20 < r.ema50:
        score -= 20; reasons.append("EMA20 < EMA50"); ema_agree = True
    if ema200_valid and r.ema50 < r.ema200:
        score -= 20; reasons.append("EMA50 < EMA200"); ema_agree = True
    if cp > r.ema20:
        score += 10; reasons.append("Price > EMA20")
    else:
        score -= 10; reasons.append("Price < EMA20")
    if ema_agree:
        confluence += 1

    rsi_agree = False
    if r.rsi >= 65:
        score += 15; reasons.append("Strong RSI Bullish"); rsi_agree = True
    elif r.rsi <= 35:
        score -= 15; reasons.append("Strong RSI Bearish"); rsi_agree = True
    elif r.rsi > 55:
        score += 8; reasons.append("RSI Bullish"); rsi_agree = True
    elif r.rsi < 45:
        score -= 8; reasons.append("RSI Bearish"); rsi_agree = True
    if r.rsi_divergence == "BULLISH":
        score += 25; reasons.append("RSI Bull Divergence"); rsi_agree = True
    elif r.rsi_divergence == "BEARISH":
        score -= 25; reasons.append("RSI Bear Divergence"); rsi_agree = True
    if rsi_agree:
        confluence += 1

    macd_agree = False
    if r.macd_hist > 0 and r.macd > r.macd_sig:
        score += 15; reasons.append("MACD Bull Momentum"); macd_agree = True
    elif r.macd_hist < 0 and r.macd < r.macd_sig:
        score -= 15; reasons.append("MACD Bear Momentum"); macd_agree = True
    elif r.macd > r.macd_sig:
        score += 8; reasons.append("MACD Cross Up"); macd_agree = True
    elif r.macd < r.macd_sig:
        score -= 8; reasons.append("MACD Cross Down"); macd_agree = True
    if r.macd_divergence == "BULLISH":
        score += 25; reasons.append("MACD Bull Divergence"); macd_agree = True
    elif r.macd_divergence == "BEARISH":
        score -= 25; reasons.append("MACD Bear Divergence"); macd_agree = True
    if macd_agree:
        confluence += 1

    vwap_agree = False
    if cp > r.vwap:
        score += 10; reasons.append("Money Above VWAP"); vwap_agree = True
    else:
        score -= 10; reasons.append("Money Below VWAP"); vwap_agree = True
    if abs(r.vwap_dev) > 0.01:
        if r.vwap_dev > 0:
            score += 5; reasons.append("Far Above VWAP")
        else:
            score -= 5; reasons.append("Far Below VWAP")
    if vwap_agree:
        confluence += 1

    _is_scalp_tf = interval in ("1m", "3m", "5m")
    candle_w = 16 if _is_scalp_tf else 12
    if r.candle_strength > 65:
        if r.buy_pressure > r.sell_pressure:
            score += candle_w; reasons.append("Strong Bull Candle")
        else:
            score -= candle_w; reasons.append("Strong Bear Candle")

    vol_agree = False
    vol_w = 15 if _is_scalp_tf else 10
    if r.vol_ratio > 120:
        if score > 0:
            score += vol_w; reasons.append(f"Bull Volume {r.vol_ratio:.0f}%"); vol_agree = True
        else:
            score -= vol_w; reasons.append(f"Bear Volume {r.vol_ratio:.0f}%"); vol_agree = True
    if vol_agree:
        confluence += 1

    pat_agree = False
    if r.w_pattern:
        score += 20; reasons.append("W Pattern"); pat_agree = True
    if r.m_pattern:
        score -= 20; reasons.append("M Pattern"); pat_agree = True

    adv_pat_agree = False
    if r.double_bottom:
        score += 35; reasons.append("DOUBLE BOTTOM (Bull Rev)"); adv_pat_agree = True
    if r.double_top:
        score -= 35; reasons.append("DOUBLE TOP (Bear Rev)"); adv_pat_agree = True
    if r.falling_wedge:
        score += 25; reasons.append("FALLING WEDGE (Bull Rev)"); adv_pat_agree = True
    if r.rising_wedge:
        score -= 25; reasons.append("RISING WEDGE (Bear Rev)"); adv_pat_agree = True
    if adv_pat_agree:
        confluence += 2
    elif pat_agree:
        confluence += 1

    if r.price_action_bias == "UPWARD":
        score += 15; reasons.append("PA: HH+HL Upward")
    elif r.price_action_bias == "DOWNWARD":
        score -= 15; reasons.append("PA: LH+LL Downward")

    alli_agree = False
    if r.alligator_state == "UPTREND":
        score += 15; reasons.append("Alligator Bull"); alli_agree = True
    elif r.alligator_state == "DOWNTREND":
        score -= 15; reasons.append("Alligator Bear"); alli_agree = True
    if alli_agree:
        confluence += 1

    bb_agree = False
    if r.bb_bandwidth < 0.008 and not is_trending:
        score = score * 0.5; reasons.append("Tight BB (Choppy)")
    if r.bb_lower > 0 and r.bb_upper > 0:
        if cp <= r.bb_lower:
            score += 12; reasons.append("BB Oversold"); bb_agree = True
        elif cp >= r.bb_upper:
            score -= 12; reasons.append("BB Overbought"); bb_agree = True
        elif r.bb_pct < 0.2:
            score += 6; reasons.append("BB Lower Band"); bb_agree = True
        elif r.bb_pct > 0.8:
            score -= 6; reasons.append("BB Upper Band"); bb_agree = True
    if bb_agree:
        confluence += 1

    obv_agree = False
    if r.obv > 0 and r.obv_ema > 0:
        if r.obv > r.obv_ema:
            score += 8; reasons.append("OBV Bull"); obv_agree = True
        else:
            score -= 8; reasons.append("OBV Bear"); obv_agree = True
    if obv_agree:
        confluence += 1

    stoch_agree = False
    if r.stoch_rsi_k < 20 and r.stoch_rsi_d < 20:
        score += 18; reasons.append("StochRSI Oversold"); stoch_agree = True
    elif r.stoch_rsi_k > 80 and r.stoch_rsi_d > 80:
        score -= 18; reasons.append("StochRSI Overbought"); stoch_agree = True
    elif r.stoch_rsi_k > r.stoch_rsi_d and r.stoch_rsi_k < 50:
        score += 10; reasons.append("StochRSI Bullish Cross"); stoch_agree = True
    elif r.stoch_rsi_k < r.stoch_rsi_d and r.stoch_rsi_k > 50:
        score -= 10; reasons.append("StochRSI Bearish Cross"); stoch_agree = True
    if stoch_agree:
        confluence += 1

    ichi_agree = False
    if r.ichi_above_cloud and r.ichi_bullish_cloud:
        score += 20; reasons.append("Above Bull Cloud"); ichi_agree = True
    elif not r.ichi_above_cloud and not r.ichi_bullish_cloud:
        score -= 20; reasons.append("Below Bear Cloud"); ichi_agree = True
    elif r.ichi_above_cloud:
        score += 10; reasons.append("Above Cloud"); ichi_agree = True
    elif not r.ichi_above_cloud:
        score -= 10; reasons.append("Below Cloud"); ichi_agree = True
    if ichi_agree:
        confluence += 1

    cci_agree = False
    if r.cci > 150:
        score += 12; reasons.append("CCI Strong Bull"); cci_agree = True
    elif r.cci < -150:
        score -= 12; reasons.append("CCI Strong Bear"); cci_agree = True
    elif r.cci > 100:
        score += 7; reasons.append("CCI Bull"); cci_agree = True
    elif r.cci < -100:
        score -= 7; reasons.append("CCI Bear"); cci_agree = True
    if cci_agree:
        confluence += 1

    wr_agree = False
    if r.williams_r > -20:
        score -= 10; reasons.append("Williams Overbought"); wr_agree = True
    elif r.williams_r < -80:
        score += 10; reasons.append("Williams Oversold"); wr_agree = True
    if wr_agree:
        confluence += 1

    # Strategy 1: Trend-Momentum Alignment
    trend_align_score = 0
    if r.ema20 > r.ema50 and r.macd_hist > 0 and r.rsi > 50:
        trend_align_score = 18
    elif r.ema20 < r.ema50 and r.macd_hist < 0 and r.rsi < 50:
        trend_align_score = -18
    elif r.ema20 > r.ema50 and r.macd_hist > 0:
        trend_align_score = 10
    elif r.ema20 < r.ema50 and r.macd_hist < 0:
        trend_align_score = -10
    score += trend_align_score
    if trend_align_score != 0:
        confluence += 1

    # Strategy 2: Volume-Confirmed Breakout
    vol_breakout = 0
    if r.vol_ratio > 150 and cp > r.bb_upper and r.obv > r.obv_ema:
        vol_breakout = 20
    elif r.vol_ratio > 150 and cp < r.bb_lower and r.obv < r.obv_ema:
        vol_breakout = -20
    elif r.vol_ratio > 120 and r.bb_pct > 0.85:
        vol_breakout = 10
    elif r.vol_ratio > 120 and r.bb_pct < 0.15:
        vol_breakout = -10
    score += vol_breakout
    if vol_breakout != 0:
        confluence += 1

    # Strategy 3: Support/Resistance Confluence with ADX
    sr_adx_score = 0
    if r.adx >= 25:
        if r.di_plus > r.di_minus and r.price_action_bias == "UPWARD":
            sr_adx_score = 15
        elif r.di_minus > r.di_plus and r.price_action_bias == "DOWNWARD":
            sr_adx_score = -15
    elif r.adx >= 18:
        if r.di_plus > r.di_minus:
            sr_adx_score = 7
        elif r.di_minus > r.di_plus:
            sr_adx_score = -7
    score += sr_adx_score
    if sr_adx_score != 0:
        confluence += 1

    # Strategy 4: Mean Reversion
    mean_rev_score = 0
    if r.rsi <= 28 and r.vwap_dev < -0.02:
        mean_rev_score = 22
    elif r.rsi >= 72 and r.vwap_dev > 0.02:
        mean_rev_score = -22
    elif r.rsi <= 35 and cp < r.vwap:
        mean_rev_score = 10
    elif r.rsi >= 65 and cp > r.vwap:
        mean_rev_score = -10
    score += mean_rev_score
    if mean_rev_score != 0:
        confluence += 1

    # Strategy 5: Alligator + Price Action
    alli_pa_score = 0
    if r.alligator_state == "UPTREND" and r.price_action_bias == "UPWARD":
        alli_pa_score = 20
    elif r.alligator_state == "DOWNTREND" and r.price_action_bias == "DOWNWARD":
        alli_pa_score = -20
    elif r.alligator_state == "WAKING" and r.price_action_bias == "UPWARD":
        alli_pa_score = 8
    elif r.alligator_state == "WAKING" and r.price_action_bias == "DOWNWARD":
        alli_pa_score = -8
    score += alli_pa_score
    if alli_pa_score != 0:
        confluence += 1

    # Strategy 6: Divergence Amplifier
    div_amp = 0
    if r.rsi_divergence == "BULLISH" and r.macd_divergence == "BULLISH":
        div_amp = 30
    elif r.rsi_divergence == "BEARISH" and r.macd_divergence == "BEARISH":
        div_amp = -30
    elif r.rsi_divergence == "BULLISH" or r.macd_divergence == "BULLISH":
        div_amp = 15
    elif r.rsi_divergence == "BEARISH" or r.macd_divergence == "BEARISH":
        div_amp = -15
    score += div_amp
    if div_amp != 0:
        confluence += 1

    # Strategy 7: Pattern Strength Bonus
    pat_bonus = 0
    if r.double_bottom and r.w_pattern:
        pat_bonus = 20
    elif r.double_top and r.m_pattern:
        pat_bonus = -20
    elif r.falling_wedge and r.price_action_bias == "UPWARD":
        pat_bonus = 12
    elif r.rising_wedge and r.price_action_bias == "DOWNWARD":
        pat_bonus = -12
    score += pat_bonus

    # Strategy 8: Candle Pattern Signal
    candle_pat_score = 0
    if r.candle_pattern_bias == "BULLISH":
        candle_pat_score = CANDLE_PATTERN_SCORE
        reasons.append(f"Candle: {r.candle_pattern} ▲")
    elif r.candle_pattern_bias == "BEARISH":
        candle_pat_score = -CANDLE_PATTERN_SCORE
        reasons.append(f"Candle: {r.candle_pattern} ▼")
    score += candle_pat_score
    if candle_pat_score != 0:
        confluence += 1

    # Strategy 9: Volume Profile Confluence
    vp_score = 0
    if r.vp_poc > 0:
        if r.vp_bias == "BULLISH":
            vp_score = 15
            reasons.append(f"VP: Price above POC ({r.vp_poc:,.4f})")
        elif r.vp_bias == "BEARISH":
            vp_score = -15
            reasons.append(f"VP: Price below POC ({r.vp_poc:,.4f})")
        if r.vp_vah > 0 and r.vp_val > 0:
            if cp > r.vp_vah:
                vp_score += 8; reasons.append("VP: Above Value Area (institutional bull)")
            elif cp < r.vp_val:
                vp_score -= 8; reasons.append("VP: Below Value Area (institutional bear)")
    score += vp_score
    if vp_score != 0:
        confluence += 1

    # Strategy 10: Funding Rate Signal
    funding_score = 0
    if r.funding_signal == "SQUEEZE_RISK":
        funding_score = 20
        reasons.append("FUNDING: Extreme negative → Short Squeeze risk ▲")
    elif r.funding_signal == "SHORT_BIAS":
        funding_score = 10
        reasons.append("FUNDING: Negative → Crowd short → Upward pressure")
    elif r.funding_signal == "LONG_BIAS":
        funding_score = -10
        reasons.append("FUNDING: Positive → Crowd long → Downward pressure")
    score += funding_score
    if funding_score != 0:
        confluence += 1

    # TREND FILTER for Mean Reversion
    if is_trending and adx_val >= 30:
        ema_trend_bull = r.ema20 > r.ema50
        if score > 0 and not ema_trend_bull and adx_val >= 30:
            score = int(score * 0.6)
            reasons.append("TREND FILTER: Counter-trend BUY dampened")
        elif score < 0 and ema_trend_bull and adx_val >= 30:
            score = int(score * 0.6)
            reasons.append("TREND FILTER: Counter-trend SELL dampened")

    if confluence < 4 and abs(score) >= 50:
        reasons.append(f"Low Confluence ({confluence})")
        score = int(score * 0.65)

    if is_volatile and confluence < 6:
        score = int(score * 0.75)
        reasons.append("VOLATILE: Reduced signal weight")

    if is_ranging:
        threshold_mod = 1.4
    elif is_trending:
        threshold_mod = 0.85
    elif is_volatile:
        threshold_mod = 1.2
    else:
        threshold_mod = 1.0

    mod = threshold_mod
    if score >= 80 * mod:
        direction = "BUY"; strength = "VERY STRONG"
    elif score >= 60 * mod:
        direction = "BUY"; strength = "STRONG"
    elif score >= 35 * mod:
        direction = "BUY"; strength = "MODERATE"
    elif score <= -80 * mod:
        direction = "SELL"; strength = "VERY STRONG"
    elif score <= -60 * mod:
        direction = "SELL"; strength = "STRONG"
    elif score <= -35 * mod:
        direction = "SELL"; strength = "MODERATE"
    else:
        direction = "WAIT"; strength = "NEUTRAL"

    atr = max(r.atr, cp * 0.002)

    if direction == "BUY":
        sl = cp - atr * 1.8
        tp = cp + atr * 4
        entry_low = cp - atr * 0.3
        entry_high = cp + atr * 0.1
    elif direction == "SELL":
        sl = cp + atr * 1.8
        tp = cp - atr * 4
        entry_low = cp - atr * 0.1
        entry_high = cp + atr * 0.3
    else:
        sl = cp - atr
        tp = cp + atr
        entry_low = cp - atr * 0.5
        entry_high = cp + atr * 0.5

    confluence_ratio = min(confluence / 14.0, 1.0)
    score_component = min(abs(score) * 0.40, 55.0)
    confl_bonus = confluence_ratio * 25.0
    raw_confidence = score_component + confl_bonus

    if confluence < 4 and abs(score) > 50:
        raw_confidence *= 0.65
    elif confluence < 6 and abs(score) > 80:
        raw_confidence *= 0.80

    if direction != "WAIT":
        ml_dir_agrees = (direction == "BUY" and ml_bias == 1) or (direction == "SELL" and ml_bias == -1)
        ml_dir_disagrees = (direction == "BUY" and ml_bias == -1) or (direction == "SELL" and ml_bias == 1)
        if ml_dir_agrees:
            raw_confidence = min(raw_confidence * 1.08, raw_confidence + 5)
        elif ml_dir_disagrees:
            raw_confidence *= 0.88

    confidence = round(min(max(raw_confidence, 10.0), 85.0), 1)

    is_meme = _is_meme_coin()
    funding_safe, funding_msg = _check_funding_risk()
    oi_safe, oi_msg = _check_oi_risk(direction)

    if is_meme and direction in ("BUY", "SELL"):
        if confidence < 75:
            direction = "WAIT"
            strength = "NEUTRAL"
            reasons.append("MEME COIN: Confidence < 75% -- Blocked")
        if not funding_safe:
            direction = "WAIT"
            strength = "NEUTRAL"
            reasons.append(funding_msg)
        if not oi_safe:
            direction = "WAIT"
            strength = "NEUTRAL"
            reasons.append(oi_msg)
        reasons.append(f"MAX LEVERAGE: {MAX_LEVERAGE_MEME}x")
    else:
        if not funding_safe:
            reasons.append(funding_msg)
        if not oi_safe:
            reasons.append(oi_msg)

    if direction == "WAIT":
        confidence = 0.0

    reason = " | ".join(reasons)
    return direction, strength, round(entry_low, 6), round(entry_high, 6), round(tp, 6), round(sl, 6), round(confidence, 1), reason, confluence

def _deep_analyze(df: pd.DataFrame, cp: float) -> AnalysisResult:
    if len(df) < 50 or cp <= 0:
        return AnalysisResult()

    df_ind = compute_indicators(df.copy())
    last = df_ind.iloc[-1]

    result = AnalysisResult()
    result.ema20 = round(last["ema20"], 6) if pd.notna(last.get("ema20")) else 0.0
    result.ema50 = round(last["ema50"], 6) if pd.notna(last.get("ema50")) else 0.0
    result.ema200 = round(last["ema200"], 6) if pd.notna(last.get("ema200")) else 0.0
    result.vwap = round(last["vwap"], 6) if pd.notna(last.get("vwap")) else 0.0
    result.macd_hist = round(last["macd_hist"], 6) if pd.notna(last.get("macd_hist")) else 0.0
    result.bb_upper = round(last["bb_upper"], 6) if pd.notna(last.get("bb_upper")) else 0.0
    result.bb_lower = round(last["bb_lower"], 6) if pd.notna(last.get("bb_lower")) else 0.0
    result.bb_mid = round(last["bb_mid"], 6) if pd.notna(last.get("bb_mid")) else 0.0
    result.bb_pct = round(last["bb_pct"], 4) if pd.notna(last.get("bb_pct")) else 0.5
    result.bb_bandwidth = round(last["bb_bandwidth"], 4) if pd.notna(last.get("bb_bandwidth")) else 0.0
    result.obv = round(last["obv"], 2) if pd.notna(last.get("obv")) else 0.0
    result.obv_ema = round(last["obv_ema"], 2) if pd.notna(last.get("obv_ema")) else 0.0
    result.adx = round(last["adx"], 2) if pd.notna(last.get("adx")) else 0.0
    result.di_plus = round(last["di_plus"], 2) if pd.notna(last.get("di_plus")) else 0.0
    result.di_minus = round(last["di_minus"], 2) if pd.notna(last.get("di_minus")) else 0.0
    result.gator_val = round(last["gator"], 6) if pd.notna(last.get("gator")) else 0.0
    result.vwap_dev = round(last["vwap_dev"], 4) if pd.notna(last.get("vwap_dev")) else 0.0
    result.rsi = round(last["rsi"], 2) if pd.notna(last.get("rsi")) else 50.0
    result.macd = round(last["macd"], 6) if pd.notna(last.get("macd")) else 0.0
    result.macd_sig = round(last["macd_sig"], 6) if pd.notna(last.get("macd_sig")) else 0.0
    vol_avg = last["vol_avg20"]
    result.vol_ratio = round(last["vol_ratio"], 1) if pd.notna(vol_avg) and vol_avg > 0 else 0.0
    result.jaw = round(last["jaw_cur"], 6) if pd.notna(last.get("jaw_cur")) else 0.0
    result.teeth = round(last["teeth_cur"], 6) if pd.notna(last.get("teeth_cur")) else 0.0
    result.lips = round(last["lips_cur"], 6) if pd.notna(last.get("lips_cur")) else 0.0

    candle_body = abs(last["c"] - last["o"])
    candle_range = max(last["h"] - last["l"], 0.000001)
    result.candle_strength = round((candle_body / candle_range) * 100, 1)
    result.candle_open_price = float(last["o"]) if last["o"] > 0 else 0.0

    if last["c"] > last["o"]:
        result.buy_pressure = result.candle_strength
    else:
        result.sell_pressure = result.candle_strength

    result.m_pattern, _ = detect_m_pattern(df_ind)
    result.w_pattern, _ = detect_w_pattern(df_ind)
    result.double_top, _dt = detect_double_top(df_ind)
    result.double_bottom, _db = detect_double_bottom(df_ind)
    result.rising_wedge, _rw = detect_rising_wedge(df_ind)
    result.falling_wedge, _fw = detect_falling_wedge(df_ind)

    pat_parts = []
    if result.double_top: pat_parts.append("DOUBLE TOP ▼")
    if result.double_bottom: pat_parts.append("DOUBLE BOTTOM ▲")
    if result.m_pattern: pat_parts.append("M-PATTERN ▼")
    if result.w_pattern: pat_parts.append("W-PATTERN ▲")
    if result.rising_wedge: pat_parts.append("RISING WEDGE ▼")
    if result.falling_wedge: pat_parts.append("FALLING WEDGE ▲")
    result.pattern_label = " | ".join(pat_parts) if pat_parts else "No Pattern"
    result.price_action_bias = compute_price_action_bias(df_ind, cp)

    jaw_shifted = round(last["jaw"], 6) if pd.notna(last.get("jaw")) else 0.0
    teeth_shifted = round(last["teeth"], 6) if pd.notna(last.get("teeth")) else 0.0
    lips_shifted = round(last["lips"], 6) if pd.notna(last.get("lips")) else 0.0
    result.alligator_state, _, _ = analyze_alligator_state(jaw_shifted, teeth_shifted, lips_shifted)

    result.atr = round(last["atr"], 6) if pd.notna(last.get("atr")) else 0.0
    result.swing_low = last["swing_low_20"] if pd.notna(last.get("swing_low_20")) else last["swing_low_10"]
    result.swing_high = last["swing_high_20"] if pd.notna(last.get("swing_high_20")) else last["swing_high_10"]
    result.fib_levels = compute_fibonacci(result.swing_low, result.swing_high)

    result.rsi_divergence = _detect_rsi_divergence(df_ind)
    result.macd_divergence = _detect_macd_divergence(df_ind)
    result.market_regime = _detect_market_regime(df_ind)

    result.stoch_rsi_k = round(float(last.get("stoch_rsi_k", 50)), 2) if pd.notna(last.get("stoch_rsi_k")) else 50.0
    result.stoch_rsi_d = round(float(last.get("stoch_rsi_d", 50)), 2) if pd.notna(last.get("stoch_rsi_d")) else 50.0
    result.cci = round(float(last.get("cci", 0)), 2) if pd.notna(last.get("cci")) else 0.0
    result.williams_r = round(float(last.get("williams_r", -50)), 2) if pd.notna(last.get("williams_r")) else -50.0

    span_a = last.get("ichi_span_a", 0)
    span_b = last.get("ichi_span_b", 0)
    if pd.notna(span_a) and pd.notna(span_b) and span_a != 0 and span_b != 0:
        cloud_top = max(float(span_a), float(span_b))
        cloud_bot = min(float(span_a), float(span_b))
        result.ichi_above_cloud = cp > cloud_top
        result.ichi_bullish_cloud = float(span_a) > float(span_b)
    else:
        result.ichi_above_cloud = False
        result.ichi_bullish_cloud = False

    # ML Prediction
    ml_bias = 0
    if SKLEARN_OK and len(df_ind) >= 30:
        try:
            n_candles_now = len(df_ind)
            with st.session_state.ml_lock:
                need_retrain = (
                    st.session_state.ml_clf is None or
                    st.session_state.ml_scaler is None or
                    (n_candles_now - st.session_state.ml_candles_at_train) >= ML_RETRAIN_EVERY_N_CANDLES
                )

            if need_retrain:
                n_train = min(len(df_ind) - 1, 150)
                df_ml = df_ind.iloc[-(n_train + 1):-1].copy().reset_index(drop=True)

                def _ml_features(row_df, i):
                    r = row_df.iloc[i]
                    prev = row_df.iloc[i - 1] if i > 0 else r
                    prev2 = row_df.iloc[i - 2] if i > 1 else prev
                    atr_ref = max(float(r.get("atr", cp * 0.002)), cp * 0.0001)
                    c_cur = float(r["c"])
                    c_prev = float(prev["c"])
                    c_p2 = float(prev2["c"])
                    mom2 = (c_cur - c_p2) / atr_ref
                    body = (c_cur - float(r["o"])) / atr_ref
                    wick_up = (float(r["h"]) - max(c_cur, float(r["o"]))) / atr_ref
                    wick_dn = (min(c_cur, float(r["o"])) - float(r["l"])) / atr_ref
                    return [
                        float(r.get("rsi", 50)) / 100.0,
                        float(r.get("macd_hist", 0)) / atr_ref,
                        float(r.get("bb_pct", 0.5)),
                        float(r.get("vwap_dev", 0)),
                        float(r.get("vol_ratio", 100)) / 200.0,
                        float(r.get("adx", 0)) / 100.0,
                        (float(r.get("ema20", cp)) - float(r.get("ema50", cp))) / atr_ref,
                        float(r.get("stoch_rsi_k", 50)) / 100.0,
                        float(r.get("cci", 0)) / 200.0,
                        (c_cur - c_prev) / atr_ref,
                        mom2,
                        body,
                        wick_up,
                        wick_dn,
                        float(r.get("obv", 0)) / max(abs(float(r.get("obv_ema", 1))), 1),
                    ]

                X_list, y_list = [], []
                for i in range(2, len(df_ml) - 3):
                    try:
                        feats = _ml_features(df_ml, i)
                        if any(np.isnan(f) or np.isinf(f) for f in feats):
                            continue
                        atr_i = max(float(df_ml.iloc[i].get("atr", cp * 0.002)), cp * 0.0001)
                        entry = float(df_ml.iloc[i]["c"])
                        tp_lvl = entry + atr_i * 1.5
                        sl_lvl = entry - atr_i * 1.0
                        label = 0
                        for fw in range(1, 4):
                            if i + fw >= len(df_ml):
                                break
                            fw_h = float(df_ml.iloc[i + fw]["h"])
                            fw_l = float(df_ml.iloc[i + fw]["l"])
                            if fw_h >= tp_lvl:
                                label = 1; break
                            if fw_l <= sl_lvl:
                                label = 0; break
                        X_list.append(feats)
                        y_list.append(label)
                    except Exception:
                        continue

                if len(X_list) >= ML_MIN_TRAIN_SAMPLES:
                    X_arr = np.array(X_list, dtype=np.float32)
                    y_arr = np.array(y_list, dtype=np.int32)

                    def _bg_train(X=X_arr, y=y_arr, nc=n_candles_now):
                        try:
                            new_scaler = StandardScaler()
                            X_scaled = new_scaler.fit_transform(X)
                            new_clf = GradientBoostingClassifier(
                                n_estimators=80, max_depth=3,
                                learning_rate=0.08, subsample=0.8,
                                min_samples_leaf=3,
                                random_state=42
                            )
                            new_clf.fit(X_scaled, y)
                            with st.session_state.ml_lock:
                                st.session_state.ml_clf = new_clf
                                st.session_state.ml_scaler = new_scaler
                                st.session_state.ml_candles_at_train = nc
                        except Exception as _e:
                            logging.debug("BG ML train error: %s", _e)

                    threading.Thread(target=_bg_train, daemon=True).start()

            with st.session_state.ml_lock:
                clf_ready = st.session_state.ml_clf is not None and st.session_state.ml_scaler is not None

            if clf_ready:
                n_total = len(df_ind)
                atr_ref_cur = max(float(df_ind["atr"].iloc[-1]) if pd.notna(df_ind["atr"].iloc[-1]) else cp * 0.002, cp * 0.0001)
                prev_c = float(df_ind["c"].iloc[-2]) if n_total > 1 else float(df_ind["c"].iloc[-1])
                prev2_c = float(df_ind["c"].iloc[-3]) if n_total > 2 else prev_c
                c_cur = float(df_ind["c"].iloc[-1])
                o_cur = float(df_ind["o"].iloc[-1])
                h_cur = float(df_ind["h"].iloc[-1])
                l_cur = float(df_ind["l"].iloc[-1])
                mom2 = (c_cur - prev2_c) / atr_ref_cur
                body = (c_cur - o_cur) / atr_ref_cur
                wick_up = (h_cur - max(c_cur, o_cur)) / atr_ref_cur
                wick_dn = (min(c_cur, o_cur) - l_cur) / atr_ref_cur
                obv_cur = float(df_ind["obv"].iloc[-1]) if "obv" in df_ind.columns and pd.notna(df_ind["obv"].iloc[-1]) else 0.0
                obv_ema_cur = float(df_ind["obv_ema"].iloc[-1]) if "obv_ema" in df_ind.columns and pd.notna(df_ind["obv_ema"].iloc[-1]) else 1.0

                def _safe(col, default):
                    v = df_ind[col].iloc[-1] if col in df_ind.columns else default
                    return float(v) if pd.notna(v) else default

                cur_feats = [
                    _safe("rsi", 50) / 100.0,
                    _safe("macd_hist", 0) / atr_ref_cur,
                    _safe("bb_pct", 0.5),
                    _safe("vwap_dev", 0),
                    _safe("vol_ratio", 100) / 200.0,
                    _safe("adx", 0) / 100.0,
                    (_safe("ema20", cp) - _safe("ema50", cp)) / atr_ref_cur,
                    _safe("stoch_rsi_k", 50) / 100.0,
                    _safe("cci", 0) / 200.0,
                    (c_cur - prev_c) / atr_ref_cur,
                    mom2,
                    body,
                    wick_up,
                    wick_dn,
                    obv_cur / max(abs(obv_ema_cur), 1),
                ]
                if not any(np.isnan(f) or np.isinf(f) for f in cur_feats):
                    with st.session_state.ml_lock:
                        cur_scaled = st.session_state.ml_scaler.transform([cur_feats])
                        prob = st.session_state.ml_clf.predict_proba(cur_scaled)[0]
                    p_up = prob[1]
                    if p_up > 0.60:
                        ml_bias = 1
                    elif p_up < 0.40:
                        ml_bias = -1
                    else:
                        ml_bias = 0
                    with st.session_state.ml_lock:
                        st.session_state.ml_last_bias = ml_bias
                        st.session_state.ml_last_prob = p_up
                    st.session_state.ml_status = (
                        f"ML: {'▲ BULL' if ml_bias==1 else '▼ BEAR' if ml_bias==-1 else 'FLAT'}"
                        f" ({p_up*100:.0f}%)"
                    )
                else:
                    st.session_state.ml_status = "ML: SKIP (NaN)"
                    with st.session_state.ml_lock:
                        ml_bias = st.session_state.ml_last_bias
            else:
                st.session_state.ml_status = "ML: Training..."
                ml_bias = 0
        except Exception as e:
            logging.debug("ML prediction error: %s", str(e)[:80])
            ml_bias = 0
            st.session_state.ml_status = "ML: ERR"

    # Volume Profile
    vp_poc = float(last.get("vp_poc", 0)) if pd.notna(last.get("vp_poc")) else 0.0
    vp_vah = float(last.get("vp_vah", 0)) if pd.notna(last.get("vp_vah")) else 0.0
    vp_val = float(last.get("vp_val", 0)) if pd.notna(last.get("vp_val")) else 0.0
    result.vp_poc = round(vp_poc, 6)
    result.vp_vah = round(vp_vah, 6)
    result.vp_val = round(vp_val, 6)
    if vp_poc > 0 and cp > 0:
        if cp > vp_vah and vp_vah > 0:
            result.vp_bias = "BULLISH"
        elif cp < vp_val and vp_val > 0:
            result.vp_bias = "BEARISH"
        elif cp > vp_poc:
            result.vp_bias = "BULLISH"
        else:
            result.vp_bias = "BEARISH"

    # Candle Pattern
    result.candle_pattern, result.candle_pattern_bias = detect_candle_patterns(df_ind)

    # Funding Rate Signal
    fr = st.session_state.funding_rate
    if fr <= FUNDING_DANGER:
        result.funding_signal = "SQUEEZE_RISK"
    elif fr <= FUNDING_WARNING:
        result.funding_signal = "SHORT_BIAS"
    elif fr >= 0.01:
        result.funding_signal = "LONG_BIAS"
    else:
        result.funding_signal = "NEUTRAL"

    # Compute forecast
    direction, strength, entry_low, entry_high, target, stop, confidence, reason, confluence =         compute_directional_forecast(cp, result, ml_bias, st.session_state.interval)

    result.direction = direction
    result.strength = strength
    result.entry_low = entry_low
    result.entry_high = entry_high
    result.tp = target
    result.sl = stop
    result.confidence = confidence
    result.reason = reason
    result.confluence = confluence

    if direction == "BUY":
        result.score = round(confidence * 2, 1)
    elif direction == "SELL":
        result.score = round(-confidence * 2, 1)
    else:
        result.score = 0.0

    # Smart SL/TP
    atr_val = max(result.atr, cp * 0.002)
    sr = st.session_state.intel_sr_levels
    if direction == "BUY":
        supports = [lvl["price"] for lvl in sr if lvl["type"] == "S" and lvl["price"] < cp]
        if result.vp_val > 0 and result.vp_val < cp:
            supports.append(result.vp_val)
        if supports:
            nearest_support = max(supports)
            result.smart_sl = round(nearest_support - atr_val * 0.3, 6)
        else:
            result.smart_sl = round(cp - atr_val * 1.8, 6)
        resistances = [lvl["price"] for lvl in sr if lvl["type"] == "R" and lvl["price"] > cp]
        if result.vp_vah > cp and result.vp_vah > 0:
            resistances.append(result.vp_vah)
        result.smart_tp = round(min(resistances), 6) if resistances else round(cp + atr_val * 3.5, 6)
    elif direction == "SELL":
        resistances = [lvl["price"] for lvl in sr if lvl["type"] == "R" and lvl["price"] > cp]
        if result.vp_vah > cp and result.vp_vah > 0:
            resistances.append(result.vp_vah)
        if resistances:
            nearest_res = min(resistances)
            result.smart_sl = round(nearest_res + atr_val * 0.3, 6)
        else:
            result.smart_sl = round(cp + atr_val * 1.8, 6)
        supports = [lvl["price"] for lvl in sr if lvl["type"] == "S" and lvl["price"] < cp]
        if result.vp_val > 0 and result.vp_val < cp:
            supports.append(result.vp_val)
        result.smart_tp = round(max(supports), 6) if supports else round(cp - atr_val * 3.5, 6)
    else:
        result.smart_sl = result.sl
        result.smart_tp = result.tp

    # Position Sizing
    risk_pct = 0.01
    sl_distance = abs(cp - result.smart_sl) if result.smart_sl > 0 else atr_val * 1.8
    sl_distance = max(sl_distance, cp * 0.001)

    with st.session_state.bt_lock:
        bt_total = st.session_state.bt_wins + st.session_state.bt_losses
        live_wr = (st.session_state.bt_wins / bt_total) if bt_total >= 5 else None

    if sl_distance > 0 and cp > 0:
        rr_ratio = abs((result.smart_tp - cp) / sl_distance) if result.smart_tp != cp else 2.0
        rr_ratio = max(rr_ratio, 0.5)
        raw_pos = (risk_pct / (sl_distance / cp)) * 100
        conf_multiplier = 0.6 + (confidence / 100) * 0.8

        if live_wr is not None:
            kelly_f = max(0, (live_wr * rr_ratio - (1 - live_wr)) / rr_ratio)
            kelly_fraction = min(kelly_f * 0.5, 0.04)
            result.position_size_pct = round(min(raw_pos * conf_multiplier, kelly_fraction * 100, 5.0), 2)
        else:
            result.position_size_pct = round(min(raw_pos * conf_multiplier, 5.0), 2)
    else:
        result.position_size_pct = 1.0

    if result.market_regime in ("VOLATILE", "RANGING"):
        result.position_size_pct = round(result.position_size_pct * 0.7, 2)

    # Win Rate Estimate
    if live_wr is not None and bt_total >= 10:
        result.win_rate_est = round(live_wr * 100, 1)
    elif live_wr is not None and bt_total >= 5:
        prior = 47.0 + min(confluence, 8) * 1.5
        result.win_rate_est = round(live_wr * 100 * 0.8 + prior * 0.2, 1)
    else:
        if confluence >= 12:
            result.win_rate_est = round(min(52 + (confluence - 12) * 1.5, 62), 1)
        elif confluence >= 8:
            result.win_rate_est = round(48 + (confluence - 8) * 1.0, 1)
        elif confluence >= 5:
            result.win_rate_est = round(44 + (confluence - 5) * 1.3, 1)
        else:
            result.win_rate_est = round(38 + confluence * 1.2, 1)

    risk = abs(cp - stop)
    reward = abs(target - cp)
    result.rr = round(reward / risk, 1) if risk > 0 else 0.0
    return result

# --- Data Fetching ---
def load_historical_klines():
    try:
        limit = _get_history_limit()
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={st.session_state.symbol}&interval={st.session_state.interval}&limit={limit}"
        r = _api_get(url, timeout=10)
        if r.status_code != 200:
            set_error(f"REST {r.status_code}: {st.session_state.symbol}")
            return
        klines = r.json()
        if not klines or len(klines) < 10:
            set_error(f"REST: Not enough data ({len(klines) if klines else 0} candles)")
            return
        df = _klines_to_df(klines)
        with st.session_state.df_lock:
            st.session_state.candle_deque.clear()
            for rec in df[["t","o","h","l","c","v"]].itertuples(index=False):
                st.session_state.candle_deque.append(Candle(t=rec.t, o=rec.o, h=rec.h, l=rec.l, c=rec.c, v=rec.v))
        last_close = float(df.iloc[-1]["c"])
        last_open = float(df.iloc[-1]["o"])
        if last_open > 0:
            st.session_state.candle_open_price = last_open
        if st.session_state.price == 0 and last_close > 0:
            st.session_state.price = last_close
            st.session_state.mark_price = last_close
            st.session_state.last_price_update = time.time()
        st.session_state.new_data_event.set()
        logging.info("Loaded %d candles for %s %s", len(df), st.session_state.symbol, st.session_state.interval)
    except Exception as e:
        set_error("REST: " + str(e)[:60])

def fetch_funding():
    try:
        if not st.session_state.symbol:
            return
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={st.session_state.symbol}&limit=1"
        r = _api_get(url, timeout=4)
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                st.session_state.funding_rate = float(data[0].get("fundingRate", 0))
                st.session_state.funding_next = int(data[0].get("fundingTime", 0)) // 1000
                st.session_state._last_funding_update = time.time()
    except Exception as e:
        logging.debug("Funding fetch error: %s", str(e)[:60])

def fetch_oi():
    try:
        if not st.session_state.symbol:
            return
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={st.session_state.symbol}"
        r = _api_get(url, timeout=4)
        if r.status_code == 200:
            data = r.json()
            oi = float(data.get("openInterest", 0))
            if st.session_state.oi_prev > 0 and oi > 0:
                st.session_state.oi_change_pct = ((oi - st.session_state.oi_prev) / st.session_state.oi_prev) * 100
            else:
                st.session_state.oi_change_pct = 0.0
            st.session_state.oi_prev = st.session_state.oi_value
            st.session_state.oi_value = oi
            st.session_state._last_oi_update = time.time()
    except Exception as e:
        logging.debug("OI fetch error: %s", str(e)[:60])

def fetch_instant_price():
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={st.session_state.symbol}"
        r = _api_get(url, timeout=2)
        if r.status_code == 200:
            data = r.json()
            p = float(data["markPrice"])
            st.session_state.price = p
            st.session_state.mark_price = p
            st.session_state.prev_price = p
            st.session_state.last_price_update = time.time()
    except Exception as e:
        set_error("Instant price: " + str(e)[:50])

def load_symbols():
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        r = _api_get(url, timeout=10)
        info = r.json()
        all_syms = [
            s for s in info["symbols"]
            if s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ]

        ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        tr = _api_get(ticker_url, timeout=10)
        tickers = {t["symbol"]: t for t in tr.json()}

        sym_data = []
        for s in all_syms:
            sym = s["symbol"]
            base = s["baseAsset"]
            t = tickers.get(sym, {})
            change = float(t.get("priceChangePercent", 0.0))
            sym_data.append((base, sym, change))

        sym_data.sort(key=lambda x: x[2], reverse=True)
        st.session_state.initial_coin_order = [x[0] for x in sym_data]
        return st.session_state.initial_coin_order
    except Exception as e:
        set_error(str(e))
        return []

# --- Intel Functions ---
def _compute_sr_levels(df: pd.DataFrame) -> list:
    highs = df["h"].values
    lows = df["l"].values
    closes = df["c"].values
    vols = df["v"].values
    n = len(df)
    pivots = []
    window = 5
    for i in range(window, n - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            pivots.append((highs[i], "R", i, vols[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            pivots.append((lows[i], "S", i, vols[i]))
    if not pivots:
        return []
    cp = st.session_state.mark_price if st.session_state.mark_price > 0 else closes[-1]
    tol = cp * 0.0025
    clusters = []
    used = [False] * len(pivots)
    for i, (p1, t1, idx1, v1) in enumerate(pivots):
        if used[i]:
            continue
        group_prices = [p1]
        group_vols = [v1]
        group_types = [t1]
        used[i] = True
        for j, (p2, t2, idx2, v2) in enumerate(pivots):
            if used[j]:
                continue
            if abs(p1 - p2) <= tol:
                group_prices.append(p2)
                group_vols.append(v2)
                group_types.append(t2)
                used[j] = True
        avg_price = float(np.mean(group_prices))
        total_vol = float(np.sum(group_vols))
        touches = len(group_prices)
        n_r = group_types.count("R")
        n_s = group_types.count("S")
        sr_type = "R" if n_r >= n_s else "S"
        clusters.append({
            "price": round(avg_price, 6),
            "type": sr_type,
            "touches": touches,
            "vol_score": round(total_vol / 1000, 1),
            "strength": "STRONG" if touches >= 3 else "MODERATE" if touches == 2 else "WEAK"
        })
    clusters.sort(key=lambda x: (x["touches"], x["vol_score"]), reverse=True)
    top = clusters[:10]
    top.sort(key=lambda x: x["price"])
    return top

def _compute_trend_500(df: pd.DataFrame) -> dict:
    closes = df["c"].values
    n = len(closes)
    ema20 = float(pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1])
    ema100 = float(pd.Series(closes).ewm(span=100, adjust=False).mean().iloc[-1])
    ema200 = float(pd.Series(closes).ewm(span=200, adjust=False).mean().iloc[-1])
    cp = closes[-1]
    if ema20 > ema50 > ema100 > ema200:
        ema_stack = "Full Bull Stack UP"; ema_score = 3
    elif ema20 < ema50 < ema100 < ema200:
        ema_stack = "Full Bear Stack DOWN"; ema_score = -3
    elif ema20 > ema50 and cp > ema200:
        ema_stack = "Partial Bull"; ema_score = 1
    elif ema20 < ema50 and cp < ema200:
        ema_stack = "Partial Bear"; ema_score = -1
    else:
        ema_stack = "Mixed / Ranging"; ema_score = 0

    swing_highs = []
    swing_lows = []
    h = df["h"].values
    l = df["l"].values
    for i in range(3, n - 3):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append(h[i])
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append(l[i])
    hh = hl = lh = ll = 0
    for i in range(1, min(len(swing_highs), 6)):
        if swing_highs[i] > swing_highs[i-1]: hh += 1
        else: lh += 1
    for i in range(1, min(len(swing_lows), 6)):
        if swing_lows[i] > swing_lows[i-1]: hl += 1
        else: ll += 1
    if hh >= 2 and hl >= 2:
        structure = f"HH+HL (Bullish) | +{hh}/{hl}"; struct_score = 2
    elif lh >= 2 and ll >= 2:
        structure = f"LH+LL (Bearish) | -{lh}/{ll}"; struct_score = -2
    else:
        structure = "Mixed Structure"; struct_score = 0

    try:
        x = np.arange(100).reshape(-1, 1)
        y = closes[-100:]
        reg = LinearRegression().fit(x, y)
        slope_pct = round(reg.coef_[0] / closes[-101] * 100, 4)
        slope_str = f"{slope_pct:+.4f}% per bar"
    except Exception:
        slope_pct = 0.0
        slope_str = "N/A (sklearn missing)"

    total_score = ema_score + struct_score + (1 if slope_pct > 0 else -1 if slope_pct < 0 else 0)
    if total_score >= 3:
        direction = "UPTREND"; strength = "STRONG"
    elif total_score >= 1:
        direction = "UPTREND"; strength = "MODERATE"
    elif total_score <= -3:
        direction = "DOWNTREND"; strength = "STRONG"
    elif total_score <= -1:
        direction = "DOWNTREND"; strength = "MODERATE"
    else:
        direction = "SIDEWAYS"; strength = "NEUTRAL"

    return {
        "direction": direction,
        "strength": strength,
        "ema_stack": ema_stack,
        "structure": structure,
        "slope": slope_str,
    }

def _compute_liquidity(df: pd.DataFrame) -> dict:
    highs = df["h"].values
    lows = df["l"].values
    closes = df["c"].values
    n = len(df)
    cp = closes[-1]
    buy_liq_levels = []
    sell_liq_levels = []
    for i in range(5, min(n, 100)):
        idx = n - 1 - i
        if lows[idx] == min(lows[max(0, idx-4):idx+5]):
            buy_liq_levels.append(lows[idx])
        if highs[idx] == max(highs[max(0, idx-4):idx+5]):
            sell_liq_levels.append(highs[idx])
    buy_liq = round(float(np.mean(buy_liq_levels[-5:])), 6) if buy_liq_levels else 0.0
    sell_liq = round(float(np.mean(sell_liq_levels[-5:])), 6) if sell_liq_levels else 0.0
    n_buy = len(buy_liq_levels)
    n_sell = len(sell_liq_levels)
    if n_buy > n_sell * 1.2:
        dominant = "BUY LIQUIDITY HEAVY"
    elif n_sell > n_buy * 1.2:
        dominant = "SELL LIQUIDITY HEAVY"
    else:
        dominant = "BALANCED"
    bl_sorted = sorted(buy_liq_levels, key=lambda x: abs(x - cp))[:3]
    sl_sorted = sorted(sell_liq_levels, key=lambda x: abs(x - cp))[:3]
    buy_str = "Buy: " + " | ".join([f"{v:,.4f}" for v in sorted(bl_sorted)])
    sell_str = "Sell: " + " | ".join([f"{v:,.4f}" for v in sorted(sl_sorted)])
    zones = f"{buy_str}\n{sell_str}"
    recent_low = min(lows[-5:])
    recent_high = max(highs[-5:])
    sweep_alert = "None detected"
    for bl in buy_liq_levels[-10:]:
        if recent_low < bl < cp:
            sweep_alert = f"BUY SWEEP at {bl:,.4f} -- Potential bounce UP"
            break
    for sl in sell_liq_levels[-10:]:
        if cp < sl < recent_high:
            sweep_alert = f"SELL SWEEP at {sl:,.4f} -- Potential drop DOWN"
            break
    return {
        "buy_liq": buy_liq,
        "sell_liq": sell_liq,
        "dominant": dominant,
        "zones": zones,
        "sweep_alert": sweep_alert,
    }

def refresh_intel():
    try:
        sym = st.session_state.symbol
        limit = max(_get_history_limit(), 300)
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={st.session_state.interval}&limit={limit}"
        r = _api_get(url, timeout=10)
        klines = r.json()
        if not klines or len(klines) < 50:
            return
        df = _klines_to_df(klines)
        sr_levels = _compute_sr_levels(df)
        trend_data = _compute_trend_500(df)
        liq_data = _compute_liquidity(df)
        with st.session_state.intel_lock:
            st.session_state.intel_sr_levels = sr_levels
            st.session_state.intel_trend500 = trend_data["direction"]
            st.session_state.intel_trend_str = trend_data["strength"]
            st.session_state.intel_ema_stack = trend_data["ema_stack"]
            st.session_state.intel_structure = trend_data["structure"]
            st.session_state.intel_slope = trend_data["slope"]
            st.session_state.intel_liq_buy = liq_data["buy_liq"]
            st.session_state.intel_liq_sell = liq_data["sell_liq"]
            st.session_state.intel_liq_dom = liq_data["dominant"]
            st.session_state.intel_liq_zones = liq_data["zones"]
            st.session_state.intel_liq_sweep = liq_data["sweep_alert"]
            st.session_state.intel_last_refresh = time.time()
    except Exception as e:
        logging.debug("Intel refresh error: %s", str(e)[:60])


# --- MTF Functions ---
def _mtf_fetch_one(tf, symbol, results, lock):
    try:
        if tf == st.session_state.interval:
            df = _deque_to_df()
            if len(df) >= 50:
                cp = st.session_state.mark_price if st.session_state.mark_price > 0 else float(df["c"].iloc[-1])
                result = _deep_analyze(df, cp)
                with lock:
                    results[tf] = (result.direction, result.strength, result.confidence, result.score, result.reason)
                return
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={tf}&limit={TF_HISTORY_MAP.get(tf, 150)}"
        r = st.session_state._session.get(url, timeout=8)
        klines = r.json()
        if not klines or len(klines) < 50:
            with lock:
                results[tf] = ("WAIT", "No data", 0, 0, "")
            return
        df = _klines_to_df(klines)
        cp = float(df["c"].iloc[-1])
        result = _deep_analyze(df, cp)
        with lock:
            results[tf] = (result.direction, result.strength, result.confidence, result.score, result.reason)
    except Exception as e:
        with lock:
            results[tf] = ("WAIT", "Error", 0, 0, str(e)[:40])

def refresh_mtf():
    results = {}
    lock = threading.Lock()
    tfs = ["15m", "1h", "4h", "1d", "1w"]
    symbol = st.session_state.symbol
    threads = [
        threading.Thread(target=_mtf_fetch_one, args=(tf, symbol, results, lock), daemon=True)
        for tf in tfs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=12)
    for tf in tfs:
        if tf not in results:
            results[tf] = ("WAIT", "Timeout", 0, 0, "")
    st.session_state.mtf_cached_results = results

    directions = [results[tf][0] for tf in tfs]
    buy_count = directions.count("BUY")
    sell_count = directions.count("SELL")
    agree_count = max(buy_count, sell_count)

    is_meme = _is_meme_coin()
    min_required = MEME_MIN_MTF if is_meme else STANDARD_MIN_MTF

    if buy_count >= min_required:
        master = "BUY"
    elif sell_count >= min_required:
        master = "SELL"
    else:
        master = "WAIT"

    st.session_state.mtf_master = master
    st.session_state.mtf_agree = agree_count
    return results

def refresh_scalp_mtf():
    scalp_tfs = ["1m", "3m", "5m", "15m", "30m"]
    symbol = st.session_state.symbol
    results = {}
    lock = threading.Lock()

    threads = [
        threading.Thread(target=_mtf_fetch_one, args=(tf, symbol, results, lock), daemon=True)
        for tf in scalp_tfs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=12)
    for tf in scalp_tfs:
        if tf not in results:
            results[tf] = ("WAIT", "Timeout", 0, 0, "")

    directions = [results[tf][0] for tf in scalp_tfs]
    buy_count = directions.count("BUY")
    sell_count = directions.count("SELL")
    agree_count = max(buy_count, sell_count)

    if buy_count >= 3:
        master = "BUY"
    elif sell_count >= 3:
        master = "SELL"
    else:
        master = "WAIT"

    return results, master, agree_count

# --- Signal Processing ---
def process_signal(result: AnalysisResult, cp: float):
    new_dir = result.direction

    with st.session_state.bt_lock:
        if st.session_state.bt_open is not None:
            o = st.session_state.bt_open
            hit_tp = (o["dir"] == "BUY" and cp >= o["tp"]) or (o["dir"] == "SELL" and cp <= o["tp"])
            hit_sl = (o["dir"] == "BUY" and cp <= o["sl"]) or (o["dir"] == "SELL" and cp >= o["sl"])
            if hit_tp:
                st.session_state.bt_wins += 1
                st.session_state.bt_signals.append({**o, "result": "WIN", "exit": round(cp, 6)})
                st.session_state.bt_open = None
            elif hit_sl:
                st.session_state.bt_losses += 1
                st.session_state.bt_signals.append({**o, "result": "LOSS", "exit": round(cp, 6)})
                st.session_state.bt_open = None

    if new_dir != st.session_state.last_signal_direction and new_dir in ("BUY", "SELL"):
        st.session_state.last_signal_direction = new_dir
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": st.session_state.symbol,
            "tf": st.session_state.interval,
            "direction": new_dir,
            "strength": result.strength,
            "confidence": round(result.confidence, 1),
            "price": round(cp, 6),
            "sl": round(result.smart_sl if result.smart_sl > 0 else result.sl, 6),
            "tp": round(result.smart_tp if result.smart_tp > 0 else result.tp, 6),
            "rr": round(result.rr, 1),
            "pos_size": round(result.position_size_pct, 2),
            "win_est": round(result.win_rate_est, 1),
        }
        st.session_state.signal_history.appendleft(entry)
        st.session_state.signal_stats[new_dir.lower()] += 1
        with st.session_state.bt_lock:
            st.session_state.bt_open = {
                "dir": new_dir,
                "entry": round(cp, 6),
                "tp": entry["tp"],
                "sl": entry["sl"],
                "time": entry["time"],
            }
    elif new_dir == "WAIT":
        st.session_state.last_signal_direction = "WAIT"

# --- Analysis Loop ---
def run_analysis():
    df = _deque_to_df()
    if len(df) < 30:
        return
    cp = st.session_state.mark_price if st.session_state.mark_price > 0 else df["c"].iloc[-1]
    if cp <= 0:
        return
    if st.session_state.price == 0 and cp > 0:
        st.session_state.price = cp
    result = _deep_analyze(df, cp)
    with st.session_state.analysis_lock:
        st.session_state.current_analysis = result
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    process_signal(result, cp)

# --- Background Data Update ---
def background_update():
    import time
    while st.session_state.running:
        try:
            fetch_instant_price()
            fetch_funding()
            fetch_oi()
            run_analysis()
            time.sleep(2)
        except Exception as e:
            logging.debug("Background update error: %s", str(e)[:60])
            time.sleep(2)


# --- UI Components ---
def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="main-header">📡 PRO AI RADAR</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">v' + VERSION + ' | FUTURES LIVE | SCALPING MODE | 15-FEATURE AI | KELLY SIZING</div>', unsafe_allow_html=True)

def render_controls():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Load coins if not already loaded
        if not st.session_state.initial_coin_order:
            try:
                with st.spinner("Loading coins from Binance..."):
                    coins = load_symbols()
            except Exception as e:
                st.warning(f"Could not load coins: {e}")
                coins = []
        else:
            coins = st.session_state.initial_coin_order

        # Build coin options
        coin_options = []
        for base in coins:
            if base in st.session_state.mtf_coin_data:
                d = st.session_state.mtf_coin_data[base]
                if d['direction'] == "WAIT" or d['agree'] == 0:
                    display = f"{base:6s}  [--/-- WAIT   0%]"
                else:
                    display = f"{base:6s}  [{d['agree']}/5 {d['direction']:4s} {d['conf']:5.1f}%]"
            else:
                display = f"{base:6s}  [--/-- WAIT   0%]"
            coin_options.append(display)

        # Fallback if no coins loaded
        if not coin_options:
            coin_options = ["BTC   [--/-- WAIT   0%]"]

        selected_coin = st.selectbox("🪙 COIN", coin_options, index=0, key="coin_selector")

        # Manual symbol entry fallback
        manual_symbol = st.text_input("Or enter symbol:", value="", key="manual_symbol", 
                                       placeholder="e.g. ETHUSDT")

        # Safely extract base from selected coin
        try:
            base = selected_coin.split()[0].strip() if selected_coin else "BTC"
        except:
            base = "BTC"

        new_symbol = base.upper() + "USDT"

        # Check if manual symbol entered
        if manual_symbol and manual_symbol.upper() != st.session_state.symbol:
            new_symbol = manual_symbol.upper()
            if not new_symbol.endswith("USDT"):
                new_symbol += "USDT"
            base = new_symbol.replace("USDT", "")

        if new_symbol != st.session_state.symbol:
            st.session_state.symbol = new_symbol
            st.session_state.current_coin_base = base
            st.session_state.candle_deque.clear()
            st.session_state.price_history.clear()
            try:
                load_historical_klines()
            except Exception as e:
                st.error(f"Failed to load data: {e}")
            st.rerun()

    with col2:
        tf_options = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]
        selected_tf = st.selectbox("⏱️ TF", tf_options, index=tf_options.index(st.session_state.interval), key="tf_selector")
        if selected_tf != st.session_state.interval:
            st.session_state.interval = selected_tf
            st.session_state.candle_deque.clear()
            load_historical_klines()
            st.rerun()

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            load_historical_klines()
            refresh_intel()
            st.rerun()

def render_price_card():
    cp = st.session_state.mark_price if st.session_state.mark_price > 0 else st.session_state.price
    change = 0.0
    open_price = st.session_state.candle_open_price
    if open_price > 0 and cp > 0:
        change = ((cp - open_price) / open_price) * 100

    change_color = "#02c076" if change >= 0 else "#cf304a"
    change_txt = f"{change:+.2f}%"

    fund_color = "#02c076" if st.session_state.funding_rate > 0 else "#cf304a" if st.session_state.funding_rate < FUNDING_WARNING else "#888"
    fund_text = f"Funding: {st.session_state.funding_rate*100:+.4f}%"
    if st.session_state.funding_next > 0:
        from datetime import datetime
        next_fund = datetime.fromtimestamp(st.session_state.funding_next).strftime("%H:%M")
        fund_text += f" (Next: {next_fund})"

    oi_color = "#02c076" if st.session_state.oi_change_pct > 0 else "#cf304a" if st.session_state.oi_change_pct < -OI_DROP_PCT else "#888"
    oi_text = f"OI: {st.session_state.oi_value/1e6:.2f}M ({st.session_state.oi_change_pct:+.1f}%)"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        html = '<div class="price-display">'
        html += '<div style="font-size:0.8rem; color:#888;">MARK PRICE (LIVE)</div>'
        html += f'<div>{cp:,.6f} $</div>'
        html += f'<div style="font-size:1rem; color:{change_color};">{change_txt}</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="color:{fund_color}; font-size:0.85rem; text-align:center;">{fund_text}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div style="color:{oi_color}; font-size:0.85rem; text-align:center;">{oi_text}</div>', unsafe_allow_html=True)

    if _is_meme_coin():
        max_lev = _get_max_leverage()
        min_mtf = _get_min_mtf()
        st.markdown(f'<div class="warning-box">⚠️ MEME COIN -- Max {max_lev}x | Need {min_mtf}/5 MTF</div>', unsafe_allow_html=True)

def render_market_tab():
    with st.session_state.analysis_lock:
        r = st.session_state.current_analysis
    cp = st.session_state.mark_price if st.session_state.mark_price > 0 else st.session_state.price

    signal_color = "signal-buy" if r.direction == "BUY" else "signal-sell" if r.direction == "SELL" else "signal-wait"
    signal_text = "▲ BUY" if r.direction == "BUY" else "▼ SELL" if r.direction == "SELL" else "WAIT"
    if r.strength in ["STRONG", "VERY STRONG", "MODERATE"]:
        signal_text += f" ({r.strength})"

    col1, col2, col3 = st.columns(3)
    with col1:
        html = f'<div class="metric-card"><div class="metric-label">DIRECTION</div><div class="{signal_color}">{signal_text}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        conf_color = "#02c076" if r.confidence >= 70 else "#f0b90b" if r.confidence >= 40 else "#cf304a"
        html = f'<div class="metric-card"><div class="metric-label">CONFIDENCE</div><div class="metric-value" style="color:{conf_color};">{r.confidence:.0f}%</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col3:
        html = f'<div class="metric-card"><div class="metric-label">SCORE</div><div class="metric-value">{r.score:+.1f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        html = f'<div class="metric-card"><div class="metric-label">ENTRY ZONE</div><div class="metric-value" style="color:#00d2ff;">{r.entry_low:,.6f} - {r.entry_high:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        html = f'<div class="metric-card"><div class="metric-label">TAKE PROFIT</div><div class="metric-value" style="color:#02c076;">{r.tp:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col3:
        html = f'<div class="metric-card"><div class="metric-label">STOP LOSS</div><div class="metric-value" style="color:#cf304a;">{r.sl:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        rr_text = f"1:{r.rr:.1f}" if r.rr > 0 else "--"
        html = f'<div class="metric-card"><div class="metric-label">RISK:REWARD</div><div class="metric-value" style="color:#f0b90b;">{rr_text}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        smart_sl = r.smart_sl if r.smart_sl > 0 else r.sl
        html = f'<div class="metric-card"><div class="metric-label">SMART SL</div><div class="metric-value" style="color:#ff6b6b;">{smart_sl:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col3:
        smart_tp = r.smart_tp if r.smart_tp > 0 else r.tp
        html = f'<div class="metric-card"><div class="metric-label">SMART TP</div><div class="metric-value" style="color:#00ff88;">{smart_tp:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pos_color = "#02c076" if r.position_size_pct >= 2 else "#f0b90b" if r.position_size_pct >= 1 else "#888"
        html = f'<div class="metric-card"><div class="metric-label">POSITION SIZE</div><div class="metric-value" style="color:{pos_color};">{r.position_size_pct:.2f}% balance</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        wr_color = "#02c076" if r.win_rate_est >= 60 else "#f0b90b" if r.win_rate_est >= 50 else "#cf304a"
        html = f'<div class="metric-card"><div class="metric-label">EST WIN RATE</div><div class="metric-value" style="color:{wr_color};">{r.win_rate_est:.1f}%</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 MARKET PULSE")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RSI", f"{r.rsi:.1f}")
    with col2:
        st.metric("MACD", f"{r.macd_hist:+.4f}")
    with col3:
        st.metric("ADX", f"{r.adx:.1f}")
    with col4:
        st.metric("Volume", f"{r.vol_ratio:.0f}%")

    st.markdown("---")
    st.markdown(f"#### 📜 SIGNAL HISTORY (BUY: {st.session_state.signal_stats['buy']} | SELL: {st.session_state.signal_stats['sell']})")

    if st.session_state.signal_history:
        hist_df = pd.DataFrame(list(st.session_state.signal_history)[:10])
        hist_df['direction'] = hist_df['direction'].apply(lambda x: f"🟢 {x}" if x == "BUY" else f"🔴 {x}")
        st.dataframe(hist_df[['time', 'symbol', 'tf', 'direction', 'strength', 'confidence', 'price', 'rr']], 
                    use_container_width=True, hide_index=True)
    else:
        st.info("No signals yet -- waiting for next signal...")

    if st.button("🗑️ Clear Signal History"):
        st.session_state.signal_history.clear()
        st.session_state.signal_stats = {"buy": 0, "sell": 0, "wait": 0}
        st.rerun()


def render_intel_tab():
    st.markdown("#### 🧠 SMART INTEL")

    col1, col2 = st.columns(2)
    with col1:
        trend_color = "#02c076" if st.session_state.intel_trend500 == "UPTREND" else "#cf304a" if st.session_state.intel_trend500 == "DOWNTREND" else "#888"
        html = f'<div class="metric-card"><div class="metric-label">TREND (500 CANDLES)</div><div class="metric-value" style="color:{trend_color};">{st.session_state.intel_trend500} ({st.session_state.intel_trend_str})</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div class="metric-card"><div class="metric-label">EMA STACK</div><div class="metric-value">{st.session_state.intel_ema_stack}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div class="metric-card"><div class="metric-label">MARKET STRUCTURE</div><div class="metric-value">{st.session_state.intel_structure}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div class="metric-card"><div class="metric-label">TREND SLOPE</div><div class="metric-value">{st.session_state.intel_slope}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📍 SUPPORT / RESISTANCE")
        sr_data = st.session_state.intel_sr_levels
        if sr_data:
            sr_df = pd.DataFrame(sr_data)
            st.dataframe(sr_df, use_container_width=True, hide_index=True)
        else:
            st.info("No S/R levels computed yet -- click Refresh Intel")

    st.markdown("---")
    st.markdown("#### 💧 LIQUIDITY MAP")
    col1, col2 = st.columns(2)
    with col1:
        dom_color = "#02c076" if "BUY" in st.session_state.intel_liq_dom else "#cf304a" if "SELL" in st.session_state.intel_liq_dom else "#888"
        html = f'<div class="metric-card"><div class="metric-label">DOMINANT LIQUIDITY</div><div class="metric-value" style="color:{dom_color};">{st.session_state.intel_liq_dom}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div class="metric-card"><div class="metric-label">BUY LIQ LEVEL</div><div class="metric-value">{st.session_state.intel_liq_buy:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

        html = f'<div class="metric-card"><div class="metric-label">SELL LIQ LEVEL</div><div class="metric-value">{st.session_state.intel_liq_sell:,.6f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    with col2:
        st.markdown("**Liquidity Zones:**")
        zones = st.session_state.intel_liq_zones
        st.text(zones)
        st.text(zones)

        sweep_color = "#02c076" if "BOUNCE" in st.session_state.intel_liq_sweep else "#cf304a" if "DROP" in st.session_state.intel_liq_sweep else "#888"
        html = f'<div class="metric-card"><div class="metric-label">SWEEP ALERT</div><div class="metric-value" style="color:{sweep_color};">{st.session_state.intel_liq_sweep}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    if st.button("🔄 Refresh Intel"):
        refresh_intel()
        st.rerun()

def render_mtf_tab():
    st.markdown("#### 🕐 MULTI-TIMEFRAME ANALYSIS")

    if st.button("🔄 Refresh MTF"):
        refresh_mtf()
        st.rerun()

    results = st.session_state.mtf_cached_results
    if not results:
        st.info("Click Refresh MTF to load data")
        return

    master = st.session_state.mtf_master
    agree = st.session_state.mtf_agree

    master_color = "#02c076" if master == "BUY" else "#cf304a" if master == "SELL" else "#888"
    st.markdown(f'<div class="metric-card"><div class="metric-label">MTF MASTER SIGNAL</div><div class="metric-value" style="color:{master_color}; font-size:1.5rem;">{master} ({agree}/5)</div></div>', unsafe_allow_html=True)

    cols = st.columns(len(results))
    for i, (tf, (direction, strength, confidence, score, reason)) in enumerate(results.items()):
        with cols[i]:
            dir_color = "#02c076" if direction == "BUY" else "#cf304a" if direction == "SELL" else "#888"
            st.markdown(f"**{tf}**")
            st.markdown(f'<div style="color:{dir_color}; font-weight:bold; font-size:1.2rem;">{direction}</div>', unsafe_allow_html=True)
            st.markdown(f"Strength: {strength}")
            st.markdown(f"Confidence: {confidence:.0f}%")
            st.markdown(f"Score: {score:+.0f}")
            if reason:
                st.caption(reason[:80])

def render_indicators_tab():
    with st.session_state.analysis_lock:
        r = st.session_state.current_analysis

    st.markdown("#### 📈 INDICATORS")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Moving Averages**")
        st.write(f"EMA20: {r.ema20:,.6f}")
        st.write(f"EMA50: {r.ema50:,.6f}")
        st.write(f"EMA200: {r.ema200:,.6f}")
        st.write(f"VWAP: {r.vwap:,.6f}")
    with col2:
        st.markdown("**Oscillators**")
        st.write(f"RSI: {r.rsi:.1f}")
        st.write(f"StochRSI K: {r.stoch_rsi_k:.1f}")
        st.write(f"StochRSI D: {r.stoch_rsi_d:.1f}")
        st.write(f"CCI: {r.cci:.1f}")
        st.write(f"Williams %R: {r.williams_r:.1f}")
    with col3:
        st.markdown("**Trend**")
        st.write(f"ADX: {r.adx:.1f}")
        st.write(f"DI+: {r.di_plus:.1f}")
        st.write(f"DI-: {r.di_minus:.1f}")
        st.write(f"ATR: {r.atr:.6f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bollinger Bands**")
        st.write(f"Upper: {r.bb_upper:,.6f}")
        st.write(f"Mid: {r.bb_mid:,.6f}")
        st.write(f"Lower: {r.bb_lower:,.6f}")
        st.write(f"Bandwidth: {r.bb_bandwidth:.4f}")
    with col2:
        st.markdown("**Volume**")
        st.write(f"Volume Ratio: {r.vol_ratio:.0f}%")
        st.write(f"OBV: {r.obv:,.0f}")
        st.write(f"OBV EMA: {r.obv_ema:,.0f}")

    st.markdown("---")
    st.markdown("**Alligator**")
    st.write(f"State: {r.alligator_state}")
    st.write(f"Jaw: {r.jaw:,.6f}")
    st.write(f"Teeth: {r.teeth:,.6f}")
    st.write(f"Lips: {r.lips:,.6f}")

    st.markdown("---")
    st.markdown("**Patterns**")
    st.write(f"M Pattern: {r.m_pattern}")
    st.write(f"W Pattern: {r.w_pattern}")
    st.write(f"Double Top: {r.double_top}")
    st.write(f"Double Bottom: {r.double_bottom}")
    st.write(f"Rising Wedge: {r.rising_wedge}")
    st.write(f"Falling Wedge: {r.falling_wedge}")
    st.write(f"Candle Pattern: {r.candle_pattern} ({r.candle_pattern_bias})")

    st.markdown("---")
    st.markdown("**Divergences**")
    st.write(f"RSI Divergence: {r.rsi_divergence}")
    st.write(f"MACD Divergence: {r.macd_divergence}")
    st.write(f"Market Regime: {r.market_regime}")

def render_scalp_tab():
    st.markdown("#### ⚡ SCALP MODE")

    if st.button("🔄 Refresh Scalp MTF"):
        results, master, agree = refresh_scalp_mtf()
        st.session_state.mtf_coin_data[st.session_state.current_coin_base] = {
            "direction": master, "agree": agree, "conf": 0
        }
        st.rerun()

    results, master, agree = refresh_scalp_mtf()

    master_color = "#02c076" if master == "BUY" else "#cf304a" if master == "SELL" else "#888"
    st.markdown(f'<div class="metric-card"><div class="metric-label">SCALP MTF MASTER</div><div class="metric-value" style="color:{master_color}; font-size:1.5rem;">{master} ({agree}/5)</div></div>', unsafe_allow_html=True)

    cols = st.columns(len(results))
    for i, (tf, (direction, strength, confidence, score, reason)) in enumerate(results.items()):
        with cols[i]:
            dir_color = "#02c076" if direction == "BUY" else "#cf304a" if direction == "SELL" else "#888"
            st.markdown(f"**{tf}**")
            st.markdown(f'<div style="color:{dir_color}; font-weight:bold; font-size:1.2rem;">{direction}</div>', unsafe_allow_html=True)
            st.markdown(f"Strength: {strength}")
            st.markdown(f"Confidence: {confidence:.0f}%")
            st.markdown(f"Score: {score:+.0f}")

def render_bt_tab():
    st.markdown("#### 📊 BACKTEST")

    with st.session_state.bt_lock:
        wins = st.session_state.bt_wins
        losses = st.session_state.bt_losses
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Wins", wins)
    with col2:
        st.metric("Losses", losses)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")

    if st.session_state.bt_signals:
        bt_df = pd.DataFrame(st.session_state.bt_signals)
        st.dataframe(bt_df, use_container_width=True, hide_index=True)
    else:
        st.info("No backtest data yet")

    if st.button("🗑️ Clear Backtest"):
        with st.session_state.bt_lock:
            st.session_state.bt_wins = 0
            st.session_state.bt_losses = 0
            st.session_state.bt_signals = []
            st.session_state.bt_open = None
        st.rerun()

def render_status_bar():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div style='font-size:0.75rem; color:#888;'>Last Update: {st.session_state.last_update}</div>", unsafe_allow_html=True)
    with col2:
        ws_color = "#02c076" if st.session_state.ws_connected else "#cf304a"
        st.markdown(f"<div style='font-size:0.75rem; color:{ws_color};'>{st.session_state.ws_status}</div>", unsafe_allow_html=True)
    with col3:
        ml_color = "#02c076" if "ACTIVE" in st.session_state.ml_status else "#f0b90b" if "Training" in st.session_state.ml_status else "#888"
        st.markdown(f"<div style='font-size:0.75rem; color:{ml_color};'>{st.session_state.ml_status}</div>", unsafe_allow_html=True)
    with col4:
        if st.session_state.error_msg:
            st.markdown(f"<div style='font-size:0.75rem; color:#cf304a;'>⚠️ {st.session_state.error_msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:0.75rem; color:#02c076;'>✅ System OK</div>", unsafe_allow_html=True)

# --- Main App ---
def main():
    render_header()
    render_controls()
    render_price_card()

    tabs = st.tabs(["📊 MARKET", "🧠 INTEL", "🕐 MTF", "📈 INDICATORS", "⚡ SCALP", "📊 BACKTEST"])

    with tabs[0]:
        render_market_tab()
    with tabs[1]:
        render_intel_tab()
    with tabs[2]:
        render_mtf_tab()
    with tabs[3]:
        render_indicators_tab()
    with tabs[4]:
        render_scalp_tab()
    with tabs[5]:
        render_bt_tab()

    render_status_bar()

    # Auto-refresh using st.rerun() with a timer
    # Note: In Streamlit, we use st.empty() and update it, or use st.rerun() with sleep
    # For live updates, we'll use a placeholder that updates

    # Start background thread only once
    if not st.session_state.threads_started:
        st.session_state.threads_started = True
        # Note: In Streamlit, threads should NOT call Streamlit APIs
        # The background_update function only fetches data, no UI calls
        bg_thread = threading.Thread(target=background_update, daemon=True)
        bg_thread.start()

    # Auto refresh using Streamlit's native auto-refresh
    # This is safer than st.rerun() in a loop
    st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)

    # Manual refresh button
    if st.button("🔄 Auto Refresh (5s)", key="auto_refresh"):
        st.rerun()

if __name__ == "__main__":
    main()
