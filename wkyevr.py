import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import os
import urllib.request
from datetime import datetime, timedelta

# ==========================================
# 0. Streamlit 設定與字型處理
# ==========================================
st.set_page_config(page_title="威科夫波段-EvR分析", layout="wide")

@st.cache_resource
def get_chinese_font():
    # 使用 "粉圓體 (Open Huninn)"
    font_url = "https://github.com/justfont/open-huninn-font/releases/download/v2.0/jf-openhuninn-2.0.ttf"
    font_path = "jf-openhuninn-2.0.ttf"
    
    if not os.path.exists(font_path):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        with st.spinner("正在下載中文字型 (粉圓體)..."):
            try:
                urllib.request.urlretrieve(font_url, font_path)
            except Exception as e:
                st.error(f"字型下載失敗：{e}")
                return None
                
    return fm.FontProperties(fname=font_path)

my_font = get_chinese_font()

SYMBOL_MAP = {
    "🇹🇼 台股權值": {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電"},
    "🇺🇸 美股指數": {"NQ=F": "那斯達克期貨", "ES=F": "S&P500期貨", "^SOX": "費半指數"},
    "🇺🇸 美股巨頭": {"NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "AMD": "超微"},
    "🪙 加密貨幣": {"BTC-USD": "比特幣", "ETH-USD": "以太幣"},
    "🌍 商品外匯": {"GC=F": "黃金", "DX-Y.NYB": "美元指數"}
}

# ==========================================
# 1. 核心計算模組
# ==========================================

def calculate_indicators(df):
    df['SMA200'] = df['Close'].rolling(window=200).mean() # 年線
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # EvR 能量指標
    vol = df['Volume'].replace(0, 1) 
    raw_force = (df['Close'] - df['Open']) * vol
    evr_ema = raw_force.ewm(span=14, adjust=False).mean()
    evr_std = evr_ema.rolling(100).std().replace(0, np.nan).ffill()
    df['EvR'] = (evr_ema / evr_std) * 10
    return df

# 大盤環境判斷
def get_market_sentiment():
    try:
        end = datetime.now()
        start = end - timedelta(days=400)
        df = yf.download("^TWII", start=start, end=end, interval="1d", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = [c[0] for c in df.columns]
        
        if len(df) < 200: return None
        
        df = calculate_indicators(df)
        last = df.iloc[-1]
        
        close = last['Close'] if np.isscalar(last['Close']) else last['Close'].iloc[0]
        sma200 = last['SMA200'] if np.isscalar(last['SMA200']) else last['SMA200'].iloc[0]
        evr = last['EvR'] if np.isscalar(last['EvR']) else last['EvR'].iloc[0]
        
        trend = "多頭 (Bull)" if close > sma200 else "空頭 (Bear)"
        
        return {
            "price": close,
            "sma200": sma200,
            "evr": evr,
            "trend": trend
        }
    except:
        return None

def analyze_classic_wyckoff(df):
    signals = []
    struct_low = np.nan; struct_high = np.nan; sc_date = None
    position = 'None'; stop_price = 0
    
    df['Wyckoff_Support'] = np.nan
    df['Wyckoff_Resistance'] = np.nan
    
    for i in range(100, len(df)):
        row = df.iloc[i]; date = df.index[i]
        
        close_p = row['Close'] if np.isscalar(row['Close']) else row['Close'].iloc[0]
        low_p = row['Low'] if np.isscalar(row['Low']) else row['Low'].iloc[0]
        high_p = row['High'] if np.isscalar(row['High']) else row['High'].iloc[0]
        open_p = row['Open'] if np.isscalar(row['Open']) else row['Open'].iloc[0]
        vol = row['Volume'] if np.isscalar(row['Volume']) else row['Volume'].iloc[0]
        vol_ma = row['Vol_MA20'] if np.isscalar(row['Vol_MA20']) else row['Vol_MA20'].iloc[0]
        
        vol_cond = vol > (vol_ma * 1.5)
        
        # 1. 更新結構
        lowest_20 = df['Low'].iloc[i-20:i].min()
        if low_p < lowest_20 and vol_cond and close_p > low_p:
            struct_low = low_p; sc_date = date
            if position == 'None': signals.append({'Date': date, 'Type': 'SC', 'Price': low_p, 'Note': 'SC 支撐'})

        highest_20 = df['High'].iloc[i-20:i].max()
        if high_p > highest_20 and vol_cond:
            struct_high = high_p
            if position == 'None': signals.append({'Date': date, 'Type': 'BC', 'Price': high_p, 'Note': 'BC 壓力'})

        df.at[date, 'Wyckoff_Support'] = struct_low
        df.at[date, 'Wyckoff_Resistance'] = struct_high
        
        # 2. 進出場
        if position == 'None':
            if np.isnan(struct_low): continue
            is_after_sc = (sc_date is not None and date > sc_date)
            penetrate = low_p < struct_low
            recover = close_p > struct_low
            dist = (low_p - struct_low) / struct_low
            near_support = 0 <= dist <= 0.015
            low_vol = vol < (vol_ma * 0.8)
            bullish = close_p > open_p

            if is_after_sc and penetrate and recover:
                position = 'Long'; stop_price = low_p * 0.99 
                signals.append({'Date': date, 'Type': 'Spring', 'Price': close_p, 'Note': 'Spring買進', 'Stop': stop_price})
            elif is_after_sc and near_support and low_vol and bullish:
                position = 'Long'; stop_price = struct_low * 0.98 
                signals.append({'Date': date, 'Type': 'Test', 'Price': close_p, 'Note': 'Test買進', 'Stop': stop_price})

        elif position == 'Long':
            if close_p < stop_price:
                position = 'None'; signals.append({'Date': date, 'Type': 'Exit_SL', 'Price': close_p, 'Note': '停損出場'})
            elif (not np.isnan(struct_high) and high_p >= struct_high) or (high_p > highest_20 and vol_cond):
                position = 'None'; signals.append({'Date': date, 'Type': 'Exit_TP', 'Price': close_p, 'Note': 'BC/壓力停利'})

    return pd.DataFrame(signals), df

def analyze_evr_trend(df, window=60):
    signals = []
    position = 'None'; stop_price = 0
    last_low_p = np.inf; last_low_e = np.inf
    last_high_p = -np.inf; last_high_e = -np.inf
    start_idx = max(window+5, 200)
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]; date = df.index[i]
        
        close_p = row['Close'] if np.isscalar(row['Close']) else row['Close'].iloc[0]
        low_p = row['Low'] if np.isscalar(row['Low']) else row['Low'].iloc[0]
        high_p = row['High'] if np.isscalar(row['High']) else row['High'].iloc[0]
        evr = row['EvR'] if np.isscalar(row['EvR']) else row['EvR'].iloc[0]
        sma200 = row['SMA200'] if np.isscalar(row['SMA200']) else row['SMA200'].iloc[0]
        prev_evr = df.iloc[i-1]['EvR'] if np.isscalar(df.iloc[i-1]['EvR']) else df.iloc[i-1]['EvR'].iloc[0]
        
        local_low = df['Low'].iloc[i-window:i+1].min()
        if low_p == local_low:
            is_buy_div = (local_low < last_low_p and evr > last_low_e)
            last_low_p = local_low; last_low_e = evr
        else: is_buy_div = False
            
        local_high = df['High'].iloc[i-window:i+1].max()
        if high_p == local_high:
            is_sell_div = (local_high > last_high_p and evr < last_high_e)
            last_high_p = local_high; last_high_e = evr
        else: is_sell_div = False
        
        signal_type = None; note = ""
        if position == 'None':
            if close_p > sma200 and is_buy_div and evr > prev_evr:
                signal_type = 'Long'; position = 'Long'; stop_price = low_p*0.98; note = f"順勢Spring"
            elif close_p < sma200 and is_sell_div and evr < prev_evr:
                signal_type = 'Short'; position = 'Short'; stop_price = high_p*1.02; note = f"順勢UT"
        elif position == 'Long':
            if close_p < stop_price: signal_type = 'Exit_SL'; position = 'None'; note = "停損"
            elif is_sell_div: signal_type = 'Exit_TP'; position = 'None'; note = "頂背離停利"
            elif close_p < sma200: signal_type = 'Exit_Trend'; position = 'None'; note = "破年線平倉"
        elif position == 'Short':
            if close_p > stop_price: signal_type = 'Exit_SL'; position = 'None'; note = "停損"
            elif is_buy_div: signal_type = 'Exit_TP'; position = 'None'; note = "底背離回補"
            elif close_p > sma200: signal_type = 'Exit_Trend'; position = 'None'; note = "過年線回補"

        if signal_type:
            signals.append({'Date': date, 'Type': signal_type, 'Price': close_p, 'Note': note, 'Stop': stop_price})
    return pd.DataFrame(signals)

# ==========================================
# 2. 繪圖模組
# ==========================================

def plot_chart(df, ticker, signals, mode_name):
    plt.close('all')
    subset = df.iloc[-250:].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # K線
    width = 0.6
    up = subset[subset.Close >= subset.Open]
    down = subset[subset.Close < subset.Open]
    ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='#ef5350', edgecolor='#ef5350')
    ax1.bar(up.index, up.High - up.Close, 0.1, bottom=up.Close, color='#ef5350')
    ax1.bar(up.index, up.Low - up.Open, 0.1, bottom=up.Open, color='#ef5350')
    ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='#26a69a', edgecolor='#26a69a')
    ax1.bar(down.index, down.High - down.Open, 0.1, bottom=down.Open, color='#26a69a')
    ax1.bar(down.index, down.Low - down.Close, 0.1, bottom=down.Close, color='#26a69a')

    # 策略繪圖
    if 'Classic Wyckoff' in mode_name:
        if 'Wyckoff_Support' in subset.columns:
            ax1.plot(subset.index, subset['Wyckoff_Support'], color='purple', linewidth=1.5, label='支撐線 (SC)')
        if 'Wyckoff_Resistance' in subset.columns:
            ax1.plot(subset.index, subset['Wyckoff_Resistance'], color='orange', linewidth=1.5, linestyle='--', label='壓力線 (BC)')
        ax2.bar(subset.index, subset['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('成交量', fontproperties=my_font)
    else: 
        ax1.plot(subset.index, subset['SMA200'], color='blue', linestyle='--', label='SMA200')
        ax2.plot(subset.index, subset['EvR'], color='#7e57c2', label='EvR')
        ax2.axhline(0, color='black')
        ax2.set_ylabel('EvR 能量', fontproperties=my_font)

    # 訊號標示
    if not signals.empty:
        mask = (signals['Date'] >= subset.index[0]) & (signals['Date'] <= subset.index[-1])
        valid_signals = signals[mask]
        for _, row in valid_signals.iterrows():
            d = row['Date']; p = row['Price']; t = row['Type']
            if t in ['Spring', 'Long', 'Test']:
                ax1.scatter(d, p*0.98, marker='^', color='purple', s=100, zorder=10)
                ax1.annotate(f"{row['Note']}", (d, p*0.97), xytext=(0,-20), textcoords='offset points', ha='center', color='purple', fontsize=9, fontproperties=my_font)
            elif t in ['Short']:
                ax1.scatter(d, p*1.02, marker='v', color='blue', s=100, zorder=10)
                ax1.annotate(f"{row['Note']}", (d, p*1.03), xytext=(0,15), textcoords='offset points', ha='center', color='blue', fontsize=9, fontproperties=my_font)
            elif 'Exit' in t:
                ax1.scatter(d, p, marker='x', color='black', s=80, zorder=15)

    ax1.set_title(f"[{ticker}] {mode_name}", fontsize=16, fontproperties=my_font)
    ax1.legend(loc='upper left', prop=my_font)
    
    # 日期軸設定
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def get_todays_action(df, signals):
    last_date = df.index[-1]
    today_signal = None
    if not signals.empty:
        last_sig = signals.iloc[-1]
        if last_sig['Date'] == last_date:
            today_signal = last_sig

    action = "【無動作 (WAIT)】"
    reason = "目前無明確訊號，持有者續抱，空手者觀望。"
    
    if today_signal is not None:
        t = today_signal['Type']; note = today_signal['Note']
        stop = today_signal.get('Stop', 0)
        
        if t in ['Spring', 'Test', 'Long']:
            action = "🟣【買進/做多 (BUY)】"
            reason = f"觸發 {note}，建議停損設 {stop:.2f}"
        elif t in ['Short']:
            action = "🔵【放空 (SHORT)】"
            reason = f"觸發 {note}，建議停損設 {stop:.2f}"
        elif 'Exit_TP' in t:
            action = "🌟【停利出場 (Take Profit)】"
            reason = f"觸發 {note}，獲利了結。"
        elif 'Exit' in t:
            action = "❌【出場 (Exit)】"
            reason = f"觸發 {note}。"
            
    return action, reason

# ==========================================
# 4. Streamlit 主程式介面
# ==========================================

st.sidebar.title("📊 威科夫 x EvR 戰情室")

with st.sidebar.expander("🌍 大盤環境 (^TWII)", expanded=True):
    with st.spinner("載入大盤數據..."):
        mkt = get_market_sentiment()
        if mkt:
            st.metric("加權指數", f"{mkt['price']:.0f}", delta=f"距離年線 {(mkt['price'] - mkt['sma200']):.0f}")
            if mkt['trend'] == "多頭 (Bull)":
                st.success(f"目前趨勢：{mkt['trend']}")
            else:
                st.error(f"目前趨勢：{mkt['trend']}")
            st.write(f"EvR 能量: {mkt['evr']:.2f}")
        else:
            st.warning("大盤數據讀取失敗")

st.sidebar.markdown("---")

category = st.sidebar.selectbox("選擇分類", list(SYMBOL_MAP.keys()))
symbol_name_map = SYMBOL_MAP[category]
selected_name = st.sidebar.selectbox("選擇標的", list(symbol_name_map.values()))
target_symbol = [k for k, v in symbol_name_map.items() if v == selected_name][0]

manual_symbol = st.sidebar.text_input("或手動輸入代號 (如 2330.TW)", value="")
# 🌟 修正點：如果有手動輸入，強制更新標題名稱
if manual_symbol:
    target_symbol = manual_symbol.upper()
    selected_name = target_symbol 

st.sidebar.markdown("---")
strategy = st.sidebar.radio("選擇策略", ["A. 威科夫波段 (區間)", "B. EvR 順勢 (趨勢)"])

evr_window = 60
if strategy.startswith("B"):
    evr_window = st.sidebar.slider("EvR 背離天數視窗", 20, 100, 60)

run_btn = st.sidebar.button("🚀 開始分析", type="primary")

if run_btn:
    st.title(f"📈 {selected_name} 分析報告") # 修正顯示格式
    
    with st.spinner(f"正在分析 {target_symbol}..."):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=600) 
            
            df = yf.download(target_symbol, start=start_date, end=end_date, interval="1d", progress=False)
            
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = [c[0] for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()]
            
            if len(df) < 150:
                st.error("❌ 數據不足，無法計算指標 (K棒少於 150 根)")
            else:
                df = calculate_indicators(df)
                
                if strategy.startswith("A"):
                    mode = "Classic Wyckoff"
                    signals, df = analyze_classic_wyckoff(df)
                else:
                    mode = f"EvR Trend ({evr_window}日)"
                    signals = analyze_evr_trend(df, window=evr_window)
                
                act, rea = get_todays_action(df, signals)
                
                st.markdown(f"### 📅 分析日期: {df.index[-1].strftime('%Y-%m-%d')}")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**今日建議**: {act}")
                with col2:
                    st.write(f"**原因**: {rea}")

                st.subheader(f"{mode} 訊號圖")
                fig = plot_chart(df, target_symbol, signals, mode)
                st.pyplot(fig)
                
                if not signals.empty:
                    st.subheader("📋 近期訊號列表")
                    last_5_sig = signals.tail(5).copy()
                    last_5_sig['Date'] = last_5_sig['Date'].dt.strftime('%Y-%m-%d')
                    last_5_sig['Price'] = last_5_sig['Price'].round(2)
                    st.dataframe(last_5_sig.style.format(precision=2), use_container_width=True)

        except Exception as e:
            st.error(f"發生錯誤: {e}")
            import traceback
            st.text(traceback.format_exc())
else:
    st.info("👈 請在左側選擇標的並點擊「開始分析」")
