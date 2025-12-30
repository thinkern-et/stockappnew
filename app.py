import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet
from textblob import TextBlob
import matplotlib.pyplot as plt
import datetime
import requests
import warnings

# 설정 및 경고 무시
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI 주식 전략 대시보드 v2.5", layout="wide")

# --- 뉴스 감성 분석 함수 ---
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- 알림 전송 함수 ---
def send_telegram_msg(message):
    # [span_2](start_span)[주의] 개정판 7장에 따라 본인의 토큰과 ID로 설정 필요[span_2](end_span)
    token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    try: requests.get(url)
    except: pass

# --- 핵심 분석 및 전략 계산 함수 ---
def run_final_strategy_analysis(df, ticker):
    [span_3](start_span)if df is None or len(df) < 30:[span_3](end_span)
        raise ValueError("데이터가 부족합니다. (최소 30일 필요)")

    close_series = df['Close'].squeeze()
    sentiment = get_sentiment_score(ticker)
    
    # Prophet 학습 데이터 준비
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment 

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment 
    forecast = model.predict(future)
    
    # [span_4](start_span)[1] 현재 및 기간별 예측가 추출[span_4](end_span)
    current_p = float(close_series.iloc[-1])
    p_5d, p_10d = float(forecast['yhat'].iloc[-26]), float(forecast['yhat'].iloc[-21])
    p_20d, p_30d = float(forecast['yhat'].iloc[-11]), float(forecast['yhat'].iloc[-1])
    
    # [span_5](start_span)[span_6](start_span)[2] 전략 가격 계산 (최고/최저가 반영)[span_5](end_span)[span_6](end_span)
    forecast_30d = forecast.iloc[-30:]
    max_p = float(forecast_30d['yhat_upper'].max())
    min_p = float(forecast_30d['yhat_lower'].min())
    
    # 적정 매수: 예측 범위 하단과 현재가의 가중 평균
    target_buy = (min_p * 0.6) + (current_p * 0.4)
    # 목표 매도: 예측 최고가와 30일 가격의 평균
    target_sell = (max_p + p_30d) / 2
    # 손절 가격: 예측 범위 하단(min_p)에서 추가 3% 하락 지점 (방어적 설계)
    stop_loss = min_p * 0.97

    rsi = ta.momentum.rsi(close_series, window=14).iloc[-1]
    return_pct = ((p_30d - current_p) / current_p) * 100
    
    return {
        'model': model, 'forecast': forecast, 'sentiment': sentiment,
        'current_p': current_p, 'p_5d': p_5d, 'p_10d': p_10d, 
        'p_20d': p_20d, 'p_30d': p_30d, 'max_p': max_p, 'min_p': min_p,
        'target_buy': target_buy, 'target_sell': target_sell, 
        'stop_loss': stop_loss, 'rsi': rsi, 'return_pct': return_pct
    }

# --- UI 레이아웃 ---
st.title("🚀 AI 주식 매매 전략 대시보드")
ticker = st.sidebar.text_input("티커 입력 (예: AAPL, 005930.KS)", "AAPL")

if st.sidebar.button("전략 분석 실행"):
    try:
        with st.spinner('AI가 매매 시나리오를 설계 중입니다...'):
            df = yf.download(ticker, period='2y')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            res = run_final_strategy_analysis(df, ticker)
            
            # 1. [span_7](start_span)현재가 및 기간별 예측 수치[span_7](end_span)
            st.subheader(f"📊 {ticker} 현재가 및 기간별 예측")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("현재 가격", f"{res['current_p']:.2f}")
            m2.metric("5일 후", f"{res['p_5d']:.2f}")
            m3.metric("10일 후", f"{res['p_10d']:.2f}")
            m4.metric("20일 후", f"{res['p_20d']:.2f}")
            m5.metric("30일 후", f"{res['p_30d']:.2f}", f"{res['return_pct']:.2f}%")

            # 2. [span_8](start_span)핵심 매매 가이드 (손절가 포함)[span_8](end_span)
            st.markdown("---")
            st.subheader("🎯 AI 추천 매매 가이드라인")
            c1, c2, c3 = st.columns(3)
            c1.success(f"**적정 매수 가격**: {res['target_buy']:.2f}")
            c2.error(f"**목표 매도 가격**: {res['target_sell']:.2f}")
            c3.warning(f"**⚠️ 손절 가격**: {res['stop_loss']:.2f}")

            # 3. [span_9](start_span)차트 시각화[span_9](end_span)
            st.markdown("---")
            st.subheader("📈 향후 30일 주가 시뮬레이션")
            fig = res['model'].plot(res['forecast'])
            plt.axvline(x=df.index[-1], color="red", linestyle="--")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"오류: {e}")
