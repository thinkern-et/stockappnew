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
        # [span_2](start_span)[span_3](start_span)최신 뉴스 5개의 제목을 분석하여 긍정/부정 수치화[span_2](end_span)[span_3](end_span)
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- 핵심 분석 및 전략 계산 함수 ---
def run_final_strategy_analysis(df, ticker):
    # [span_4](start_span)[span_5](start_span)[데이터 검증] 최소 30일 데이터 확인[span_4](end_span)[span_5](end_span)
    if df is None or len(df) < 30:
        raise ValueError("분석을 위한 데이터가 부족합니다. (최소 30일 필요)")

    close_series = df['Close'].squeeze()
    sentiment = get_sentiment_score(ticker)
    
    # [span_6](start_span)Prophet 학습 데이터 준비[span_6](end_span)
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment 

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    # [span_7](start_span)향후 30일 예측 수행[span_7](end_span)
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment 
    forecast = model.predict(future)
    
    # [span_8](start_span)[1] 현재 및 기간별 예측가 추출[span_8](end_span)
    current_p = float(close_series.iloc[-1])
    # 인덱스 역산으로 기간별 값 추출
    p_5d = float(forecast['yhat'].iloc[-26])
    p_10d = float(forecast['yhat'].iloc[-21])
    p_20d = float(forecast['yhat'].iloc[-11])
    p_30d = float(forecast['yhat'].iloc[-1])
    
    # [span_9](start_span)[2] 전략 가격 계산 (예측 범위 반영)[span_9](end_span)
    forecast_30d = forecast.iloc[-30:]
    max_p = float(forecast_30d['yhat_upper'].max())
    min_p = float(forecast_30d['yhat_lower'].min())
    
    # [span_10](start_span)적정 매수: 예측 범위 하단(지지선)과 현재가의 가중 평균[span_10](end_span)
    target_buy = (min_p * 0.6) + (current_p * 0.4)
    # [span_11](start_span)목표 매도: 예측 최고가와 30일 예측가의 평균[span_11](end_span)
    target_sell = (max_p + p_30d) / 2
    # [span_12](start_span)손절 가격: 예측 범위 하단에서 3% 추가 하락 지점 (방어적 설계)[span_12](end_span)
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
st.title("🚀 AI 주식 매매 전략 대시보드 v2.5")
st.markdown("---")

ticker = st.sidebar.text_input("티커 입력 (예: AAPL, 005930.KS)", "AAPL")

if st.sidebar.button("전략 분석 실행"):
    try:
        with st.spinner('AI가 시나리오를 설계 중입니다...'):
            df = yf.download(ticker, period='2y', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            res = run_final_strategy_analysis(df, ticker)
            
            # 1. [span_13](start_span)현재가 및 기간별 예측 수치[span_13](end_span)
            st.subheader(f"📊 {ticker} 현재가 및 기간별 예측")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("현재 가격", f"{res['current_p']:.2f}")
            m2.metric("5일 후", f"{res['p_5d']:.2f}")
            m3.metric("10일 후", f"{res['p_10d']:.2f}")
            m4.metric("20일 후", f"{res['p_20d']:.2f}")
            m5.metric("30일 후", f"{res['p_30d']:.2f}", f"{res['return_pct']:.2f}%")

            # 2. [span_14](start_span)[span_15](start_span)핵심 매매 가이드라인[span_14](end_span)[span_15](end_span)
            st.markdown("---")
            st.subheader("🎯 AI 추천 매매 가이드")
            c1, c2, c3 = st.columns(3)
            c1.success(f"**적정 매수 가격**: {res['target_buy']:.2f}")
            c2.error(f"**목표 매도 가격**: {res['target_sell']:.2f}")
            c3.warning(f"**⚠️ 손절 가격**: {res['stop_loss']:.2f}")

            # 3. [span_16](start_span)[span_17](start_span)차트 시각화[span_16](end_span)[span_17](end_span)
            st.markdown("---")
            st.subheader("📈 향후 30일 주가 시뮬레이션")
            fig = res['model'].plot(res['forecast'])
            plt.axvline(x=df.index[-1], color="red", linestyle="--", label="Today")
            plt.legend()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"오류 발생: {e}")

st.markdown("---")
[span_18](start_span)[span_19](start_span)st.caption("면책 조항: 본 시스템은 통계적 모델 기반 예측치이며, 모든 투자 책임은 본인에게 있습니다.[span_18](end_span)[span_19](end_span)")
