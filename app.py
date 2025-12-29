import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet
from textblob import TextBlob
import matplotlib.pyplot as plt
import datetime
import warnings

# 설정 및 경고 무시
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI 주식 예측 대시보드 v2.0", layout="wide")

# --- 뉴스 감성 분석 함수 ---
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        # 최신 뉴스 5개의 제목을 분석하여 긍정/부정 수치화
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- 고도화된 분석 및 예측 함수 ---
def run_enhanced_analysis(df, ticker):
    # [span_1](start_span)[span_2](start_span)[데이터 검증] 최소 30일 데이터 확인[span_1](end_span)[span_2](end_span)
    if df is None or len(df) < 30:
        raise ValueError("분석을 위한 충분한 데이터가 없습니다. (최소 30영업일 필요)")

    # [span_3](start_span)데이터 정제 및 차원 평탄화[span_3](end_span)
    close_series = df['Close'].squeeze()
    sentiment = get_sentiment_score(ticker)
    
    # [span_4](start_span)Prophet 데이터 준비[span_4](end_span)
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment # 추가 회귀 변수 삽입

    # [span_5](start_span)[span_6](start_span)모델 설정 및 학습[span_5](end_span)[span_6](end_span)
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    # [span_7](start_span)[span_8](start_span)미래 예측 (30일)[span_7](end_span)[span_8](end_span)
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment 
    forecast = model.predict(future)
    
    # 수치 계산
    current_price = float(close_series.iloc[-1])
    predicted_price = float(forecast['yhat'].iloc[-1])
    return_pct = ((predicted_price - current_price) / current_price) * 100
    
    # [span_9](start_span)기술적 지표 계산[span_9](end_span)
    rsi = ta.momentum.rsi(close_series, window=14).iloc[-1]
    
    return {
        'model': model, 
        'forecast': forecast, 
        'sentiment': sentiment,
        'current_p': current_price,
        'pred_p': predicted_price,
        'return_pct': return_pct,
        'rsi': rsi
    }

# --- UI 레이아웃 ---
st.title("🚀 나만의 AI 주식 예측 대시보드 v2.0")
st.markdown("---")

# [span_10](start_span)사이드바 설정[span_10](end_span)
st.sidebar.header("🔍 분석 설정")
ticker = st.sidebar.text_input("티커 입력 (예: AAPL, 005930.KS)", "AAPL")
analyze_btn = st.sidebar.button("AI 분석 시작")

if analyze_btn:
    try:
        with st.spinner(f'{ticker} 데이터 수집 및 AI 모델 학습 중...'):
            # [span_11](start_span)데이터 수집[span_11](end_span)
            df = yf.download(ticker, period='2y')
            
            # [span_12](start_span)Multi-index 처리[span_12](end_span)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 분석 실행
            res = run_enhanced_analysis(df, ticker)
            
            # 1. [span_13](start_span)지표 표시 (Metrics)[span_13](end_span)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("현재 주가", f"{res['current_p']:.2f}")
            col2.metric("30일 예측", f"{res['pred_p']:.2f}", f"{res['return_pct']:.2f}%")
            col3.metric("RSI (상대 강도)", f"{res['rsi']:.2f}")
            col4.metric("감성 점수", f"{res['sentiment']:.2f}")

            # 2. [span_14](start_span)예측 차트 시각화[span_14](end_span)
            st.markdown("---")
            st.subheader(f"📈 {ticker} 향후 30일 주가 예측 시뮬레이션")
            
            fig = res['model'].plot(res['forecast'])
            plt.axvline(x=df.index[-1], color="red", linestyle="--", label="Today")
            plt.legend()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"⚠️ 분석 오류: {e}")

# [span_15](start_span)하단 면책 조항[span_15](end_span)
st.markdown("---")
st.caption("면책 조항: 본 시스템의 예측 결과는 통계적 모델에 의한 참고 자료일 뿐이며, 모든 투자 결정의 책임은 투자자 본인에게 있습니다.")

