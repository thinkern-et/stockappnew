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
st.set_page_config(page_title="AI 주식 예측 및 전략 대시보드 v2.5", layout="wide")

# --- 뉴스 감성 분석 함수 ---
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- 단기/중기 예측 및 매수매도 전략 계산 함수 ---
def run_enhanced_strategy_analysis(df, ticker):
    if df is None or len(df) < 30:
        raise ValueError("분석을 위한 충분한 데이터가 없습니다. (최소 30영업일 필요)")

    close_series = df['Close'].squeeze()
    sentiment = get_sentiment_score(ticker)
    
    # Prophet 데이터 준비
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment 

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    # 30일 예측 수행
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment 
    forecast = model.predict(future)
    
    # [1] 기간별 예측값 추출
    current_price = float(close_series.iloc[-1])
    p_5d = float(forecast['yhat'].iloc[-26])  # 오늘로부터 5일 후 (인덱스 계산)
    p_10d = float(forecast['yhat'].iloc[-21])
    p_20d = float(forecast['yhat'].iloc[-11])
    p_30d = float(forecast['yhat'].iloc[-1])
    
    # [2] 예측 기간 내 최고/최저가 및 전략가 계산
    forecast_period = forecast.iloc[-30:] # 향후 30일 데이터
    max_p = float(forecast_period['yhat_upper'].max()) # 예측 범위 상단 기준 최고가
    min_p = float(forecast_period['yhat_lower'].min()) # 예측 범위 하단 기준 최저가
    
    # 적정 매수가: 예측 최저가와 현재가의 가중 평균 (보수적 접근)
    target_buy = (min_p * 0.7) + (current_price * 0.3)
    # 적정 매도가: 예측 최고가와 30일 예측가의 평균
    target_sell = (max_p + p_30d) / 2

    rsi = ta.momentum.rsi(close_series, window=14).iloc[-1]
    
    return {
        'model': model, 'forecast': forecast, 'sentiment': sentiment,
        'current_p': current_price, 'p_5d': p_5d, 'p_10d': p_10d, 
        'p_20d': p_20d, 'p_30d': p_30d, 'max_p': max_p, 'min_p': min_p,
        'target_buy': target_buy, 'target_sell': target_sell, 'rsi': rsi
    }

# --- UI 레이아웃 ---
st.title("🚀 AI 주식 전략 대시보드 v2.5")
st.sidebar.header("🔍 분석 설정")
ticker = st.sidebar.text_input("티커 입력 (예: AAPL, 005930.KS)", "AAPL")

if st.sidebar.button("AI 분석 및 전략 수립 시작"):
    try:
        with st.spinner('AI가 기간별 흐름을 분석하고 최적의 매매 가격을 산출 중입니다...'):
            df = yf.download(ticker, period='2y')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            res = run_enhanced_strategy_analysis(df, ticker)
            
            # 1. 기간별 예측값 분석 (Metrics)
            st.subheader("📅 기간별 주가 예측 분석")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("5일 후 예측", f"{res['p_5d']:.2f}")
            m2.metric("10일 후 예측", f"{res['p_10d']:.2f}")
            m3.metric("20일 후 예측", f"{res['p_20d']:.2f}")
            m4.metric("30일 후 예측", f"{res['p_30d']:.2f}")

            # 2. AI 추천 매매 전략
            st.markdown("---")
            st.subheader("🎯 AI 추천 매매 가격 가이드")
            c1, c2, c3 = st.columns(3)
            c1.success(f"**적정 매수 포인트**: {res['target_buy']:.2f}")
            c2.error(f"**목표 매도 가격**: {res['target_sell']:.2f}")
            c3.info(f"**예측 범위(최고-최저)**: {res['min_p']:.2f} ~ {res['max_p']:.2f}")

            # 3. 시각화
            st.markdown("---")
            st.subheader("📈 향후 30일 시뮬레이션 및 신뢰 구간")
            fig = res['model'].plot(res['forecast'])
            plt.axvline(x=df.index[-1], color="red", linestyle="--", label="Today")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")

st.markdown("---")
st.caption("면책 조항: 본 시스템은 통계적 예측치를 제공할 뿐이며 모든 투자 책임은 본인에게 있습니다.")

