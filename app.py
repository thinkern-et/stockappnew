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
st.set_page_config(page_title="AI 주식 예측 대시보드", layout="wide")

# --- 뉴스 감성 분석 함수 ---
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- 분석 및 예측 함수 ---
def run_enhanced_analysis(df, ticker):
    # [데이터 검증] 7장: 데이터 부족 예외 처리 반영
    if df is None or len(df) < 30:
        raise ValueError("분석을 위한 데이터가 부족합니다.")

    close_series = df['Close'].squeeze()
    sentiment = get_sentiment_score(ticker)
    return_pct=((predicted_price-current_price)/current_price)*100)
    # Prophet 데이터 준비
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment # 추가 회귀 변수

    # 모델 설정 및 학습
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    # 미래 예측 (30일)
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment
    forecast = model.predict(future)
    
    return {
        'model': model, 'forecast': forecast, 'sentiment': sentiment,
        'current_p': float(close_series.iloc[-1]),
        'pred_p': float(forecast['yhat'].iloc[-1]),
        "return_pct": return_pct,
        'rsi': ta.momentum.rsi(close_series, window=14).iloc[-1]
    }

# --- UI 레이아웃 (중략) ---
st.title("🚀 AI 주식 예측 대시보드 v2.0")
ticker = st.sidebar.text_input("티커 입력", "AAPL")

if st.sidebar.button("분석 시작"):
    try:
        df = yf.download(ticker, period='2y')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        res = run_enhanced_analysis(df, ticker)
        
        # 결과 표시
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재가", f"{res['current_p']:.2f}")
        col2.metric("30일 후 예측가", f"{res['pred_p']:.2f}", f"{res['return_pct']:.2f}%")
        col3.metric("RSI(상대 강도)", f"{res['rsi']:.2f}")
        col4.metric("감성 점수", f"{res['sentiment']:.2f}")
        

        st.subheader('향후 30일 가격 예측 차트')
        fig=res['model'].plot(res['forecast'])
        plt.axvline(x=df.index[-1], color="red", linestyle="--")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"오류 발생: {e}")
