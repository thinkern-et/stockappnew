import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet
from textblob import TextBlob  # 감성 분석 라이브러리 추가
import matplotlib.pyplot as plt
import datetime

# --- 뉴스 감성 분석 함수 ---
def get_sentiment_score(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0.0
        
        scores = []
        for item in news[:5]: # 최신 뉴스 5개 분석
            analysis = TextBlob(item['title'])
            # polarity: -1.0(매우 부정) ~ 1.0(매우 긍정)
            scores.append(analysis.sentiment.polarity)
        
        return sum(scores) / len(scores)
    except:
        return 0.0

# --- 수정된 분석 함수 (Prophet 고도화) ---
def run_enhanced_analysis(df, ticker):
    # [span_10](start_span)[span_11](start_span)데이터 부족 예외 처리[span_10](end_span)[span_11](end_span)
    if df.empty or len(df) < 30:
        raise ValueError("데이터가 부족합니다.")

    close_series = df['Close'].squeeze()
    
    # 1. 감성 점수 가져오기
    sentiment_score = get_sentiment_score(ticker)
    
    # 2. Prophet 데이터 준비
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    
    # 감성 점수를 데이터프레임에 추가 (과거 데이터에는 평균값 적용, 당일은 실시간 점수)
    p_df['sentiment'] = sentiment_score 

    # 3. Prophet 모델 설정 및 추가 회귀 변수 등록
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment') # 감성 분석 변수 추가
    model.fit(p_df)
    
    # 4. 미래 예측 시에도 감성 점수 반영
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment_score # 향후 30일도 현재의 심리가 유지된다고 가정
    forecast = model.predict(future)
    
    # [span_12](start_span)[span_13](start_span)[span_14](start_span)지표 계산[span_12](end_span)[span_13](end_span)[span_14](end_span)
    rsi = ta.momentum.rsi(close_series, window=14).iloc[-1]
    curr_p = float(close_series.iloc[-1])
    pred_p = float(forecast['yhat'].iloc[-1])
    return_pct = (pred_p - curr_p) / curr_p * 100
    
    return {
        'model': model, 'forecast': forecast,
        'current_p': curr_p, 'pred_p': pred_p,
        'return_pct': return_pct, 'rsi': rsi,
        'sentiment': sentiment_score
    }

# --- UI 레이아웃 부분에 감성 지표 출력 추가 ---
# (중략) ... res = run_enhanced_analysis(df, ticker) 수행 후
# col1, col2, col3, col4 = st.columns(4)
# col4.metric("시장 감성 점수", f"{res['sentiment']:.2f}")
