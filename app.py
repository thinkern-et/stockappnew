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
        # 최신 뉴스 5개의 제목 긍정/부정 분석
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- 고도화된 분석 및 예측 함수 ---
def run_enhanced_analysis(df, ticker):
    # [데이터 검증] 최소 30일 데이터 필요
    if df is None or len(df) < 30:
        raise ValueError("분석을 위한 충분한 데이터가 없습니다. (최소 30영업일 필요)")

    # 1차원 데이터 추출 및 차원 평탄화
    close_series = df['Close'].squeeze()
    sentiment = get_sentiment_score(ticker)
    
    # Prophet 데이터 준비
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment # 추가 회귀 변수 삽입

    # 모델 설정 및 학습
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    # 미래 예측 (30일)
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment # 현재 감성이 유지된다고 가정
    forecast = model.predict(future)
    
    # 수치 계산 (변수 선언 순서 최적화)
    current_price = float(close_series.iloc[-1])
    predicted_price = float(forecast['yhat'].iloc[-1])
    return_pct = ((predicted_price - current_price) / current_price) * 100
    
    # 기술적 지표 계산
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

# 사이드바 설정
st.sidebar.header("🔍 분석 설정")
ticker = st.sidebar.text_input("티커 입력 (예: AAPL, 005930.KS)", "AAPL")
analyze_btn = st.sidebar.button("AI 분석 시작")

if analyze_btn:
    try:
        with st.spinner(f'{ticker} 데이터 수집 및 AI 모델 학습 중...'):
            # 데이터 수집
            df = yf.download(ticker, period='2y')
            
            # Multi-index 처리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 분석 실행
            res = run_enhanced_analysis(df, ticker)
            
            # 1. 상단 주요 지표 표시 (Metrics)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("현재 주가", f"{res['current_p']:.2f}")
            with col2:
                st.metric("30일 후 예측가", f"{res['pred_p']:.2f}", f"{res['return_pct']:.2f}%")
            with col3:
                st.metric("RSI (상대 강도)", f"{res['rsi']:.2f}", 
                          "과매수" if res['rsi'] > 70 else "과매도" if res['rsi'] < 30 else "정상")
            with col4:
                st.metric("시장 감성 점수", f"{res['sentiment']:.2f}", 
                          "긍정" if res['sentiment'] > 0 else "부정" if res['sentiment'] < 0 else "중립")

            # 2. 주가 예측 차트 시각화
            st.markdown("---")
            st.subheader(f"📈 {ticker} 향후 30일 가격 예측 시뮬레이션")
            
            fig = res['model'].plot(res['forecast'])
            # 오늘 날짜에 수직선 표시
            plt.axvline(x=df.index[-1], color="red", linestyle="--", label="Today")
            plt.legend()
            st.pyplot(fig)
            
            # 3. 추가 인사이트 제공
            st.info(f"💡 **AI 의견:** Prophet 모델은 과거 패턴과 감성 점수를 바탕으로 30일 후 주가가 약 **{res['return_pct']:.2f}%** 변동할 것으로 예측했습니다.")

    except Exception as e:
        st.error(f"⚠️ 분석 중 오류가 발생했습니다: {e}")
        st.info("팁: 티커가 정확한지 확인하시고(예: 삼전은 005930.KS), 충분한 과거 데이터가 있는지 확인하세요.")

# 하단 면책 조항
st.markdown("---")
[span_5](start_span)[span_6](start_span)st.caption("면책 조항: 본 시스템의 예측 결과는 통계적 모델에 의한 참고 자료일 뿐이며, 모든 투자 결정의 책임은 투자자 본인에게 있습니다.")[span_5](end_span)[span_6](end_span)
