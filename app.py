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
st.set_page_config(page_title="AI Stock Trade-Aid", layout="wide")

# --- [내부 엔진 1] 뉴스 감성 분석 ---
def get_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        # 최신 뉴스 5개의 제목을 분석하여 긍정/부정 수치화 (-1.0 ~ 1.0)
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- [내부 엔진 2] 전 종목 동적 스크리닝 (S&P 500) ---
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
    except:
        return ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META']

# --- [내부 엔진 3] 정밀 분석 및 전략 수립 ---
def run_full_analysis(ticker):
    # 데이터 수집 (최근 2년)
    df = yf.download(ticker, period='2y', progress=False)
    
    # [데이터 검증] 최소 35거래일 이상의 데이터 확인
    if df is None or len(df) < 35:
        raise ValueError("분석을 위한 충분한 데이터가 없습니다 (최소 35일 필요).")
    
    # Multi-index 데이터 구조 평탄화 (yfinance 최신 버전 대응)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close_series = df['Close'].squeeze()
    sentiment = get_sentiment(ticker)
    
    # Prophet 데이터 준비 및 감성 변수 추가
    p_df = df[['Close']].reset_index().rename(columns={'Date':'ds', 'Close':'y'})
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment')
    model.fit(p_df)
    
    # 향후 30일 예측
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment
    forecast = model.predict(future)
    
    # 기간별 예측값 산출 (인덱스 역계산)
    curr_p = float(close_series.iloc[-1])
    p_5d = float(forecast['yhat'].iloc[-26])
    p_10d = float(forecast['yhat'].iloc[-21])
    p_20d = float(forecast['yhat'].iloc[-11])
    p_30d = float(forecast['yhat'].iloc[-1])
    
    # 전략가 산출 (예측 범위 상/하단 활용)
    window = forecast.iloc[-30:]
    max_p = float(window['yhat_upper'].max())
    min_p = float(window['yhat_lower'].min())
    
    return {
        'model': model, 'forecast': forecast, 'sentiment': sentiment,
        'current_p': curr_p, 'p_5d': p_5d, 'p_10d': p_10d, 'p_20d': p_20d, 'p_30d': p_30d,
        'return_pct': ((p_30d - curr_p) / curr_p) * 100,
        'buy': (min_p * 0.7) + (curr_p * 0.3), # 보수적 매수가 제안
        'sell': max_p, 'stop': min_p * 0.95,  # 리스크 관리용 손절가
        'rsi': ta.momentum.rsi(close_series, window=14).iloc[-1]
    }

# --- UI 레이아웃 설계 ---
st.title("🤖 Stock Trade-Aid v3.5")
st.markdown("---")

st.sidebar.header("🕹️ Sidebar ")
menu = st.sidebar.radio("모드 선택", ["🔍 실시간 종목 스크리닝", "🎯 단일 종목 정밀 분석"])

# [모드 1] 실시간 종목 스크리닝
if menu == "🔍 실시간 종목 스크리닝":
    st.subheader("오늘의 S&P 500 화제 종목 발굴")
    st.write("당일 거래량이 급증하고 기술적 지표가 우수한 유망 종목을 AI가 자동으로 스캔합니다.")
    
    if st.button("시장 스캔 및 추천 리스트 생성"):
        all_tickers = get_sp500_tickers()
        recommends = []
        
        with st.spinner('전 시장 데이터를 실시간 스캔 중...'):
            # 성능을 위해 상위 50개 종목 우선 스캔 가이드
            for t in all_tickers[:50]:
                try:
                    df_brief = yf.download(t, period='20d', progress=False)
                    if isinstance(df_brief.columns, pd.MultiIndex): 
                        df_brief.columns = df_brief.columns.get_level_values(0)
                    
                    # 거래량 화제성(최근 20일 평균 대비 1.5배) 필터링
                    vol_ratio = df_brief['Volume'].iloc[-1] / df_brief['Volume'].mean()
                    if vol_ratio > 1.5:
                        res = run_full_analysis(t)
                        if res['return_pct'] > 5.0: # 수익률 5% 이상 종목만
                            recommends.append(res)
                except: continue
        
        if recommends:
            res_df = pd.DataFrame(recommends).sort_values('return_pct', ascending=False)
            st.success(f"조건에 맞는 화제 종목 {len(recommends)}개를 발견했습니다!")
            st.table(res_df[['Ticker', 'current_p', 'p_30d', 'return_pct', 'buy', 'sell', 'stop']])
        else:
            st.info("현재 조건을 만족하는 급등 유망 종목이 없습니다.")

# [모드 2] 단일 종목 정밀 분석
elif menu == "🎯 단일 종목 정밀 분석":
    st.subheader("🎯 특정 종목 정밀 분석 및 매매 전략")
    
    # 입력창 레이아웃 구성
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        target_ticker = st.text_input("분석할 티커를 입력하세요", placeholder="예: NVDA, AAPL, 005930.KS").upper()
    with col_btn:
        st.write(" ") # 수직 정렬을 위한 여백
        start_analyze = st.button("정밀 전략 수립 실행")

    if start_analyze:
        if not target_ticker:
            st.warning("분석할 티커를 입력해 주세요.")
        else:
            try:
                with st.spinner(f'AI가 {target_ticker}의 과거 패턴과 시장 심리를 학습 중입니다...'):
                    res = run_full_analysis(target_ticker)
                    
                    # 1. 상단 핵심 지표
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("현재가", f"{res['current_p']:.2f}")
                    c2.metric("30일 예측 수익률", f"{res['return_pct']:.2f}%")
                    c3.metric("RSI (상대강도)", f"{res['rsi']:.2f}")
                    c4.metric("시장 감성 점수", f"{res['sentiment']:.2f}")

                    # 2. 기간별 상세 예측 테이블
                    st.markdown("---")
                    st.subheader("📅 AI 기간별 상세 가격 예측")
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("5일 후", f"{res['p_5d']:.2f}")
                    d2.metric("10일 후", f"{res['p_10d']:.2f}")
                    d3.metric("20일 후", f"{res['p_20d']:.2f}")
                    d4.metric("30일 후", f"{res['p_30d']:.2f}")

                    # 3. AI 매매 가이드 (손절가 포함)
                    st.info(f"🎯 **AI 추천 전략:** 적정 매수가 **{res['buy']:.2f}** | 목표 매도가 **{res['sell']:.2f}** | 손절 가격 **{res['stop']:.2f}**")

                    # 4. 예측 시뮬레이션 차트
                    st.subheader(f"📈 {target_ticker} 향후 30일 시뮬레이션 및 신뢰 구간")
                    fig = res['model'].plot(res['forecast'])
                    # 오늘 시점 표시
                    plt.axvline(x=datetime.datetime.now(), color='red', linestyle='--', label='Today')
                    plt.legend()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"분석 실패: {e}. 티커를 확인하거나 잠시 후 다시 시도해 주세요.")

st.markdown("---")
st.caption("면책 조항: 본 시스템은 통계적 모델에 기반한 정보 제공이 목적이며, 모든 투자의 책임은 사용자 본인에게 있습니다.")
