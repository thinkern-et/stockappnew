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
st.set_page_config(page_title="AI 실전 투자 통합 대시보드", layout="wide")

# --- [내부 엔진 1] 뉴스 감성 분석 (TextBlob 활용) ---
def get_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news: return 0.0
        # 최신 뉴스 5개의 제목을 분석하여 긍정/부정 수치화 (-1.0 ~ 1.0)
        scores = [TextBlob(item['title']).sentiment.polarity for item in news[:5]]
        return sum(scores) / len(scores)
    except: return 0.0

# --- [내부 엔진 2] 전 종목 동적 스크리닝 로직 ---
@st.cache_data(ttl=3600) # 1시간 동안 리스트 캐싱
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]

# --- [내부 엔진 3] 정밀 분석 및 전략 수립 (5/10/20/30일 예측 포함) ---
def run_full_analysis(ticker):
    df = yf.download(ticker, period='2y', progress=False)
    [span_0](start_span)if df is None or len(df) < 35: # 기술 지표 및 학습을 위한 최소 데이터 검증[span_0](end_span)
        raise ValueError("분석을 위한 데이터가 부족합니다.")
    
    # [오류 해결] Multi-index 데이터 구조 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close_series = df['Close'].squeeze()
    sentiment = get_sentiment(ticker)
    
    # [span_1](start_span)Prophet 데이터 준비 및 감성 변수 추가[span_1](end_span)
    p_df = df[['Close']].reset_index().rename(columns={'Date':'ds', 'Close':'y'})
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    p_df['sentiment'] = sentiment

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.add_regressor('sentiment') # 감성 분석 점수 반영
    model.fit(p_df)
    
    future = model.make_future_dataframe(periods=30)
    future['sentiment'] = sentiment
    forecast = model.predict(future)
    
    # 기간별 예측가 추출
    curr_p = float(close_series.iloc[-1])
    p_5d = float(forecast['yhat'].iloc[-26])
    p_10d = float(forecast['yhat'].iloc[-21])
    p_20d = float(forecast['yhat'].iloc[-11])
    p_30d = float(forecast['yhat'].iloc[-1])
    
    # 전략가 산출: 최고/최저가 시나리오 반영
    window = forecast.iloc[-30:]
    max_p = float(window['yhat_upper'].max())
    min_p = float(window['yhat_lower'].min())
    
    return {
        'model': model, 'forecast': forecast, 'sentiment': sentiment,
        'current_p': curr_p, 'p_5d': p_5d, 'p_10d': p_10d, 'p_20d': p_20d, 'p_30d': p_30d,
        'return_pct': ((p_30d - curr_p) / curr_p) * 100,
        'buy': (min_p * 0.7) + (curr_p * 0.3), # 보수적 매수가 제안
        [span_2](start_span)'sell': max_p, 'stop': min_p * 0.95,  # 리스크 관리용 손절가[span_2](end_span)
        'rsi': ta.momentum.rsi(close_series, window=14).iloc[-1]
    }

# --- UI 레이아웃 설계 ---
st.title("🤖 AI 실전 투자 통합 대시보드 v3.5")
st.sidebar.header("🕹️ 컨트롤 센터")
menu = st.sidebar.radio("모드 선택", ["🔍 전 종목 실시간 스크리닝", "🎯 단일 종목 정밀 분석"])

if menu == "🔍 전 종목 실시간 스크리닝":
    st.subheader("오늘의 S&P 500 화제 종목 발굴")
    if st.button("시장 스캔 시작"):
        all_tickers = get_sp500_tickers()
        recommends = []
        with st.spinner('거래량 및 추세 데이터 분석 중...'):
            for t in all_tickers[:50]: # 성능 최적화를 위한 상위 샘플링
                try:
                    df_brief = yf.download(t, period='20d', progress=False)
                    if isinstance(df_brief.columns, pd.MultiIndex): df_brief.columns = df_brief.columns.get_level_values(0)
                    vol_focus = df_brief['Volume'].iloc[-1] / df_brief['Volume'].mean()
                    
                    [span_3](start_span)if vol_focus > 1.5: # 거래량 급증 종목 우선 추출[span_3](end_span)
                        res = run_full_analysis(t)
                        [span_4](start_span)if res['return_pct'] > 5.0: # 기대수익률 5% 이상만 추천[span_4](end_span)
                            recommends.append(res)
                except: continue
        
        if recommends:
            res_df = pd.DataFrame(recommends).sort_values('return_pct', ascending=False)
            st.success(f"당일 화제 종목 {len(recommends)}개를 발견했습니다!")
            st.table(res_df[['Ticker', 'current_p', 'p_30d', 'return_pct', 'buy', 'sell', 'stop']])
        else:
            st.info("현재 조건을 만족하는 급등 유망 종목이 없습니다.")

elif menu == "🎯 단일 종목 정밀 분석":
    target_ticker = st.text_input("분석할 티커를 입력하세요 (예: TSLA, 005930.KS)", "NVDA")
    if st.button("정밀 전략 수립"):
        try:
            with st.spinner('AI 모델 학습 및 전략 산출 중...'):
                res = run_full_analysis(target_ticker)
                
                # 상단 지표 출력
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("현재가", f"{res['current_p']:.2f}")
                col2.metric("30일 예측 수익률", f"{res['return_pct']:.2f}%")
                col3.metric("RSI (상대강도)", f"{res['rsi']:.2f}")
                col4.metric("시장 감성 점수", f"{res['sentiment']:.2f}")

                # 기간별 예측 테이블
                st.markdown("---")
                st.subheader("📅 기간별 상세 가격 예측")
                d1, d2, d3, d4 = st.columns(4)
                d1.write(f"**5일 후:** {res['p_5d']:.2f}")
                d2.write(f"**10일 후:** {res['p_10d']:.2f}")
                d3.write(f"**20일 후:** {res['p_20d']:.2f}")
                d4.write(f"**30일 후:** {res['p_30d']:.2f}")

                # 매매 가이드
                st.info(f"🎯 **AI 추천 전략:** 적정 매수가 **{res['buy']:.2f}** | 목표 매도가 **{res['sell']:.2f}** | 손절 가격 **{res['stop']:.2f}**")

                # 차트 시각화
                st.subheader("📈 향후 30일 시뮬레이션 차트")
                fig = res['model'].plot(res['forecast'])
                plt.axvline(x=datetime.datetime.now(), color='red', linestyle='--')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

st.markdown("---")
st.caption("면책 조항: 본 시스템은 통계 모델 기반의 정보 제공이 목적이며, 모든 투자의 책임은 사용자 본인에게 있습니다.")
