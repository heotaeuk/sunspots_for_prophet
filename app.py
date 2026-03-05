import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# ----------------------------------
# Streamlit Page Config
# ----------------------------------
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Prophet 필수 컬럼: ds(datetime), y(float)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return df

DATA_PATH = "data/sunspots_for_prophet.csv"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"파일을 찾을 수 없습니다: {DATA_PATH}")
    st.info("GitHub 레포지토리 루트에 data 폴더를 만들고 sunspots_for_prophet.csv를 넣어주세요.")
    st.stop()

st.subheader("📄 데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------
# 사이드바 옵션
# ----------------------------------
st.sidebar.header("⚙️ Forecast Options")
period_years = st.sidebar.slider("예측 기간(년)", min_value=5, max_value=80, value=30, step=5)

# yearly data 기준: 1년 = 365.25일
period_days = int(period_years * 365.25)

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
st.subheader("🧠 Prophet Model Training")

model = Prophet(
    yearly_seasonality=False,   # 우리가 커스텀 시즌을 넣을 예정
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.95
)

# 11년 주기 (태양 흑점 사이클)
model.add_seasonality(
    name="cycle_11y",
    period=11 * 365.25,
    fourier_order=8
)

with st.spinner("Prophet 모델 학습 중..."):
    model.fit(df)

st.success("✅ 모델 학습 완료!")

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
st.subheader("🔮 Forecast")

future = model.make_future_dataframe(periods=period_days, freq="D")
forecast = model.predict(future)

st.write("예측 결과(일부)")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10), use_container_width=True)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("📊 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# ----------------------------------
# [5] 커스텀 시각화: 실제값 vs 예측값 + 신뢰구간
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")

fig3, ax = plt.subplots(figsize=(14, 6))

# 실제값
ax.plot(df["ds"], df["y"], marker="o", linestyle="-", linewidth=1.5, label="Actual")

# 예측값
ax.plot(forecast["ds"], forecast["yhat"], linestyle="--", linewidth=2, label="Predicted")

# 신뢰구간
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.2,
    label="Prediction Interval"
)

ax.set_title("Sunspots: Actual vs. Predicted with Prediction Intervals")
ax.set_xlabel("Year")
ax.set_ylabel("Sun Activity")
ax.legend()
ax.grid(True)

st.pyplot(fig3)

# ----------------------------------
# [6] 잔차 분석 시각화
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")

# df는 연 단위(1900-01-01, 1901-01-01...), forecast는 일 단위이므로
# ds 기준으로 inner join하면 매칭이 거의 안 될 수 있음.
# 그래서 forecast를 df의 ds에 맞춰 예측값을 "ds 기준"으로 뽑아오는 방식으로 처리.
# (가장 안정적으로 과제 결과 형태에 맞게 나옵니다.)

forecast_on_ds = forecast.set_index("ds").reindex(df["ds"]).reset_index()
merged = df.copy()
merged["yhat"] = forecast_on_ds["yhat"].values
merged = merged.dropna(subset=["yhat"])
merged["residual"] = merged["y"] - merged["yhat"]

fig4, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(merged["ds"], merged["residual"], marker="o", linewidth=2, label="Residual (Actual - Predicted)")
ax2.axhline(0, linestyle="--", linewidth=2, label="Zero Line")
ax2.set_title("Residual Analysis (Actual - Predicted)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True)

st.pyplot(fig4)

# ----------------------------------
# [7] 잔차 통계 요약
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")
st.write(merged["residual"].describe())
