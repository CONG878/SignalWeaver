# 🚀 SignalWeaver

SignalWeaver는 한국 주식 시장(KRX) 데이터를 기반으로 단기 및 중장기 주가를 예측하는 **다중 시점 직접 예측(Direct Multi-step Forecasting)** 프레임워크입니다.

단순히 다음 날의 주가를 맞추는 것을 넘어, 미래 5일(1주일)의 흐름을 직접 예측하고, **"수익률 중심 정렬 + 하드 필터링"** 전략을 통해 실전 투자 가능한 최적의 유니버스를 선정합니다.

## ✨ Key Features

* **Direct Multi-step Forecasting**: 1일 단위 예측의 오차 누적 문제를 해결하기 위해 Horizon별(h1~h5) 독립 모델을 학습하는 직접 예측 방식 채택.
* **Target-Centric Alignment**: 예측 결과의 날짜가 실제 발생일과 일치하도록 데이터를 정렬()하여 분석 직관성 극대화.
* **KRX-Specific Hard Filtering**: 거래정지, 상장폐지, 초저유동성, 작전주(테마주) 혐의 종목을 물리적으로 제거하여 안정성 확보.
* **Human-in-the-loop Selection**: 기계는 기대 수익률 순으로 후보를 정렬하고, 최종 투자는 리포트를 검토한 인간이 결정하는 협업 구조.
* **Modular Architecture**: 수집 → 전처리 → 학습 → 미래예측 → 선정의 5단계 파이프라인이 독립적으로 구성됨.

## 🛠️ Tech Stack

* **Data**: FinanceDataReader, KRX Stock Data
* **Model**: LightGBM (Multi-output Regressor)
* **Processing**: Pandas, NumPy
* **Environment**: Python 3.10+ / Jupyter Notebook

## 📂 Project Structure

```bash
SignalWeaver/
├── config/                   # 통합 설정 파일
│   └── config.yaml              # 프로젝트 전체 파라미터
├── data/                     # 데이터 저장소 (날짜별 격리)
│   ├── 01_raw/{YYYYMMDD}/       # Step 1: 원천 데이터
│   ├── 02_processed/{YYYYMMDD}/ # Step 2: Feature + Target 데이터셋
│   ├── 03_results/{YYYYMMDD}/   # Step 3: 과거 검증용 예측 결과
│   ├── 04_models/{YYYYMMDD}/    # Step 3: 모델 아티팩트 (PKL)
│   └── 05_universe/{YYYYMMDD}/  # Step 5: 최종 선정된 후보군 및 리포트
├── src/                      # 소스 코드 모듈
│   ├── data_loader/             # 데이터 수집
│   ├── features/                # 피처 엔지니어링
│   ├── modeling/                # 학습 루프 (WalkForward)
│   ├── models/                  # 모델 래퍼
│   ├── universe/                # 유니버스 선정 로직
│   └── utils/                   # 필터, 리스크, 트레이딩 유틸
├── 01_collect_data.ipynb     # Step 1: 데이터 수집
├── 02_build_dataset.ipynb    # Step 2: 전처리 + Feature 생성
├── 03_train_predict.ipynb    # Step 3: 모델 학습 및 검증
├── 04_forecast_future.ipynb  # Step 4: 미래(Next 5 days) 예측
└── 05_universe_selection.ipynb # Step 5: 최종 유니버스 선정

```

## 🚀 Quick Start

1. `config/config.yaml`에서 프로젝트 기준일(`reference_date`) 설정.
2. `01` ~ `05` 단계 노트북을 순차적으로 실행.
3. `data/05_universe/{date}/investment_report.xlsx`를 열어 투자 종목 최종 선별.

---

## 🔧 Pipeline Steps

### **Step 1: 원시 데이터 수집** (`01_collect_data.ipynb`)

* **역할**: KRX 전 종목 시세 수집, 종목 마스터 생성
* **특징**: FinanceDataReader 활용, 로그 스케일 랜덤 대기(서버 차단 회피)
* **출력**: `krx_prices_{date}.parquet`, `ticker_master_{date}.csv`

### **Step 2: 데이터셋 구축** (`02_build_dataset.ipynb`)

* **역할**: 기술적 지표(MA, RSI, MACD 등) 계산, 메타 지표(유동성, 리스크) 생성, **Target(`target_log_close`)** 정의
* **특징**: 초기 Warm-up 기간(60일) 제거, 결측치 처리
* **출력**: `dataset.parquet` (학습용 통합 데이터)

### **Step 3: Multi-Horizon 학습 및 검증** (`03_train_predict.ipynb`)

* **역할**: Horizon(1~5일)별 독립 LightGBM 모델 학습, Walk-Forward Validation 수행
* **특징**: 과거 데이터를 통해 모델의 **정확도(Accuracy)**와 **신뢰도(Confidence)**를 검증하는 단계
* **출력**: `predictions.parquet` (과거 예측값), `lightgbm_multi.pkl` (학습된 모델)

### **Step 4: 미래 예측** (`04_forecast_future.ipynb`)

* **역할**: 학습된 모델을 사용하여 **"아직 오지 않은 미래 5일"**의 주가 흐름 예측
* **프로세스**:
* 가장 최신 데이터(`latest_data`) 로드
* h1~h5 모델을 사용하여 향후 5일간의 로그 종가 경로(`pred_log_close`) 생성


* **출력**: `future_forecasts.parquet` (미래 예측 경로)

### **Step 5: 유니버스 선정** (`05_universe_selection.ipynb`)

* **역할**: 예측된 미래 수익률과 리스크를 종합하여 **최종 투자 후보군(Top-K)** 선정
* **전략 (Return-First, Risk-Aware)**:
1. **Hard Filtering**: 거래정지, 상폐, 초저유동성, **작전주/테마주** 혐의 종목 물리적 제거.
2. **Ranking**: **시간당 로그 수익률(`daily_log_return`)** 기준으로 내림차순 정렬.
3. **Reporting**: 수익률, 방향성 정확도, 리스크 점수가 포함된 엑셀 리포트 생성.


* **출력**:
* `universe_candidates.parquet`: 시스템 선정 Top-K 후보 (시스템용)
* `investment_report.xlsx`: 의사결정 지원 리포트 (사람용)



---

## 📊 Data Policy & Filters

* **Target**: `log(close)` (로그 종가)
* **Hard Filters**:
* **Tradability**: 거래정지/상장폐지 종목 제외
* **Liquidity**: 20일 평균 거래대금 5천만 원 미만 제외
* **Price**: 예측가 1,000원 미만(동전주) 제외
* **Manipulation**: 20일 내 100% 이상 급등 + 거래량 5배 폭증 종목 제외 (작전주 필터)



## 🔜 Next Steps

* **Step 6**: Portfolio Optimization (MVO, Risk Parity)
* **Step 7**: Backtesting & Simulation