# 🚀 SignalWeaver

SignalWeaver는 한국 주식 시장(KRX) 데이터를 기반으로 단기 및 중장기 주가를 예측하는 **다중 시점 직접 예측(Direct Multi-step Forecasting)** 프레임워크입니다. 

단순히 다음 날의 주가를 맞추는 것을 넘어, 미래 5일(1주일)의 흐름을 직접 예측하고, **"수익률 중심 정렬 + 하드 필터링"** 전략을 통해 실전 투자 가능한 최적의 유니버스를 선정합니다.

## ✨ Key Features

- **Direct Multi-step Forecasting**: 1일 단위 예측의 오차 누적 문제를 해결하기 위해 Horizon별(h1~h5) 독립 모델을 학습하는 직접 예측 방식 채택.
- **Target-Centric Alignment**: 예측 결과의 날짜가 실제 발생일과 일치하도록 데이터를 정렬($X_{t-h} \rightarrow y_t$)하여 분석 직관성 극대화.
- **Step-Aligned Architecture**: 파이프라인 단계와 데이터 저장소가 1:1로 매핑되는 직관적인 구조 (`03_training`, `04_forecasts`, `05_universe`).
- **Centralized Path Management**: `ProjectPaths` 클래스를 통해 모든 경로를 중앙에서 관리하여 유지보수성 및 코드 안정성 확보.

## 🛠️ Tech Stack

- **Data**: FinanceDataReader, KRX Stock Data
- **Model**: LightGBM (Multi-output Regressor)
- **Processing**: Pandas, NumPy (Vectorized Operations)
- **Environment**: Python 3.10+ / Jupyter Notebook

## 📂 Project Structure

```bash
SignalWeaver/
├── config/                   # 통합 설정 파일
│   └── config.yaml              # 프로젝트 전체 파라미터 (경로, 모델 하이퍼파라미터)
├── data/                     # 데이터 저장소 (날짜별 격리)
│   ├── 01_raw/{YYYYMMDD}/       # Step 1: 원천 데이터 (Parquet + CSV)
│   ├── 02_processed/{YYYYMMDD}/ # Step 2: Feature + Target 데이터셋
│   ├── 03_training/{YYYYMMDD}/  # Step 3: 모델 아티팩트(.pkl) + 검증용 과거 예측
│   ├── 04_forecasts/{YYYYMMDD}/  # Step 4: 미래(Next 5 days) 예측 결과
│   └── 05_universe/{YYYYMMDD}/  # Step 5: 최종 선정된 후보군 및 리포트
├── src/                      # 소스 코드 모듈
│   ├── data_loader/             # 데이터 수집 (Collector)
│   ├── features/                # 피처 엔지니어링 (Builder, Technical)
│   ├── modeling/                # 학습 루프 (Trainer)
│   ├── models/                  # 모델 래퍼 (LGBM)
│   ├── universe/                # 유니버스 선정 로직 (Selector)
│   └── utils/                   # 유틸리티 (Config, Path, Filter, Risk)
├── 01_collect_data.ipynb     # Step 1: 데이터 수집
├── 02_build_dataset.ipynb    # Step 2: 전처리 + Feature 생성
├── 03_train_predict.ipynb    # Step 3: 모델 학습 및 검증
├── 04_forecast_future.ipynb  # Step 4: 미래 예측 (Inference)
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
* **출력**: `data/01_raw/{date}/krx_prices_{date}.parquet`

### **Step 2: 데이터셋 구축** (`02_build_dataset.ipynb`)

* **역할**: 기술적 지표 계산, 메타 지표(유동성, 리스크) 생성, Target 정의
* **출력**: `data/02_processed/{date}/dataset.parquet`

### **Step 3: Multi-Horizon 학습 및 검증** (`03_train_predict.ipynb`)

* **역할**: Horizon(1~5일)별 독립 모델 학습 및 Walk-Forward Validation 수행
* **특징**: 모델과 검증 결과를 하나의 폴더(`03_training`)에서 통합 관리
* **출력**:
* `data/03_training/{date}/lightgbm_multi.pkl` (학습된 모델)
* `data/03_training/{date}/predictions.parquet` (과거 검증 예측값)



### **Step 4: 미래 예측** (`04_forecast_future.ipynb`)

* **역할**: 학습된 모델로 "아직 오지 않은 미래 5일"의 주가 경로 예측
* **특징**: 검증 데이터와 분리된 전용 폴더(`04_forecasts`) 사용
* **출력**: `data/04_forecasts/{date}/future_forecasts.parquet`

### **Step 5: 유니버스 선정** (`05_universe_selection.ipynb`)

* **역할**: 예측된 수익률과 리스크를 종합하여 **최종 투자 후보군(Top-K)** 선정
* **전략**:
1. **Hard Filtering**: 거래정지/상폐, 초저유동성, 작전주, 동전주(Penny Stock) 물리적 제거.
2. **Log-Space Ranking**: `daily_log_return` 기준으로 정렬 (연산 효율 최적화).
3. **Reporting**: 수익률, 정확도, 리스크 점수가 포함된 리포트 생성.


* **출력**:
* `data/05_universe/{date}/universe_candidates.parquet` (시스템용 후보군)
* `data/05_universe/{date}/investment_report.xlsx` (투자자용 리포트)



---

## 📊 Data Policy & Filters

* **Target**: `log(close)` (로그 종가)
* **Hard Filters**:
* **Tradability**: 거래정지/상장폐지 종목 제외
* **Liquidity**: 20일 평균 거래대금 5천만 원 미만 제외
* **Price**: 예측가 1,000원 미만(동전주) 제외 (Log-space comparison 적용)
* **Manipulation**: 20일 내 100% 이상 급등 + 거래량 5배 폭증 종목 제외



## 🔜 Next Steps

* **Step 6**: Portfolio Optimization (MVO, Risk Parity)
* **Step 7**: Backtesting & Simulation