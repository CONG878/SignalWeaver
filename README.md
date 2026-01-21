# 🚀 SignalWeaver: Multi-horizon Stock Forecasting Framework

SignalWeaver는 한국 주식 시장(KRX) 데이터를 기반으로 단기 및 중장기 주가를 예측하는 **다중 시점 직접 예측(Direct Multi-step Forecasting)** 프레임워크입니다. 

단순히 다음 날의 주가를 맞추는 것을 넘어, 미래 5일(1주일)의 흐름을 직접 예측하고 이를 결합하여 장기적인 추세를 분석하는 하이브리드 전략을 사용합니다.

---

## ✨ Key Features

- **Direct Multi-step Forecasting**: 1일 단위 예측의 오차 누적 문제를 해결하기 위해 Horizon별(h1~h5) 독립 모델을 학습하는 직접 예측 방식 채택.
- **Target-Centric Alignment**: 예측 결과의 날짜가 실제 발생일과 일치하도록 데이터를 정렬($X_{t-h} \rightarrow y_t$)하여 분석 직관성 극대화.
- **Walk-Forward Validation**: 시계열 데이터의 특성을 고려한 롤링 윈도우 기반의 전진 검증 체계 구축.
- **Modular Architecture**: 데이터 수집, 전처리, 학습, 예측 프로세스가 완전히 분리된 파이프라인.

## 🛠️ Tech Stack

- **Data**: FinanceDataReader (Open Source Financial Data Reader)
- **Model**: LightGBM (Multi-output implementation)
- **Processing**: Pandas, NumPy
- **Environment**: Python 3.10+ / Jupyter Notebook

---

## 📂 Project Structure

- `01_collect_data.ipynb`: KRX 상장 종목 및 주가 데이터 수집
- `02_build_dataset.ipynb`: 기술적 지표 생성 및 **타깃(`target_log_close`)** 정의
- `03_train_predict.ipynb`: Multi-horizon 모델 학습 및 결과 생성
- `src/`: 모델 아키텍처, 트레이너, 피처 엔지니어링 모듈

```bash
SignalWeaver/
├── config/                   # 통합 설정 파일
│   └── config.yaml              # 프로젝트 전체 파라미터
├── data/                     # 데이터 저장소 (날짜별 격리)
│   ├── 01_raw/{YYYYMMDD}/       # Step 1: 원천 데이터 (Parquet + CSV + Master)
│   ├── 02_processed/{YYYYMMDD}/ # Step 2: Feature + Target 데이터셋
│   ├── 03_results/{YYYYMMDD}/   # Step 3: 예측 결과 및 리포트
│   └── 04_models/{YYYYMMDD}/    # Step 3: 모델 아티팩트 (PKL)
├── docs/                     # 아키텍처 문서
│   ├── data_schema.md           # 데이터 스키마 정의 (v3.1.1)
│   └── changelog_schema.md      # 스키마 변경 이력
├── notebooks/                # 실행 가능한 노트북 파이프라인
│   ├── 01_collect_data.ipynb    # Step 1: 데이터 수집
│   ├── 02_build_dataset.ipynb   # Step 2: 전처리 + Feature 생성
│   └── 03_train_predict.ipynb   # Step 3: Multi-horizon 학습 및 예측
├── src/                      # 소스 코드 모듈
│   ├── data_loader/             # 데이터 수집 (RawPriceCollector)
│   ├── features/                # 피처 엔지니어링 (Builder, Technical)
│   ├── modeling/                # 학습 루프 (WalkForwardTrainer)
│   ├── models/                  # 모델 래퍼 (LightGBMModel)
│   ├── universe/                # 유니버스 선정 (select_universe)
│   └── utils/                   # 유틸리티 (Config Loader)
└── README.md                 # 본 문서
```

---

## 📊 Data Policy (v3.1.1)

- **Target Origin**: 재현성 및 책임 분리를 위해 **02단계(전처리)**에서 타깃 생성 완료.
- **Horizon Strategy**: 기본 5거래일(1주일) 단위의 직접 예측 모델을 병렬로 운용.
- 상세 스키마는 [docs/data_schema.md](./docs/data_schema.md)를 참조하십시오.

## 🚀 Quick Start

1. `config/config.yaml`에서 프로젝트 기준일 및 예측 Horizon 설정.
2. 01~03단계 노트북을 순차적으로 실행.
3. `results/` 폴더에서 생성된 `predictions.parquet` 분석.

---

## 🔧 Pipeline Steps

### **Step 1: 원시 데이터 수집** (`01_collect_data.ipynb`)

**역할**: KRX 전 종목 시세 수집, 종목 마스터 생성

**설정**: `config.yaml` → `data_collection` 섹션
```yaml
data_collection:
  start_date: "20221019"
  end_date: "20260120"
  save_csv: True        # 개별 CSV 저장 (디버깅용)
  save_parquet: True    # 통합 Parquet 저장 (파이프라인용)
```

**출력**:
```
data/01_raw/{YYYYMMDD}/
├── krx_prices_{YYYYMMDD}.parquet  # 통합 시세 데이터
├── ticker_master_{YYYYMMDD}.csv   # 종목 코드-이름 매핑
└── csv/{종목명}.csv                # 개별 CSV (옵션)
```

**특징**:
- ✅ FinanceDataReader 사용 (API 호출 간소화)
- ✅ 로그 스케일 랜덤 대기로 서버 차단 회피
- ✅ 하이브리드 저장 (기계용 Parquet + 사람용 CSV)

---

### **Step 2: 데이터셋 구축** (`02_build_dataset.ipynb`)

**역할**: 기술적 지표 계산, 메타 지표 생성, **Target 생성**, 데이터 정제

**설정**: `config.yaml` → `preprocessing` 섹션
```yaml
preprocessing:
  min_history: 60              # 초기 제거 기간 (warmup)
  technical_windows: [5, 20, 60]
  rsi_period: 14
```

**처리 과정**:
```
Raw OHLCV
  ↓ [Technical Indicators]
+ feature_ma_5, feature_rsi_14, feature_macd, ...
  ↓ [Universe Meta]
+ liquidity_score, risk_composite, is_suspended, ...
  ↓ [Target Generation]
+ target_log_close = log(close)
  ↓ [History Filter]
- 초기 60일 제거 (지표 계산 준비 기간)
  ↓
dataset.parquet (학습 준비 완료)
```

**출력**:
```
data/02_processed/{YYYYMMDD}/
├── dataset.parquet  # Feature + Target 포함 (학습 준비 완료)
└── csv/{종목명}.csv # 개별 CSV (옵션)
```

**주요 컬럼**:
- **Features**: `feature_ma_5`, `feature_rsi_14`, `feature_macd`, ...
- **Meta**: `liquidity_score`, `risk_composite`, `is_suspended`, ...
- **Target**: `target_log_close` (로그 종가, 예측 대상)

---

### **Step 3: Multi-Horizon 학습 및 예측** (`03_train_predict.ipynb`)

**역할**: 
- 5일치(h1~h5) 동시 예측 모델 학습
- Walk-Forward Validation
- Target-Centric 정렬로 예측 결과 생성

**설정**: `config.yaml` → `training` 섹션
```yaml
training:
  train_end: "2025-01-22"
  horizons: [1, 2, 3, 4, 5]      # 예측 시차
  target_col_name: "target_log_close"
  valid_window_days: 60
  test_window_days: 60
  num_valid: 3                   # Walk-forward 검증 횟수
  
  lgbm_params:
    objective: "regression"
    learning_rate: 0.05
    num_leaves: 31
```

**Multi-Horizon 예측 개념**:
```
t일의 가격을 맞추기 위해:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
h=1: [t-1일 피처] → [모델 h1] → [t일 예측]
h=2: [t-2일 피처] → [모델 h2] → [t일 예측]
h=3: [t-3일 피처] → [모델 h3] → [t일 예측]
h=4: [t-4일 피처] → [모델 h4] → [t일 예측]
h=5: [t-5일 피처] → [모델 h5] → [t일 예측]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

→ 5개의 독립 모델이 동일한 정답(t일 가격) 학습
→ 1주일 앞까지 가격 예측 가능
```

**Target-Centric Alignment**:
```
예측 결과의 date 컬럼 = 실제 예측 대상일

Before (Feature-Centric):
date       | features (현재) | target (다음날)
2026-01-15 | {...}           | 102

After (Target-Centric):
date       | features (과거) | target (현재)
2026-01-16 | {...from t-1}   | 102
```

**출력**:
```
data/03_results/{YYYYMMDD}/
├── predictions.parquet  # 통합 예측 결과
│   ├── date (예측 대상 날짜)
│   ├── ticker
│   ├── pred_target_log_close_h1  # h=1 예측값
│   ├── pred_target_log_close_h2  # h=2 예측값
│   └── ... (h3, h4, h5)
└── csv/{종목명}.csv     # 종목별 예측 (디버깅용)

data/04_models/{YYYYMMDD}/
├── lightgbm_multi.pkl   # 학습된 Multi-output 모델
└── registry.json        # 모델 메타데이터
```