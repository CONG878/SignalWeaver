# SignalWeaver

**SignalWeaver**는 시계열 데이터(Time-series)로부터 유의미한 투자 신호(Signal)를 직조(Weave)해내는 **MLOps 기반의 정량 투자 파이프라인**입니다.

## 🏗 Architectural Blueprint

### 1. 핵심 설계 원칙 (Design Principles)

1. **계약 기반 분리 (Decoupling by Contract)**: `ModelBase` 인터페이스를 통해 모든 모델 객체의 **fit/predict/save** 메서드 호출 규약을 통일
2. **검증 전략 고정 (Fixed Validation Strategy)**: `WalkForwardTrainer`가 Walk-Forward 검증 로직을 전담
3. **데이터 스키마 고정**: 모든 단계의 <a href="./docs/data_schema.md" target="_blank">데이터 입/출력 스키마</a>(`data_schema.md`)를 고정하여 모듈 간 데이터 오염 방지

### 2. 파이프라인 <a href="./docs/dependency_diagram.svg" target="_blank">의존성 다이어그램</a>

`docs/dependency_diagram.svg` 참조

## 💻 Core Implementation

| 모듈 | 파일 | 담당 역할 |
| :--- | :--- | :--- |
| **Model Interface** | `src/models/base.py` | 모든 예측 모델의 **fit/predict/save** 규격 정의 |
| **Validation Engine** | `src/modeling/trainer.py` | 시계열 **Walk-Forward** 학습 로직 전담 |
| **Model Artifact** | `src/models/artifact.py` | 모델 메타데이터 및 버전 **Registry** 관리 |
| **Data Collector** | `src/data_loader/collector.py` | pykrx API 원시 데이터 수집 전담 |
| **Technical Indicators** | `src/features/technical.py` | 기술적 지표 계산 (RSI, MACD 등) |
| **First-Stage Model** | `src/models/lightgbm_model.py` | `ModelBase`를 구현한 LightGBM 예측기 |
| **Selection Logic** | `src/universe/select_universe.py` | 학습용/운영용 **유니버스 선정 로직** |

## 📂 Directory Structure

```bash
SignalWeaver/
├── config/                   # 설정 및 하이퍼파라미터
├── data/                     # 데이터 저장소
│   ├── 01_raw/                  # 원천 데이터 (pykrx CSV)
│   ├── 03_processed/            # 학습용 통합 데이터셋 (Parquet)
│   ├── 04_models/               # 모델 아티팩트 & Registry
│   └── 05_results/              # 최종 산출물
├── docs/                     # 아키텍처 문서
│   ├── data_schema.md           # 데이터 스키마 (계약 문서)
│   └── dependency_diagram.svg   # 의존성 다이어그램
├── scripts/                  # 실행 가능한 파이프라인 스크립트
├── src/                      # 소스 코드 모듈
│   ├── data_loader/             # 데이터 수집 (pykrx)
│   ├── features/                # 기술적 지표 계산
│   ├── modeling/                # 학습 루프 및 평가 로직
│   ├── models/                  # 모델 구현체 (Base, LightGBM)
│   └── universe/                # 종목 선정 및 리스크 관리
└── README.md
```

## 🚀 Getting Started

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 파이프라인 실행 순서

전체 파이프라인은 Jupyter Notebook을 순차적으로 실행하여 재현할 수 있습니다.

#### **Step 1: 원시 데이터 수집**
```bash
jupyter notebook 01_collect_data.ipynb
```
- **책임**: pykrx API를 통한 KRX 원시 시세 다운로드만 수행
- **출력**: `data/01_raw/krx_prices_YYYYMMDD/*.csv`
- **특징**:
  - Feature 계산 없음 (API가 주는 것만 받음)
  - 로그 스케일 랜덤 대기로 서버 차단 회피
  - 컬럼명 표준화 (한글 → 영문)

#### **Step 2: Feature 및 Dataset 구축**
```bash
jupyter notebook 02_build_dataset.ipynb
```
- **책임**: 
  - 기술적 지표 계산 (MA, RSI, MACD, Bollinger 등)
  - Universe Meta 생성 (liquidity_score, risk_composite)
  - 단일 통합 데이터셋 생성
- **출력**: `data/03_processed/dataset_YYYYMMDD.parquet`
- **특징**:
  - Feature 준비 기간(60일) 제거
  - **Target은 생성하지 않음** (03단계에서 필요 시 생성)
  - Parquet 포맷으로 저장 (빠른 I/O)

#### **Step 3: 학습 및 예측** (예정)
```bash
jupyter notebook 03_train_predict.ipynb
```
- **책임**:
  - Target 생성 (예측 기간 정의 후)
  - Walk-Forward 학습
  - 후보 종목 선정
- **출력**: `data/05_results/`

---

## 🎯 데이터 명명 규칙

### Feature Prefix
모든 Feature는 `feature_` prefix를 사용합니다:
```python
feature_ma_5, feature_ma_20, feature_ma_60    # 이동평균
feature_rsi_14                                # RSI
feature_macd, feature_macd_signal             # MACD
feature_bollinger_upper, feature_bollinger_middle, feature_bollinger_lower
```

### Meta Columns
Universe 선정용 메타 지표:
```python
liquidity_score      # 유동성 점수 (20일 평균 거래대금)
risk_composite       # 복합 리스크 점수
is_suspended         # 거래정지 플래그
is_delisted          # 상장폐지 플래그
```

---

## 🛠 Tech Stack

- **Language**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: LightGBM
- **Format**: Parquet (Snappy compression)
- **Data Source**: pykrx (KRX API)

## 📝 License

This project is for portfolio purposes.