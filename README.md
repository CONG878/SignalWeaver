# SignalWeaver

**SignalWeaver**는 시계열 데이터(Time-series)로부터 유의미한 투자 신호(Signal)를 직조(Weave)해내는 **MLOps 기반의 정량 투자 파이프라인**입니다.

## 🏗 Architectural Blueprint

### 1. 핵심 설계 원칙 (Design Principles)
1. **설정 중앙화 (Config-Driven)**: `config/config.yaml`을 통해 모든 파이프라인의 경로, 파라미터, 기간을 제어
2. **계약 기반 분리 (Decoupling)**: 데이터 입출력 경로와 스키마를 엄격히 분리하여 모듈 간 의존성 최소화
3. **원자성 보존 (Atomicity)**: 각 실행 단계의 산출물(데이터, 모델, 결과)을 `기준일(Reference Date)` 폴더에 격리하여 관리

## 📂 Directory Structure (v2.0 Refactored)

```bash
SignalWeaver/
├── config/                   # [New] 통합 설정 파일 (config.yaml)
├── data/                     # 데이터 저장소 (날짜별 격리 저장)
│   ├── 01_raw/                  # Step 1: 원천 데이터 (Parquet + CSV + Master)
│   ├── 02_processed/            # Step 2: 학습용 Feature 데이터셋 (Parquet)
│   ├── 03_results/              # Step 3: 예측 결과 및 리포트
│   └── 04_models/               # Step 3: 모델 아티팩트 (PKL)
├── docs/                     # 아키텍처 문서
├── scripts/                  # 실행 가능한 파이프라인 스크립트
├── src/                      # 소스 코드 모듈
│   ├── data_loader/             # 데이터 수집 (RawPriceCollector)
│   ├── features/                # 피처 엔지니어링 (Builder, Technical)
│   ├── modeling/                # 학습 루프 (Trainer)
│   ├── models/                  # 모델 래퍼 (LightGBM)
│   └── utils/                   # 유틸리티 (Config Loader)
└── README.md

```

## 🚀 Pipeline Steps

#### **Step 1: 원시 데이터 수집 (`01_collect_data.ipynb`)**

* **설정**: `config.yaml`의 `data_collection` 섹션 참조
* **역할**: KRX 전 종목 시세 수집, 종목 마스터 생성
* **출력**:
  * 통합 데이터: `data/01_raw/{date}/krx_prices_{date}.parquet`
  * 디버깅용: `data/01_raw/{date}/csv/*.csv`



#### **Step 2: 데이터셋 구축 (`02_build_dataset.ipynb`)**

* **설정**: `config.yaml`의 `preprocessing` 섹션 참조
* **역할**: 기술적 지표 계산, 메타 지표 생성, 데이터 정제
* **출력**: `data/02_processed/{date}/dataset.parquet`

#### **Step 3: 학습 및 예측 (`03_train_predict.ipynb`)**

* **설정**: `config.yaml`의 `training` 섹션 참조
* **역할**:
  * T-1 시점 피처 정렬 (Data Shift)
  * Rolling Walk-Forward 학습 및 검증
  * 모델 및 예측 결과 분리 저장


* **출력**:
  * 결과: `data/03_results/{date}/predictions.parquet`

  * 모델: `data/04_models/{date}/*.pkl`
