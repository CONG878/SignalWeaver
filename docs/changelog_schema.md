# Schema Changelog (Updated v3.9.1)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.9.1] - 2026-03-09

### 🔵 PATCH Changes — log_return_1d 역산 보정 (사다리꼴 적분)

변경 범위: **04단계 (Forecasts), 05단계 (Universe)**
01~03단계 스키마 및 산출물 변경 없음.

---

#### 1. log_return_1d 역산 로직 개선

**배경:**
v3.9.0에서 도입된 `log_return_1d` 모드의 단순 누적(cumsum) 방식은 이산 데이터의 특성상 오차가 발생하기 쉽습니다. 이를 보정하기 위해 사다리꼴 적분(Trapezoidal Rule) 근사법을 도입합니다.

**수정 내용:**
- **역산 공식 변경**: `y(t+k) = y(t) + ΣΔy_i + (Δy(t) - Δy(t+k))/2`
- **앵커 컬럼 추가**: 사다리꼴 보정의 앵커가 되는 `Δy(t)`(당일 실측 등락률)를 `log_close_ref`에 포함하여 05단계에 전달하도록 수정하였습니다.

---

## [3.9.0] - 2026-03-09

### 🟢 MINOR Changes — log_return_1d 타겟 도입 및 log_return 폐기

변경 범위: **02단계 (Feature), 03단계 (Training), 04단계 (Forecasts), 05단계 (Universe)**
01단계 스키마 변경 없음.

---

#### 1. log_return_1d 모드 신설 및 타겟 스키마 변경

**배경:**
기존 누적 수익률 방식의 불안정성을 해소하기 위해 1일 단위 등락률(`log1p(change_pct)`)을 직접 예측하는 모드를 추가합니다.

**수정 내용:**
- **신규 타겟**: `target_log_return_1d_h{n}` (v3.9.0 신규).
- **정식 폐기**: 이전 버전에서 deprecated 되었던 `target_log_return_h{n}` (누적 로그 수익률) 타겟을 물리적으로 삭제하였습니다.
- **보고 지표 스케일 통일**: 모델이 예측한 raw log return 스케일 그대로 `val/test_predictions.parquet`에 저장하되, 평가 시 변환 과정을 거치도록 구조를 개선했습니다.

#### 2. 모델 평가 함수 시그니처 확장

**수정 내용:**
- `evaluate_model_accuracy()` 및 `select_investment_universe()`에 `target_columns` 및 `log_close_ref` 인자를 추가하여 모드별 가변적인 평가 기준을 지원합니다.

---

## [3.8.1] - 2026-03-05

### 🔵 PATCH Changes — API Fallback 및 datetime 타입 버그 수정

변경 범위: **98단계, 02단계, 04단계, src/data_loader/collector.py**
01, 03, 05단계 스키마 변경 없음.

---

#### 1. 매크로 및 종목 리스트 수집 Fallback 강화

**배경:**
FDR API 장애 시 파이프라인 전체가 중단되는 현상을 방지하기 위해 로컬 CSV 대체 경로를 확보했습니다.

**수정 내용:**
- **매크로 지표 (98단계)**: API 실패 시 `data/99_meta/{indicator}.csv`에서 데이터를 복구하는 헬퍼 셀을 추가했습니다.
- **코스피 판별 (02단계)**: `StockListing` 실패 시 로컬 `stock_list.csv` 또는 `ticker_master`를 참조하는 3단계 Fallback 체계를 구축했습니다.

#### 2. 04단계 numpy.datetime64 속성 오류 수정

**배경:**
미래 날짜 생성 시 `values` 속성 사용으로 인해 데이터 타입이 `numpy.datetime64`로 고정되어 `.weekday()` 메서드 호출 시 `AttributeError`가 발생하는 버그를 수정했습니다.

**수정 내용:**
- `future_dates` 생성 시 `.values`를 제거하고 `pd.Series` 타입을 유지하여 인덱싱 결과가 항상 `pd.Timestamp`가 되도록 보장하였습니다.

---

## [3.8.0] - 2026-02-28

### 🟢 MINOR Changes — MLP 모델 도입 및 앙상블 확장성

변경 범위: **03단계 (Training), 03b단계 (Ensemble), 04단계 (Forecasts), src/utils, src/models**
01, 02, 05단계 스키마 변경 없음.

---

#### 1. MLP Multi-output 모델 신설
- **내용**: PyTorch 기반의 단일 네트워크로 h1~h5를 동시 출력하는 `MLPModel`을 도입했습니다.

#### 2. 앙상블 확장성 — 모델 조합 동적 지정
- **내용**: `active_model`에 `lgbm+rf+mlp`와 같이 `+` 구분자를 사용하여 임의의 조합을 지정하고 SLSQP로 가중치를 최적화하도록 개선했습니다.

#### 3. 04단계 예측 실패 종목 건너뜀
- **내용**: 특정 종목의 피처 오류(`inf` 등) 발생 시 전체 중단 대신 해당 종목만 `skipped_tickers`에 기록하고 다음 종목으로 진행하도록 예외 처리를 추가했습니다.

---

## [3.7.2] - 2026-02-25

### 🔵 PATCH Changes — 비현실적 수익률 거래 필터링
- **내용**: `strategy.max_daily_return` 설정을 도입하여 일평균 수익률 상한을 초과하는 급등주 거래를 투자 후보에서 제외합니다.

---

## [3.7.1] - 2026-02-24

### 🔵 PATCH Changes — 04단계 버그 수정 및 97단계 신설
- **내용**: 04단계 피처 스키마를 v3.6.0에 맞게 동기화하고, 매크로 지표 미래값 추정(97단계) 결과를 Recursive Extension에 반영하도록 수정했습니다.

---

## [3.7.0] - 2026-02-22

### 🟢 MINOR Changes - log_close 롤백 + Embargo Gap
- **내용**: 타겟을 `log_close`로 롤백하고, look-ahead 편향 방지를 위한 Embargo Gap 로직을 `WalkForwardTrainer`에 도입했습니다.

---

## [3.6.0] - 2026-02-21

### 🟢 MINOR Changes - Scale-Invariant Features & IC Evaluation
- **내용**: 이격도/무차원 지표 중심의 피처 개편, 매크로 피처 통합(98단계), IC/ICIR 지표 기반 평가 및 가중치 최적화 체계를 수립했습니다.

---

## [3.5.0] - 2026-02-20

### 🟢 MINOR Changes - Training Pipeline Improvement
- **내용**: 2-Fold Walk-Forward 구조를 도입하여 검증셋과 테스트셋을 물리적으로 분리했습니다.

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification
- **내용**: RandomForest 멀티아웃풋 모델 및 앙상블 학습 단계를 추가했습니다.

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization
- **내용**: 단계별 독립 폴더 구조 및 `ProjectPaths`를 통한 경로 관리 중앙화를 실시했습니다.

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes
- **내용**: Multi-Horizon 버그 수정 및 유동성 점수 산출 방식(최근 20일 평균)을 개선했습니다.

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes
- **내용**: 04단계(미래 예측) 및 05단계(유니버스 선정) 기능을 추가했습니다.

---

## [3.1.x] / [3.0.0]

- **3.1.1/3.1.0**: Target 생성 위치 02단계 이동 및 Multi-Horizon Direct Forecasting 도입.
- **3.0.0**: Target 생성 위치 변경 (01 → 02).

---

## [2.0.0] / [1.0.0]

- **2.0.0**: Feature Prefix 통일 (`feature_` 접두어).
- **1.0.0**: Initial Release (수집 → Feature → LightGBM 학습).

---

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **3.9.1** | 2026-03-09 | 🔵 PATCH | log_return_1d 역산: 사다리꼴 적분 보정 (오차 감소) |
| **3.9.0** | 2026-03-09 | 🟢 MINOR | log_return_1d 타겟 신규 추가 + log_return 정식 폐기 |
| **3.8.1** | 2026-03-05 | 🔵 PATCH | API Fallback 강화 및 datetime 타입 버그 수정 |
| **3.8.0** | 2026-02-28 | 🟢 MINOR | MLP 모델 도입 + 앙상블 조합 동적 지정 + 04단계 건너뜀 처리 |
| **3.7.2** | 2026-02-25 | 🔵 PATCH | 비현실적 수익률 필터링 (max_daily_return) |
| **3.7.1** | 2026-02-24 | 🔵 PATCH | 04단계 피처 스키마 동기화 + 97단계 신설 |
| **3.7.0** | 2026-02-22 | 🟢 MINOR | log_close 롤백 + Embargo Gap |
| **3.6.0** | 2026-02-21 | 🟢 MINOR | Scale-invariant 피처 + IC 평가 + 매크로 통합 |
| **3.5.0** | 2026-02-20 | 🟢 MINOR | 2-Fold 구조 도입 |
| **3.0.0** | 2026-01-18 | 🔴 MAJOR | Target 생성 및 위치 변경 |

---

**Last Updated**: 2026-03-09
**Schema Version**: 3.9.1
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team