# Schema Changelog (Updated v3.10.0)

All notable changes to the data schema will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.10.0] - 2026-03-16

### 🟢 MINOR Changes — 사다리꼴 모듈화 + pred_log_return 스키마 분리 + 앙상블 버그 수정

변경 범위: **02단계, 03단계, 03b단계, 04단계, 05단계, src/features/builder.py, src/modeling/trainer.py, src/universe/select_universe.py**  
신규 파일: **src/utils/trapezoidal.py**  
02단계 스키마 변경 있음 (feature_risk_composite 추가). 전체 재실행 필요.

---

#### 1. `src/utils/trapezoidal.py` 신설 — 사다리꼴 적분 모듈화

**배경:** v3.9.1에서 도입된 사다리꼴 역산 수식이 03단계 Trainer 내부(`_evaluate`), 03단계 노트북 finalize 셀, 04단계 Recursive Extension 루프, 05단계 `evaluate_model_accuracy()` 네 곳에 중복 정의되어 있었습니다. 수식 변경 시 네 곳을 동시에 수정해야 하는 유지보수 리스크가 있었습니다.

**수정 내용:**
- `src/utils/trapezoidal.py` 신설. `trapezoid_log_close(log_close_base, cum_pred, delta_y_t, delta_y_h)` 함수 정의.
- numpy 브로드캐스팅을 활용하므로 scalar, array, Series 모두 동일하게 처리.
- 중복 수식이 있던 네 곳 모두 이 함수 호출로 교체.

#### 2. 03b단계 `log_return_1d` 타겟 미인식 버그 수정

**배경:** v3.9.0에서 `log_return_1d` 타겟이 신설되었으나, `03b_train_ensemble.ipynb`의 `target_prefix` 결정 로직이 `target_type`과 무관하게 `'pred_target_log_close_h'`로 하드코딩되어 있었습니다. `log_return_1d` 모드로 학습한 단일 모델의 `val_predictions.parquet`를 읽으면 `pred_cols`가 빈 리스트가 되고, 이후 최적화가 의미 없는 결과를 반환했습니다.

**수정 내용:**
- `target_prefix`를 `target_type`에 따라 동적으로 결정:
  - `log_return_1d` → `'pred_target_log_return_1d_h'`
  - `log_close`     → `'pred_target_log_close_h'`
- `pred_cols`가 빈 리스트인 경우 명시적 `KeyError` 발생 및 진단 메시지 출력.

#### 3. 03b단계 `base_canonical` 취약점 수정

**배경:** 앙상블 예측 결과를 저장할 때 첫 번째 구성 모델의 DataFrame을 통째로 `copy()` 후 `pred_` 컬럼만 덮어쓰는 방식을 사용했습니다. 구성 모델 순서가 바뀌거나 단일 모델 간 스키마가 달라질 경우, 05단계에서 잘못된 메타값(`fold`, `close`, `target_*` 등)을 조용히 참조할 수 있었습니다.

**수정 내용:**
- `_build_ensemble_df()` 헬퍼 함수를 신설. `date`, `ticker`, `fold`, `true_*` 컬럼만 명시적으로 추출해 새 DataFrame을 구성하고 `pred_` 컬럼을 직접 채움.
- val과 test 두 곳 모두 동일하게 적용.

#### 4. `pred_log_return` 스키마 분리 (`04_forecast_future.ipynb`)

**배경:** `future_forecasts.parquet`에 `log_return_1d` 모드의 raw 예측값인 `pred_log_return`이 "참고용"으로 포함되어 있었습니다. 공식 산출 스키마에 디버깅용 컬럼이 혼재하면 05단계 등 하위 스텝에서 실수로 이 컬럼을 참조할 위험이 있었습니다.

**수정 내용:**
- 04단계 예측 루프의 `row` 딕셔너리 구성에서 `pred_log_return` 키를 기본 제거.
- `config.yaml`에 `debug.save_raw_predictions: true` 플래그를 추가하면 필요 시에만 포함되도록 분리.
- `future_forecasts.parquet` 공식 스키마: `date`, `ticker`, `horizon`, `chunk_idx`, `pred_log_close`, `pred_close`.

#### 5. `risk_composite` 역할 분리 (`src/features/builder.py`, `04_forecast_future.ipynb`)

**배경:** `risk_composite`가 모델 학습 피처와 Universe 선정 운영 메타의 두 역할을 `feature_` 접두사 없이 겸하고 있었습니다. 03단계 `feature_cols` 정의에서 `or c in ['risk_composite']` 예외 처리가 필요했으며, 이는 개발 초기 잔재였습니다.

**수정 내용:**
- `build_universe_meta()`에서 동일한 값으로 두 컬럼을 명시적으로 생성: `feature_risk_composite`(모델 학습 피처, `feature_` 접두사), `risk_composite`(운영 메타, 접두사 없음, Universe 선정 전용).
- `04_forecast_future.ipynb`의 `calculate_features_for_ticker()`도 동일하게 적용.
- `03_train_predict.ipynb`의 `feature_cols` 정의에서 `or c in ['risk_composite']` 예외 제거.

#### 6. Directional Accuracy 추가 (`src/universe/select_universe.py`, `05_universe_selection.ipynb`)

**배경:** 기획서 8.2에 명시된 방향성 예측 정확도가 미구현 상태였습니다.

**수정 내용:**
- `evaluate_model_accuracy()`에서 종목별 `directional_accuracy` 산출. 인접 시점 간 방향 일치율(`sign(pred 변화) == sign(true 변화)`).
- `investment_report.xlsx`의 정확도 섹션에 `방향성정확도` 컬럼 추가.

#### 7. Top-k Precision 추가 (`05_universe_selection.ipynb`)

**배경:** 기획서 8.2에 명시된 상위 추천 종목의 실현 수익 비율 지표가 미구현 상태였습니다.

**수정 내용:**
- 05단계 노트북에 6️⃣번 셀 신설. `test_predictions.parquet`만으로 계산 가능하므로 추가 데이터 수집 불필요.
- K=10, 20, 50, 100, 전체 후보 각각에 대해 Precision과 무작위 선택 기준선을 함께 출력.

---

## [3.9.2] - 2026-03-16

### 🔵 PATCH Changes — 코드·문서 정합성 패치 (스키마 변경 없음)

변경 범위: **02단계, 03단계, 05단계, 99단계, src/utils/risk.py, src/models/artifact.py**  
산출물 스키마 변경 없음. 기존 parquet·모델 파일 재실행 불필요.

---

#### 1. 위험도 지표 정의 통일 (`src/utils/risk.py`)

**배경:** 기획서의 5대 표준 지표(Volatility/Downside Risk/VaR/CVaR/MDD)와 달리
`calculate_composite_risk_score()`에 VaR 대신 Kurtosis가 포함되어 있었습니다.

**수정 내용:**
- `kurtosis_pos` (0.10) 제거 → `var_abs` (0.20) 추가
- 연쇄 조정: `cvar_abs` 0.20→0.15, `mdd_abs` 0.15→0.10 (합계 1.00 유지)

#### 2. 인간 검수 리포트 개편 (`05_universe_selection.ipynb`)

**배경:** `risk_composite_raw` 단일 점수가 5개 지표를 압축하여 정보 손실을 유발했습니다.

**수정 내용:**
- `리스크점수` 컬럼 제거
- `하방위험`, `VaR(95%)`, `CVaR(95%)` 개별 컬럼 추가
- Excel Sheet 3 위험등급 분류 기준: `리스크점수` → `변동성` 분위수(4구간)

#### 3. `filter_statistics.json` 저장 누락 수정 (`05_universe_selection.ipynb`)

**수정 내용:** 저장 셀(5️⃣)에 `json.dump(filter_stats, ...)` 블록 추가

#### 4. `99_save_trading_days` 가드 제거 및 경고 억제

**수정 내용:**
- `if __name__` 블록 제거, `update_market_calendar()` 호출을 독립 셀로 분리
- `get_calendar()` 호출을 `warnings.catch_warnings()` 컨텍스트로 감싸 경고 억제

#### 5. 수동 배치 파일 명시 (`02_build_dataset.ipynb`)

**수정 내용:** Fallback 설명 셀에 수동 준비 안내 및 코드 예시 추가

#### 6. `param_hash` 문서화 (`src/models/artifact.py`, `03_train_predict.ipynb`)

**수정 내용:**
- `save_model_artifact()` docstring에 `Notes` 섹션 추가 (hash 충돌 위험 명시)
- `03_train_predict.ipynb` 호출부에 `"hyperparameters"` 키 명시 추가

---

## [3.9.1] - 2026-03-09

### 🔵 PATCH Changes — log_return_1d 역산 보정 (사다리꼴 적분)

변경 범위: **04단계 (Forecasts), 05단계 (Universe)**

#### 1. log_return_1d 역산 로직 개선

**수정 내용:**
- **역산 공식 변경**: `y(t+k) = y(t) + ΣΔy_i + (Δy(t) - Δy(t+k))/2`
- **앵커 컬럼 추가**: 사다리꼴 보정의 앵커가 되는 `Δy(t)`를 `log_close_ref`에 포함하여 05단계에 전달.

---

## [3.9.0] - 2026-03-09

### 🟢 MINOR Changes — log_return_1d 타겟 도입 및 log_return 폐기

변경 범위: **02단계 (Feature), 03단계 (Training), 04단계 (Forecasts), 05단계 (Universe)**

#### 1. log_return_1d 모드 신설 및 타겟 스키마 변경

**수정 내용:**
- **신규 타겟**: `target_log_return_1d_h{n}` (v3.9.0 신규).
- **정식 폐기**: `target_log_return_h{n}` (누적 로그 수익률) 물리적 삭제.
- **보고 지표 스케일 통일**: raw log return 스케일 그대로 저장, 평가 시 변환.

---

## [3.8.1] - 2026-03-05

### 🔵 PATCH Changes — API Fallback 및 datetime 타입 버그 수정

---

## [3.8.0] - 2026-02-28

### 🟢 MINOR Changes — MLP 모델 도입 및 앙상블 확장성

---

## [3.7.2] - 2026-02-25

### 🔵 PATCH Changes — 비현실적 수익률 거래 필터링

---

## [3.7.1] - 2026-02-24

### 🔵 PATCH Changes — 04단계 버그 수정 및 97단계 신설

---

## [3.7.0] - 2026-02-22

### 🟢 MINOR Changes - log_close 롤백 + Embargo Gap

---

## [3.6.0] - 2026-02-21

### 🟢 MINOR Changes - Scale-Invariant Features & IC Evaluation

---

## [3.5.0] - 2026-02-20

### 🟢 MINOR Changes - Training Pipeline Improvement

---

## [3.4.0] - 2026-02-17

### 🟢 MINOR Changes - Model Diversification

---

## [3.3.0] - 2026-02-09

### 🟢 MINOR Changes - Infrastructure Modernization

---

## [3.2.1] - 2026-02-09

### 🔵 PATCH Changes

---

## [3.2.0] - 2026-02-07

### 🟢 MINOR Changes

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
| **3.10.0** | 2026-03-16 | 🟢 MINOR | 사다리꼴 모듈화 + 03b 버그·취약점 수정 + pred_log_return 분리 + risk_composite 역할 분리 + Directional Accuracy + Top-k Precision |
| **3.9.2** | 2026-03-16 | 🔵 PATCH | 위험도 지표 통일 + 리포트 개편 + 필터 통계 저장 + 노트북 구조·문서화 |
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

**Last Updated**: 2026-03-16
**Schema Version**: 3.10.0
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team
