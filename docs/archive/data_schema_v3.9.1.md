# 📄 Data Schema Definition (v3.9.1)

본 스키마는 SignalWeaver 프로젝트의 데이터 계약을 정의합니다.

---

## 📌 Schema Version & Metadata

| 속성 | 값 |
|------|-----|
| **Schema Version** | `3.9.1` |
| **Last Updated** | 2026-03-09 |
| **Latest Changes** | log_return_1d 타겟 신규 추가, 사다리꼴 적분 보정, API Fallback 강화 및 버그 수정 |
| **Compatibility** | `target_type: "log_return"` 사용 불가 (ValueError). `"log_close"` 또는 `"log_return_1d"` 사용 필수 |

---

## 🔄 최근 변경 이력 요약

### v3.9.1 (2026-03-09) - 🔵 PATCH
- **log_return_1d 역산 보정**: cumsum 방식에서 사다리꼴 적분 보정(`y(t+h) = y(t) + cumsum_h + (Δy(t) − Δy(t+h)) / 2`)으로 변경하여 오차 감소.
- **log_close_ref 앵커 추가**: 05단계 등에서 평가를 위한 기준 DataFrame에 `target_log_return_1d` 컬럼 추가 참조.

### v3.9.0 (2026-03-09) - 🟢 MINOR
- **log_return_1d 타겟 신규 추가**: 1일 당일 등락률 예측을 위한 모드 추가.
- **log_return 정식 폐기**: 누적 로그 수익률 타겟 완전 삭제 및 사용 금지.
- **보고 지표 스케일 통일**: 모델 출력 공간과 관계없이 지표(RMSE/IC) 계산 시 스케일 변환 적용.

### v3.8.1 (2026-03-05) - 🔵 PATCH
- **API Fallback 강화**: `fdr.DataReader()` (매크로 지표) 실패 시 로컬 CSV(`data/99_meta/*.csv`) 대체 지원. `fetch_company_info()` (코스피 판별) 3단계 Fallback 체계 도입.
- **datetime 버그 수정**: 04단계 미래 예측 시 `numpy.datetime64`에 의한 `.weekday()` AttributeError 수정. 
- **예외 메시지 개선**: `get_ticker_universe()` 실패 경위 명시적 전달 (`RuntimeError`).

### v3.8.0 (2026-02-28) - 🟢 MINOR
- **MLP Multi-output 모델 신설**: `src/models/mlp_model.py`. PyTorch 기반 단일 네트워크로 h1~h5 동시 출력.
- **앙상블 조합 동적 지정**: `active_model`에 `+` 구분자로 조합 지정. `"lgbm+rf"`, `"lgbm+rf+mlp"` 등.
- **04단계 건너뜀 처리**: 종목별 예측 루프 `try/except` 감싸기 및 `skipped_tickers` 수집.

### v3.7.2 (2026-02-25) - 🔵 PATCH
- **비현실적 수익률 필터링**: `config.yaml`에 `strategy.max_daily_return` 추가.

### v3.7.1 (2026-02-24) - 🔵 PATCH
- **97단계 신설**: `97_forecast_macro.ipynb` — 매크로 지표 미래값 추정.

### v3.7.0 (2026-02-22) - 🟢 MINOR
- **log_return deprecated → log_close 롤백**: 타겟 컬럼명 `target_log_close_h{n}` 복원. Embargo Gap 도입.

---

## 📌 1. 파일 저장 규칙 / 포맷

### 1.1 기본 포맷

| 단계 | 폴더 | 포맷 |
|------|------|------|
| **01단계 (Raw)** | `data/01_raw/{date}/` | CSV + 통합 Parquet |
| **02단계 (Processed)** | `data/02_processed/{date}/` | Parquet + 선택적 CSV |
| **03단계 (Training)** | `data/03_training/{date}/{folder_name}/` | Parquet + 모델별 폴더 |
| **04단계 (Forecasts)** | `data/04_forecasts/{date}/{folder_name}/` | Parquet + 선택적 CSV |
| **05단계 (Universe)** | `data/05_universe/{date}/{folder_name}/` | Parquet + CSV + JSON |
| **99_meta** | `data/99_meta/` | Parquet + CSV |

### 1.2 파일 네이밍 규칙


```

# 01단계: 원시 데이터

data/01_raw/{YYYYMMDD}/
├── krx_prices_{YYYYMMDD}.parquet
├── ticker_master_{YYYYMMDD}.csv
└── csv/{종목명}.csv

# 02단계: 전처리 데이터

data/02_processed/{YYYYMMDD}/
├── dataset.parquet
└── csv/{종목명}.csv

# 03단계: 학습/검증/테스트 예측

data/03_training/{YYYYMMDD}/
├── lightgbm/                          # 단일 모델: canonical 정칭
│   ├── v1_lgbm_{YYYYMMDD}_{hash}.pkl
│   ├── registry.json
│   ├── val_predictions.parquet
│   └── test_predictions.parquet
├── randomforest/  { 동일 구조 }
├── mlp/           { 동일 구조 }
└── lgbm+rf/       { 동일 구조 }       # 앙상블은 short 약칭 조합

# 04단계: 미래 예측

data/04_forecasts/{YYYYMMDD}/
├── lightgbm/future_forecasts.parquet
└── lgbm+rf/future_forecasts.parquet

# 05단계: 유니버스 선정

data/05_universe/{YYYYMMDD}/
└── {folder_name}/
├── universe_full.parquet
└── investment_report.xlsx

# 전역 메타 데이터

data/99_meta/
├── krx_calendar.csv
├── macro_regime.parquet               # 98단계 출력 (과거 실측)
└── macro_regime_forecast.parquet      # 97단계 출력 (미래 추정)

```

---

## 📌 2. 공통 기본 컬럼

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| **date** | datetime64 | 거래일 | 2024-01-15 |
| **ticker** | str | 종목 코드 | 005930 |
| **close** | float64 | 종가 | 70500.0 |

---

## 📌 3. Step 1 (Raw Data) - 입력 스키마

### 3.1 OHLCV 데이터

```python
dtypes = {
    'Date'  : 'datetime64[ns]',
    'Open'  : 'float64',
    'High'  : 'float64',
    'Low'   : 'float64',
    'Close' : 'float64',
    'Volume': 'float64',
}
index.name = 'ticker'

```

---

## 📌 4. Step 2 (Processed) - Feature + Target 스키마

### 4.1 Feature 카테고리

모든 학습 피처는 `feature_` 접두어를 가집니다.
`feature_cols = [c for c in df.columns if c.startswith('feature_')]`로 자동 인식.

### 4.2 Target 컬럼 (✨ v3.9.1 갱신)

```python
# 02단계에서 생성되는 기준값
'target_log_close'        # log(close)         — log_close 모드용 (기존)
'target_log_return_1d'    # log1p(change_pct)  — log_return_1d 모드용 (v3.9.0 신규)
                          #                      사다리꼴 역산의 Δy(t) 앵커로도 사용 (v3.9.1)

# Trainer 내부에서 동적 생성 (horizon별)
'target_log_close_h1'          ~ 'target_log_close_h5'          # log_close 모드
'target_log_return_1d_h1'      ~ 'target_log_return_1d_h5'      # log_return_1d 모드

# [REMOVED v3.9.0] — 완전 삭제, 사용 불가
# 'target_log_return_h{n}'  — 누적 log 수익률 타겟, 정식 폐기

```

---

## 📌 5. Step 3 (Training) - 모델 & 예측 스키마

### 5.1 폴더 구조 및 Walk-Forward 검증

* 기존 2-Fold Walk-Forward 구조 및 Embargo Gap(G) 로직 유지

### 5.2 val/test_predictions.parquet 스키마 (✨ v3.9.1 갱신)

**`log_close` 모드:**

```python
columns = [
    'date', 'ticker', 'fold',
    'pred_target_log_close_h1', ~ 'pred_target_log_close_h5',
    'true_target_log_close_h1', ~ 'true_target_log_close_h5',
]

```

**`log_return_1d` 모드 사용 시:**

```python
# pred/true: Trainer 내부 raw log return 스케일로 저장 (모델 학습 공간)
columns = [
    'date', 'ticker', 'fold',
    'pred_target_log_return_1d_h1', ~ 'pred_target_log_return_1d_h5',
    'true_target_log_return_1d_h1', ~ 'true_target_log_return_1d_h5',
]
# 보고 지표(RMSE/IC) 산출 시에는 아래 사다리꼴 변환 적용 (_evaluate 내부, 외부 노출 없음)
# v3.9.1 사다리꼴 변환: y(t+h) = y(t) + cumsum_h + (Δy(t) − Δy(t+h)) / 2

```

---

## 📌 6. Step 3b (Ensemble) - 앙상블 가중치 최적화

* 입력: `val_predictions.parquet` (검증 폴드 전용)
* 목표: `-IC(Spearman)` 최소화 (SLSQP 활용, 합계 1 제약)

---

## 📌 7. Step 4 (Forecasts) - 미래 예측 결과 스키마

### 7.1 Recursive Extension 역산 로직 (✨ v3.9.1 갱신)

```python
# ── log_close 모드 (v3.7.0~ 기본값, 권장) ──────────────────────────────
pred_log_close = model.predict(X)          # 절대 log 가격 직접 출력
pred_close     = exp(pred_log_close)

# ── log_return_1d 모드 (v3.9.0 신규, v3.9.1 사다리꼴 보정) ────────────
#
# 수정 후 (v3.9.1 사다리꼴):
#   pred_log_close_h{k} = log(close_base) + cumsum_k + (Δy(t) − Δy(t+k)) / 2
#
# 기호 정의:
#   close_base  : chunk 시작 기준가
#                 첫 chunk → 실제 close(t)
#                 이후 chunk → 직전 chunk h_max 사다리꼴 예측값
#   cumsum_k    : Σ raw_preds[i],  i=0..k-1  (예측 log return 누적합)
#   Δy(t)       : target_log_return_1d       (현재 시점 실측 log return, 앵커)
#                 04단계: 첫 chunk → 실측값, 이후 chunk → 직전 raw_preds[-1]
#   Δy(t+k)     : raw_preds[k-1]            (h_k 시점 예측 log return)
#   pred_close  : exp(pred_log_close_h{k})

# [REMOVED v3.9.0] 누적 log_return 모드 완전 삭제

```

### 7.2 future_forecasts.parquet 스키마 (✨ v3.9.1 갱신)

```python
columns = [
    'date',             # 예측 대상 날짜
    'ticker',           # 종목 코드
    'horizon',          # 예측 시차 (1~5)
    'chunk_idx',        # Recursive Extension chunk 번호
    'pred_log_close',   # 예측 로그 종가 (두 모드 모두 동일 컬럼명 통일, v3.9.1: 사다리꼴 보정 갱신)
    'pred_close',       # 예측 종가 (원화)
    # log_return_1d 모드에서만 추가:
    'pred_log_return',  # 당일 등락률 로그값 (raw 예측, 참고용)
]

```

### 7.3 예측 실패 종목 처리

종목별 루프를 `try/except`로 감싸 예측 실패 시 해당 종목을 건너뜁니다 (`skipped_tickers` 수집).

---

## 📌 8. Step 5 (Universe) - 최종 선정 결과 스키마

### 8.1 universe_full.parquet / candidates

* **universe_full**: `ticker`, `name`, `accuracy_score`, `profitability_score`, `risk_composite`, `composite_score`, `selected`
* **universe_candidates**: `ticker`, `name`, `composite_score`, `rank`, `recommendation`

### 8.2 모델 성능 평가(log_close_ref) 시그니처 (✨ v3.9.1 추가)

```python
# log_return_1d 모드 사용 시, 모델 평가 함수 내부에서 기준 DataFrame 참조 필요
# 05단계 run_facade 내부에서 조립
ref_cols = ['ticker', 'date', 'target_log_close']
if 'target_log_return_1d' in df_meta.columns:      # v3.9.1 앵커 컬럼 추가
    ref_cols.append('target_log_return_1d')
LOG_CLOSE_REF = df_meta[ref_cols].copy()           # log_close 모드: None

```

---

## 📌 9. Step 99_meta - 전역 메타 데이터 스키마

### 9.1 macro_regime.parquet (과거 실측값) & API Fallback (✨ v3.8.1 갱신)

`98_save_macro_data.ipynb` 실행 시 API (`fdr.DataReader()`) 장애 발생 대비 로컬 CSV Fallback 지원

```
Fallback CSV 경로: data/99_meta/kospi.csv, sp500.csv, usd_krw.csv, vix.csv

```

### 9.2 코스피 판별(is_kospi) Fallback (✨ v3.8.1 갱신)

02단계 `fetch_company_info()` 진행 시 3단계 Fallback 작동:

1. `fdr.StockListing('KRX')`
2. `data/01_raw/{ref_date}/stock_list.csv`
3. `data/01_raw/{ref_date}/ticker_master_{ref_date}.csv`

### 9.3 macro_regime_forecast.parquet (미래 추정값)

97단계 출력 (Damped Holt / SES). 04단계에서 과거 실측값과 concat하여 미래 기간 예측에 사용.

---

## 📌 10. 전체 파이프라인 데이터 흐름

*이전 버전과 기본 데이터 흐름 파이프라인(98 → 97 → 02 → 03 → 03b → 04 → 05) 구조 유지.*

---

## 📌 11. 설정 파일 스키마 (config.yaml) - ✨ v3.9.1 갱신

```yaml
training:
  train_end: "2025-08-14"
  valid_window_days: 60
  test_window_days: 60
  target_col_name: "target_log_close"
  target_type: "log_close"        # ✨ v3.9.0: "log_return" 사용 시 ValueError. "log_close" 또는 "log_return_1d" 지원
  horizons: [1, 2, 3, 4, 5]

  # ... lgbm_params, randomforest_params, mlp_params 설정 생략 ...

active_model: "lgbm+rf"           # 단일 모델 및 앙상블 조합 지원

strategy:
  min_hold_days: 5
  max_daily_return: 0.16          # 일평균 수익률 상한 (비현실적 거래 필터)

```

---

## 📌 12. 호환성 노트

### v3.8.0 → v3.9 마이그레이션

**비호환 (조치 필요):**

* `target_type: "log_return"` 사용 시 → `ValueError` 즉시 발생 (v3.9.0)
→ `"log_close"` 또는 `"log_return_1d"`로 변경 필요

**호환 (재실행 불필요):**

* `target_type: "log_close"` 운용 중인 경우: 모든 단계 변경 없음
* 01단계 산출물: 변경 없음

**재실행 필요:**

* `log_return_1d` 모드 신규 실험 시: 02단계부터 전체 재실행
* v3.9.0에서 `log_return_1d`를 사용 중이었다면 v3.9.1 적용 시 03~05단계 재실행 (사다리꼴 변환 보정 적용)

### v3.8.0 → v3.8.1 마이그레이션

**호환 (재실행 불필요):**

* 모든 기존 산출물 (.parquet, 모델 파일) 스키마 변경 없음
* 98단계/02단계: API/가져오기 정상 처리 시 결과 동일

**재실행 권장:**

* 98단계 이후 → 04단계: `future_dates` 타입 버그 수정 효과 적용 (04단계만 재실행 권장)

---

## 🔍 스키마 버전 관리 정책

### Semantic Versioning

```
MAJOR: 근본 구조 변경 (하위 호환 불가)
MINOR: 기능 추가 / 파이프라인 개선
PATCH: 버그 수정

```

### 버전별 변경 이력

| Version | Date | Type | 주요 변경 사항 |
| --- | --- | --- | --- |
| **3.9.1** | 2026-03-09 | 🔵 PATCH | log_return_1d 역산: cumsum → 사다리꼴 적분 보정 (오차 감소) |
| **3.9.0** | 2026-03-09 | 🟢 MINOR | log_return_1d 타겟 신규 추가 + log_return 정식 폐기 + 보고 지표 스케일 통일 |
| **3.8.1** | 2026-03-05 | 🔵 PATCH | API Fallback 및 datetime 타입 버그 수정 |
| **3.8.0** | 2026-02-28 | 🟢 MINOR | MLP 모델 + 앙상블 동적 조합 + 04단계 건너뜀 처리 |
| **3.7.2** | 2026-02-25 | 🔵 PATCH | 비현실적 수익률 필터링 (max_daily_return) |
| **3.7.1** | 2026-02-24 | 🔵 PATCH | 04단계 피처 스키마 동기화 + 97단계 신설 |
| **3.7.0** | 2026-02-22 | 🟢 MINOR | log_close 롤백 + Embargo Gap |
| **3.6.0** | 2026-02-21 | 🟢 MINOR | Scale-invariant 피처 + IC 평가 + 매크로 통합 |
| **3.5.0** | 2026-02-20 | 🟢 MINOR | 2-Fold 구조 + log_return 타겟 (현재 deprecated) |

---

**Last Updated**: 2026-03-09
**Schema Version**: 3.9.1
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team