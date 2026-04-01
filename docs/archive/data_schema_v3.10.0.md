# 📄 Data Schema Definition (v3.10.0)

본 스키마는 SignalWeaver 프로젝트의 데이터 계약을 정의합니다.

---

## 📌 Schema Version & Metadata

| 속성 | 값 |
|------|-----|
| **Schema Version** | `3.10.0` |
| **Last Updated** | 2026-03-16 |
| **Latest Changes** | 사다리꼴 모듈화, 03b log_return_1d 버그·취약점 수정, pred_log_return 스키마 분리 |
| **Compatibility** | `target_type: "log_return"` 사용 불가 (ValueError). `"log_close"` 또는 `"log_return_1d"` 사용 필수 |

---

## 🔄 최근 변경 이력 요약

### v3.10.0 (2026-03-16) - 🟢 MINOR
- **`src/utils/trapezoidal.py` 신설**: `trapezoid_log_close()` 함수로 사다리꼴 역산 수식 단일화. 03·04·05단계 및 Trainer에서 중복 정의된 수식을 모두 이 함수 호출로 교체.
- **03b `log_return_1d` 버그 수정**: `target_prefix`를 `target_type`에 따라 동적으로 결정. `log_return_1d` 모드에서 `pred_cols`가 빈 리스트가 되던 문제 해결.
- **03b `base_canonical` 취약점 수정**: `_build_ensemble_df()` 헬퍼로 date·ticker·fold·true_ 컬럼만 명시적으로 추출해 앙상블 예측 DataFrame 구성. 첫 번째 모델의 메타 컬럼이 암묵적으로 포함되던 문제 해결.
- **`pred_log_return` 스키마 분리**: `future_forecasts.parquet` 공식 스키마에서 `pred_log_return` 컬럼 제거. `config.yaml`에 `debug.save_raw_predictions: true` 추가 시에만 포함.
- **`risk_composite` 역할 분리**: `feature_risk_composite`(모델 학습 피처)와 `risk_composite`(운영 메타) 두 컬럼 동시 생성.
- **`feature_cols` 예외 처리 제거**: 03단계에서 `risk_composite` 예외 제거, `feature_` 접두사만으로 자동 인식.
- **Directional Accuracy 추가**: `evaluate_model_accuracy()`에 방향성 일치율 산출. Excel 리포트에 `방향성정확도` 컬럼 추가.
- **Top-k Precision 추가**: 05단계 노트북 신규 셀. K=10·20·50·100 및 전체 후보 기준 정밀도와 기준선 출력.

### v3.9.2 (2026-03-16) - 🔵 PATCH
- 위험도 지표 정의 통일, 리포트 개편, 필터 통계 저장, 노트북 구조·문서화 패치.

### v3.9.1 (2026-03-09) - 🔵 PATCH
- **log_return_1d 역산 보정**: cumsum 방식에서 사다리꼴 적분 보정으로 변경.
- **log_close_ref 앵커 추가**: 05단계 평가에 `target_log_return_1d` 컬럼 추가 참조.

### v3.9.0 (2026-03-09) - 🟢 MINOR
- **log_return_1d 타겟 신규 추가**: 1일 당일 등락률 예측 모드 추가.
- **log_return 정식 폐기**: 누적 로그 수익률 타겟 완전 삭제.

### v3.8.1 (2026-03-05) - 🔵 PATCH
- API Fallback 강화, datetime 버그 수정.

### v3.8.0 (2026-02-28) - 🟢 MINOR
- MLP Multi-output 모델 신설, 앙상블 조합 동적 지정, 04단계 건너뜀 처리.

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
├── lightgbm/
│   ├── v1_lgbm_{YYYYMMDD}_{hash}.pkl
│   ├── registry.json
│   ├── val_predictions.parquet
│   └── test_predictions.parquet
├── randomforest/  { 동일 구조 }
├── mlp/           { 동일 구조 }
└── lgbm+rf/       { 동일 구조 }

# 04단계: 미래 예측
data/04_forecasts/{YYYYMMDD}/
├── lightgbm/future_forecasts.parquet
└── lgbm+rf/future_forecasts.parquet

# 05단계: 유니버스 선정
data/05_universe/{YYYYMMDD}/
└── {folder_name}/
    ├── universe_full.parquet
    ├── universe_candidates.parquet
    ├── investment_report.csv / .xlsx
    └── filter_statistics.json

# 전역 메타 데이터
data/99_meta/
├── krx_calendar.csv
├── macro_regime.parquet
└── macro_regime_forecast.parquet
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

### 4.1 Target 컬럼

```python
# 02단계에서 생성되는 기준값
'target_log_close'        # log(close)         — log_close 모드용
'target_log_return_1d'    # log1p(change_pct)  — log_return_1d 모드용 (v3.9.0)
                          # 사다리꼴 역산의 Δy(t) 앵커로도 사용 (v3.9.1)

# Trainer 내부에서 동적 생성 (horizon별)
'target_log_close_h1'          ~ 'target_log_close_h5'
'target_log_return_1d_h1'      ~ 'target_log_return_1d_h5'

# [REMOVED v3.9.0] — 완전 삭제, 사용 불가
# 'target_log_return_h{n}'
```

---

## 📌 5. Step 3 (Training) - 모델 & 예측 스키마

### 5.1 val/test_predictions.parquet 스키마

**`log_close` 모드:**
```python
columns = [
    'date', 'ticker', 'fold',
    'pred_target_log_close_h1', ~ 'pred_target_log_close_h5',
    'true_target_log_close_h1', ~ 'true_target_log_close_h5',
]
```

**`log_return_1d` 모드:**
```python
columns = [
    'date', 'ticker', 'fold',
    'pred_target_log_return_1d_h1', ~ 'pred_target_log_return_1d_h5',
    'true_target_log_return_1d_h1', ~ 'true_target_log_return_1d_h5',
]
# 보고 지표 산출 시 trapezoid_log_close() 적용 (Trainer 내부, 외부 비노출)
```

---

## 📌 6. Step 3b (Ensemble) - 앙상블 가중치 최적화

- 입력: `val_predictions.parquet` (검증 폴드 전용)
- 목표: `-IC(Spearman)` 최소화 (SLSQP, 합계 1 제약)
- **v3.10.0**: `target_prefix`를 `target_type`에 따라 동적 결정. `_build_ensemble_df()` 헬퍼로 안전한 컬럼 구성.

```python
# v3.10.0 수정 후 — 명시적 DataFrame 구성
keep_cols = ['date', 'ticker', 'fold'] + true_cols
result = base_df[keep_cols].iloc[mask].copy().reset_index(drop=True)
for i, col in enumerate(pred_cols):
    result[col] = blended[:, i]
```

---

## 📌 7. Step 4 (Forecasts) - 미래 예측 결과 스키마

### 7.1 Recursive Extension 역산 로직 (v3.10.0 모듈화)

```python
from src.utils.trapezoidal import trapezoid_log_close

# ── log_close 모드 ──────────────────────────────────────────────────────
pred_log_close = model.predict(X)
pred_close     = exp(pred_log_close)

# ── log_return_1d 모드 (v3.9.1 ~ / v3.10.0 모듈화) ─────────────────────
pred_log_close = trapezoid_log_close(
    log_close_base,   # y(t): 청크 시작 기준 로그 가격
    cum_log_return,   # Σ Δy_i, i=0..h-1
    delta_y_t,        # Δy(t): 앵커
    pred_delta_h,     # Δy(t+h): 현재 h 시차 예측값
)
pred_close = exp(pred_log_close)
```

### 7.2 future_forecasts.parquet 스키마 (✨ v3.10.0 갱신)

```python
# 공식 컬럼 (항상 포함)
columns = [
    'date',             # 예측 대상 날짜
    'ticker',           # 종목 코드
    'horizon',          # 예측 시차 (1~5)
    'chunk_idx',        # Recursive Extension chunk 번호
    'pred_log_close',   # 예측 로그 종가
    'pred_close',       # 예측 종가 (원화)
]

# 디버그 컬럼 (config.yaml debug.save_raw_predictions: true 시에만 포함)
# 'pred_log_return'   # 당일 등락률 로그값 (raw 예측, 참고용)
```

---

## 📌 8. Step 5 (Universe) - 최종 선정 결과 스키마

- `universe_full.parquet`: `ticker`, `name`, `accuracy_score`, `profitability_score`, `risk_composite`, `composite_score`, `selected`
- `universe_candidates.parquet`: `ticker`, `name`, `composite_score`, `rank`, `recommendation`

---

## 📌 9. Step 99_meta - 전역 메타 데이터 스키마

### 9.1 `src/utils/trapezoidal.py` (✨ v3.10.0 신설)

```python
from src.utils.trapezoidal import trapezoid_log_close

# y(t+h) = y(t) + Σ Δy_i + (Δy(t) - Δy(t+h)) / 2
pred_log_close = trapezoid_log_close(log_close_base, cum_pred, delta_y_t, delta_y_h)
```

사용 위치: `src/modeling/trainer.py`, `03_train_predict.ipynb`, `03b_train_ensemble.ipynb`, `04_forecast_future.ipynb`, `src/universe/select_universe.py`

---

## 📌 10. 설정 파일 스키마 (config.yaml)

```yaml
training:
  target_col_name: "target_log_return_1d"
  target_type: "log_return_1d"   # "log_close" 또는 "log_return_1d"
  horizons: [1, 2, 3, 4, 5]

active_model: "mlp+lgbm"

# ✨ v3.10.0 신설: 디버그용 raw 예측값 저장 플래그
debug:
  save_raw_predictions: false    # true 시 pred_log_return 컬럼 포함
```

---

## 📌 11. 호환성 노트

### v3.9.x → v3.10.0 마이그레이션

**재실행 필요 (스키마 변경):**
- `log_return_1d` 모드: 03b단계부터 재실행 권장 (버그 수정 효과 적용)
- 04단계 산출물: `pred_log_return` 컬럼이 기본 제거됨. 하위 스텝에서 이 컬럼을 직접 참조하는 코드가 있다면 수정 필요.

**재실행 불필요:**
- `log_close` 모드 운용 중인 경우: 03b의 `target_prefix` 결정 로직이 동일하므로 결과 변화 없음.
- 01·02단계: 스키마 변경 없음.

---

## 🔍 스키마 버전 관리 정책

```
MAJOR: 근본 구조 변경 (하위 호환 불가)
MINOR: 기능 추가 / 파이프라인 개선
PATCH: 버그 수정
```

| Version | Date | Type | 주요 변경 사항 |
|---------|------|------|----------------|
| **3.10.0** | 2026-03-16 | 🟢 MINOR | 사다리꼴 모듈화 + 03b 버그·취약점 + pred_log_return 분리 |
| **3.9.2** | 2026-03-16 | 🔵 PATCH | 위험도 지표 통일 + 리포트 개편 + 필터 통계 저장 |
| **3.9.1** | 2026-03-09 | 🔵 PATCH | log_return_1d 역산: 사다리꼴 적분 보정 |
| **3.9.0** | 2026-03-09 | 🟢 MINOR | log_return_1d 타겟 신규 추가 + log_return 정식 폐기 |
| **3.8.1** | 2026-03-05 | 🔵 PATCH | API Fallback 및 datetime 타입 버그 수정 |
| **3.8.0** | 2026-02-28 | 🟢 MINOR | MLP 모델 + 앙상블 동적 조합 + 04단계 건너뜀 처리 |

---

**Last Updated**: 2026-03-16
**Schema Version**: 3.10.0
**Status**: ✅ Stable
**Maintained by**: SignalWeaver Team
