# 📄 **data_schema.md (권장 스키마 – Initial Version)**

본 스키마는 다음 요구사항을 모두 충족하도록 구성되었다.

* LightGBM 1차 예측
* 후보 종목 선택(유동성·리스크 필터)
* GRU 2차 정밀 예측
* 운영/학습 데이터 분리
* 향후 백테스트 자동화, 모델 재현성, 데이터 버전 관리
* 시퀀스 모델(GRU)과 트리 모델(LGBM)이 **동일 원본 데이터 구조**를 공유하도록 설계

---

# 📌 1. 파일 저장 규칙 / 포맷

### **1.1 기본 포맷**

* 기본 저장 포맷: **Parquet**
* 압축: `snappy`
* 파티셔닝(선택):

  * `date` (일자 단위)
  * `ticker` (종목 단위)

### **1.2 파일 네이밍**

```
data/03_processed/{YYYYMMDD}/market_data.parquet
data/03_processed/{YYYYMMDD}/universe_candidates.parquet
data/03_processed/{YYYYMMDD}/features.parquet
data/03_processed/{YYYYMMDD}/sequences_L{seq_len}.parquet
```

### **1.3 스키마 버전 관리**

* `schema_version`: 스키마가 변경될 때마다 버전 증가
* `as_of_date`: 데이터 생성 기준일

---

# 📌 2. 공통 기본 컬럼 (모든 단계에서 사용)

| 컬럼명              | 타입     | 설명         |
| ---------------- | ------ | ---------- |
| `date`           | date   | 거래일        |
| `ticker`         | string | 종목 코드      |
| `close`          | float  | 종가         |
| `open`           | float  | 시가         |
| `high`           | float  | 고가         |
| `low`            | float  | 저가         |
| `volume`         | int    | 거래량        |
| `market_cap`     | float  | 시가총액       |
| `is_suspended`   | bool   | 거래정지 여부    |
| `is_delisted`    | bool   | 상폐 여부      |
| `as_of_date`     | date   | 데이터 생성 기준일 |
| `schema_version` | int    | 스키마 버전     |

---

# 📌 3. Label (타깃) 스키마

### **3.1 공통 라벨**

| 컬럼명             | 타입    | 설명             |
| --------------- | ----- | -------------- |
| `log_return_1d` | float | 다음날 로그수익률      |
| `return_1d`     | float | 다음날 수익률 (옵션)   |
| `direction_1d`  | int   | 방향성 라벨 (+1/-1) |

### **3.2 LightGBM용 타깃**

| 컬럼명           | 타입    | 설명                                      |
| ------------- | ----- | --------------------------------------- |
| `target_lgbm` | float | LightGBM 라벨(다음날 log_return 또는 score 변형) |

### **3.3 GRU용 타깃**

| 컬럼명          | 타입    | 설명                         |
| ------------ | ----- | -------------------------- |
| `target_gru` | float | GRU가 예측할 수익률 또는 log_return |
| `mask`       | int   | 시퀀스 패딩 여부 (0/1)            |

---

# 📌 4. Feature 스키마

### **4.1 가격 기반 기본 지표**

| 컬럼                       | 설명         |
| ------------------------ | ---------- |
| `ma_5`, `ma_20`, `ma_60` | 이동 평균      |
| `ema_12`, `ema_26`       | 지수 이동평균    |
| `volatility_20`          | 20일 롤링 변동성 |
| `log_price`              | 로그 가격      |
| `log_return`             | 로그 수익률     |
| `volume_change`          | 거래량 변화율    |

### **4.2 기술적 지표(Technical Indicators)**

| 컬럼                               | 설명         |
| -------------------------------- | ---------- |
| `rsi_14`                         | RSI        |
| `macd`                           | MACD 값     |
| `macd_signal`                    | MACD 시그널   |
| `macd_hist`                      | MACD 히스토그램 |
| `bb_upper`, `bb_mid`, `bb_lower` | 볼린저 밴드     |
| `atr_14`                         | ATR        |

### **4.3 리스크 플래그**

| 컬럼                      | 설명         |
| ----------------------- | ---------- |
| `risk_liquidity`        | 유동성 리스크    |
| `risk_suspect`          | 작전주/이상치 탐지 |
| `risk_volatility_spike` | 변동성 급등 플래그 |
| `risk_composite`        | 복합 리스크 점수  |

---

# 📌 5. Universe (후보 종목 선정용)

### LightGBM → 후보선정 단계에서 필요한 특수 컬럼

| 컬럼                | 타입    | 설명              |
| ----------------- | ----- | --------------- |
| `score_lgbm`      | float | LightGBM 예측 스코어 |
| `universe_rank`   | int   | 종목별 ranking     |
| `liquidity_score` | float | 거래대금 기반 스코어     |
| `universe_flag`   | bool  | 후보 종목 여부        |

### GRU 단계에 넘길 데이터셋 예시

* 후보 종목만 필터링
* 시퀀스 변환 전 단계
* 피처 집합은 동일

---

# 📌 6. 시퀀스 스키마(GRU 입력용)

| 컬럼           | 타입     | 설명                            |
| ------------ | ------ | ----------------------------- |
| `seq_id`     | string | `{ticker}_{date}` 형태의 시퀀스 식별자 |
| `seq_index`  | int    | 윈도우 내 위치 (0~L-1)              |
| `is_valid`   | int    | 마스크(1=유효, 0=패딩)               |
| `feature_*`  | float  | 입력 피처                         |
| `target_gru` | float  | GRU 훈련용 라벨                    |

---

# 📌 7. 모델 예측/결과 스키마

### LightGBM 예측 결과

| 컬럼              | 설명                |
| --------------- | ----------------- |
| `date`          | 예측 기준일            |
| `ticker`        | 종목                |
| `score_lgbm`    | 예측 점수             |
| `rank_lgbm`     | 순위                |
| `selected_lgbm` | 선택 여부(True/False) |

### GRU 예측 결과

| 컬럼             | 설명       |
| -------------- | -------- |
| `score_gru`    | GRU 예측 값 |
| `rank_gru`     | 최종 정밀 순위 |
| `selected_gru` | 최종 선택    |

---

# 📌 8. 백테스트/실매매용 시그널 스키마

| 컬럼                | 설명     |
| ----------------- | ------ |
| `signal_buy`      | 1/0    |
| `signal_sell`     | 1/0    |
| `weight`          | 비중     |
| `execution_price` | 체결가    |
| `position`        | 포지션 상태 |
| `pnl_daily`       | 일일 PnL |
| `pnl_cum`         | 누적 PnL |

---

# 📌 9. 운영 단계(Production) 스키마 추가 항목

| 컬럼                   | 설명              |
| -------------------- | --------------- |
| `data_version`       | 데이터셋 버전(날짜·해시)  |
| `model_version_lgbm` | LightGBM 버전     |
| `model_version_gru`  | GRU 버전          |
| `latency_ms`         | 추론 시간           |
| `trace_id`           | 전체 파이프라인 트래킹 ID |

---

# 📌 10. 스키마 변경 정책

* 새로운 피처 추가 시: `schema_version + 1`
* 삭제 시: `major_version + 1`
* 모델용/운영용 스키마는 “하위호환성”을 유지하도록 설계
* 변경 내역을 `docs/changelog_schema.md` 에 기록

---

# ✔ 결론

* 지금 설계한 스키마는 LightGBM 1차 후보 선정 → GRU 2차 정밀 예측 → 시그널 생성 → 백테스트/운영까지
  **전 과정에서 재사용 가능한 단일 통합 스키마**입니다.
* 현재 스캐폴딩/의존성 구조는 이 스키마를 모두 수용하므로 **추가 변경은 필요 없습니다.**
* 즉, 리팩토링 절차를 방해하지 않으며 확장에도 충분히 대응 가능합니다.