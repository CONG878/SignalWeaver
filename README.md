# 📈 SignalWeaver (v3.9.1)

**SignalWeaver**는 Target-Centric 정렬 방식과 Multi-Horizon Direct Forecasting을 활용하여 미래 주가를 예측하고, 최적의 투자 후보군(Universe)을 선정하는 퀀트 투자 프레임워크입니다.

단순한 가격 예측을 넘어 **수익성, 정확도, 위험도**를 종합적으로 평가하며, 비현실적인 거래를 배제하는 전략적 필터링을 통해 실제 투자 의사결정에 즉시 활용할 수 있는 인사이트를 제공합니다.

---

## ✨ 핵심 기능 (Key Features)

- **Target-Centric Walk-Forward Validation**: 시계열 데이터의 Look-ahead 편향을 원천 차단하기 위해 타겟 시차에 비례하는 Embargo Gap을 자동 적용하며, 시간 흐름에 따른 2-Fold 검증을 수행합니다.
- **다중 모델 및 동적 앙상블**: `LightGBM`, `RandomForest`, `MLP(다층 퍼셉트론)`를 지원하며, `active_model: "lgbm+rf+mlp"`와 같이 설정 파일에 조합을 명시하면 SLSQP 알고리즘을 통해 최적의 가중치를 자동 산출합니다.
- **Recursive Extension 예측**: $h_1 \sim h_5$ 예측 결과를 다음 청크(Chunk)의 새로운 기준가로 삼아 미래 기간을 재귀적으로 확장 예측합니다.
  - **사다리꼴 적분 보정(Trapezoidal Rule)**: 당일 등락률(`log_return_1d`) 모드 사용 시, 이산 데이터의 누적 오차를 최소화하기 위해 사다리꼴 적분 기반의 정밀한 역산을 수행합니다.
- **Scale-Invariant Feature Engineering**: 절대 가격에 의존하지 않는 무차원 이격도(Disparity), %B, 밴드폭 등의 피처를 구성하여 종목 간 스케일 차이로 인한 왜곡을 방지합니다.
- **견고한 데이터 수집 (Fallback System)**: 외부 API(FDR) 장애 시 로컬 백업 CSV를 자동으로 파싱하여 파이프라인의 중단을 방지합니다.

---

## 🔄 파이프라인 아키텍처 (Pipeline Flow)

SignalWeaver는 데이터 수집부터 최종 투자 리포트 생성까지 독립적인 단계(Step)로 구성되어 있습니다.

```text
[데이터 수집 및 전처리]
 98_save_macro_data     : 글로벌 거시 경제 지표 및 Market Regime 수집 (CSV Fallback 지원)
  └─ 97_forecast_macro  : 매크로 지표의 미래값 추정 (Recursive Extension 대비)
      └─ 01_collect_data: KRX 전 종목 주가 및 거래량 데이터 수집
          └─ 02_build_dataset: Scale-Invariant 기술적 지표 및 유동성/리스크 메타 생성

[모델 학습 및 예측]
 03_train_predict       : Target-Centric Walk-Forward 단일 모델 학습 (h1~h5 동시 학습)
  └─ 03b_train_ensemble : (선택) 검증 폴드 예측값을 활용한 앙상블 가중치 최적화
      └─ 04_forecast_future: 사다리꼴 보정 기반 Recursive Extension 미래 주가 예측

[투자 유니버스 선정]
 05_universe_selection  : 예측 정확도(IC/RMSE), 기대 수익률(최적 보유기간), 위험도 종합 평가
                          (Facade Pattern을 통한 평가 로직 캡슐화 및 상세 Excel/CSV 리포트 출력)

```

---

## 📂 디렉토리 구조 (Directory Structure)

```text
SignalWeaver/
├── config/
│   └── config.yaml          # 프로젝트 전역 설정 (파라미터, 모델, 경로)
├── data/                    # 파이프라인 단계별 산출물 보관
│   ├── 01_raw/              # 수집된 원시 파켓 및 CSV
│   ├── 02_processed/        # 피처 엔지니어링 완료 데이터셋
│   ├── 03_training/         # 학습 모델 아티팩트 및 검증/테스트 예측 결과
│   ├── 04_forecasts/        # 미래 재귀 예측 결과
│   ├── 05_universe/         # 최종 투자 후보군 및 Excel 리포트
│   └── 99_meta/             # 매크로 지표, 달력, KOSPI 마스터 백업
├── docs/                    # 스키마 및 체인지로그 문서 (v3.9.1)
├── src/
│   ├── data_loader/         # API 수집 및 Fallback 모듈
│   ├── features/            # 기술적 지표 및 메타 빌더
│   ├── modeling/            # WalkForwardTrainer 모듈
│   ├── models/              # LGBM, RF, MLP 및 Ensemble 구현체
│   ├── universe/            # 수익률/위험도 평가 및 필터링 (Facade)
│   └── utils/               # 거래 전략, 최적화, 경로 관리 (ProjectPaths)
└── *.ipynb                  # 단계별 실행 노트북

```

---

## 🚀 빠른 시작 (Quick Start)

**1. 환경 설정**
필요한 라이브러리를 설치합니다. (Python 3.10+ 권장)

```bash
pip install -r requirements.txt

```

**2. 설정 파일 확인 (`config/config.yaml`)**
예측 타겟 모드(`log_close` 또는 `log_return_1d`), 활성화 모델(`active_model`), 전략 수익률 상한(`max_daily_return`) 등을 설정합니다.

**3. 파이프라인 실행**
번호가 매겨진 Jupyter Notebook을 순서대로 실행합니다. 각 노트북은 이전 단계의 산출물(`.parquet`)을 자동으로 로드하여 작업을 수행합니다. 최종적으로 `05_universe_selection.ipynb`를 실행하면 `data/05_universe/` 디렉토리에 투자 의사결정을 위한 상세 엑셀 리포트가 생성됩니다.

---

## 📊 주요 설정 안내 (`config.yaml`)

* **`target_type`**:
* `"log_close"` (기본값): 종가의 로그값을 직접 예측합니다.
* `"log_return_1d"`: 당일 등락률을 예측하며, 역산 시 누적 오차 방지를 위해 사다리꼴 적분 공식이 적용됩니다.


* **`active_model`**: `"lgbm"`, `"rf"`, `"mlp"` 등의 단일 모델 약칭 또는 `"lgbm+rf"` 형식의 앙상블 조합을 지원합니다.
* **`strategy.max_daily_return`**: 일평균 기대 수익률이 상한(예: 0.16 = 16%)을 초과하는 비현실적인 급등 예측을 자동 제외하고 차선책을 찾습니다.

---

**Maintained by**: SignalWeaver Team

**Schema Version**: 3.9.1 (Stable)
