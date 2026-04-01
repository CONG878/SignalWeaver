# 📈 SignalWeaver (v4.0.0)

**SignalWeaver**는 Target-Centric 정렬 방식과 Multi-Horizon Direct Forecasting, 그리고 **딥러닝 기반 시계열 예측(Sequence Modeling)**을 활용하여 미래 주가를 예측하고 최적의 투자 후보군(Universe)을 선정하는 퀀트 투자 프레임워크입니다.

단순한 가격 예측을 넘어 **수익성, 정확도, 위험도**를 종합적으로 평가하며, 비현실적인 거래를 배제하는 전략적 필터링을 통해 실제 투자 의사결정에 즉시 활용할 수 있는 인사이트를 제공합니다.

---

## ✨ 핵심 기능 (Key Features)

- **시계열 딥러닝 트랙 (Seq Model)**: 기존 Tabular 모델과 완벽히 분리된 독립적인 Sequence 파이프라인을 제공하며, 3D 텐서 기반의 `GRU` 모델을 통해 과거 데이터의 시계열적 특성을 깊이 있게 학습합니다.
- **메모리 및 I/O 최적화 (On-the-fly & FastParquet)**: 대용량 시퀀스 데이터 처리 시 메모리 폭증을 막기 위해 인덱스 기반의 On-the-fly 데이터셋을 구성하며, `fastparquet` 엔진을 도입해 3D 텐서 예측 결과의 메타데이터 충돌을 원천 차단합니다.
- **Target-Centric Walk-Forward Validation**: 시계열 데이터의 Look-ahead 편향을 원천 차단하기 위해 타겟 시차에 비례하는 Embargo Gap을 자동 적용하며, 시간 흐름에 따른 검증을 수행합니다.
- **다중 모델 및 동적 앙상블**: `LightGBM`, `RandomForest`, `MLP`, `GRU`를 지원하며, 설정 파일에 조합을 명시하면 SLSQP 알고리즘을 통해 최적의 가중치를 자동 산출합니다.
- **Recursive Extension 예측**: $h_1 \sim h_{20}$ 예측 결과를 다음 청크(Chunk)의 새로운 기준가로 삼아 미래 기간을 재귀적으로 확장 예측합니다.
  - **사다리꼴 적분 보정(Trapezoidal Rule)**: 당일 등락률(`log_return_1d`) 모드 사용 시, 이산 데이터의 누적 오차를 최소화하기 위해 사다리꼴 적분 기반의 정밀한 역산을 수행합니다.
- **Scale-Invariant Feature Engineering**: 절대 가격에 의존하지 않는 무차원 이격도(Disparity), %B, 밴드폭 등의 피처를 구성하여 종목 간 스케일 차이로 인한 왜곡을 방지합니다.

---

## 🔄 파이프라인 아키텍처 (Pipeline Flow)

SignalWeaver는 데이터 수집부터 최종 투자 리포트 생성까지 독립적인 단계(Step)로 구성되어 있습니다.

```text
[데이터 수집 및 전처리]
 00a_save_trading_days  : 영업일 캘린더 생성 (시간 독립 데이터)
  └─ 00b_save_macro_data: 글로벌 거시 경제 지표 및 Market Regime 수집
      └─ 00c_forecast_macro: 매크로 지표의 미래값 추정 (Recursive Extension 대비)
          └─ 01_collect_data: KRX 전 종목 주가 및 거래량 데이터 수집
              └─ 02_build_dataset: Scale-Invariant 기술적 지표 및 유동성/리스크 메타 생성

[모델 학습 및 예측]
 03a_train_tabular      : Tabular 기반 단일 모델 WFT 학습 (LGBM, RF, MLP)
 03b_train_ensemble     : (선택) 앙상블 가중치 최적화
 03c_train_seq          : Seq 모델 전용 학습 (GRU) - 대화형 환경
  └─ scripts/train_seq.py: Seq 모델 전용 CLI 학습 스크립트 (장시간 학습 및 메모리 안정성)
      └─ 04_forecast_future: 사다리꼴 보정 기반 미래 주가 재귀 예측 (Tabular/Seq 분기 처리)

[투자 유니버스 선정]
 05_universe_selection  : 예측 정확도(IC/RMSE/방향성), 기대 수익률, 위험도(VaR/MDD) 종합 평가
                          (투자 의사결정을 위한 상세 Excel/CSV 리포트 및 통계 JSON 출력)
```

---

## 📂 디렉토리 구조 (Directory Structure)

```text
SignalWeaver/
├── config/
│   └── config.yaml          # 프로젝트 전역 설정 (파라미터, 모델, 경로)
├── data/                    # 파이프라인 단계별 산출물 보관
│   ├── 00_prep/             # 매크로 지표, 달력 등 전처리 준비 데이터
│   ├── 01_raw/              # 수집된 원시 파켓 및 CSV
│   ├── 02_processed/        # 피처 엔지니어링 완료 데이터셋
│   ├── 03_training/         # Tabular 모델 아티팩트 및 예측 결과
│   ├── 03_seq/              # Seq(GRU) 모델 가중치(weights.pt), 체크포인트 및 예측 결과
│   ├── 04_forecasts/        # 미래 재귀 예측 결과
│   └── 05_universe/         # 최종 투자 후보군 및 Excel 리포트
├── docs/                    # 스키마 및 체인지로그 문서 (v4.0.0)
├── scripts/                 
│   └── train_seq.py         # Seq 모델 CLI 학습 스크립트
├── src/
│   ├── data_loader/         # API 수집, SeqDataset(On-the-fly) 모듈
│   ├── features/            # 기술적 지표 및 메타 빌더
│   ├── modeling/            # WalkForwardTrainer 및 SeqTrainer 모듈
│   ├── models/              # LGBM, RF, MLP, GRU 및 Ensemble 구현체
│   ├── universe/            # 수익률/위험도 평가 및 필터링 (Facade)
│   └── utils/               # 거래 전략, 최적화, ProjectPaths 모듈
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
* Tabular 트랙 실행 시: `active_model: "lgbm"` 등 명시
* Seq 트랙 실행 시: `active_seq_model: "gru"` 명시 및 `sequence` 파라미터 확인

**3. 파이프라인 실행**
번호가 매겨진 Jupyter Notebook(00a ~ 05)을 순서대로 실행합니다. 
* Seq 모델 학습은 메모리 관리를 위해 터미널에서 `python scripts/train_seq.py` 실행을 권장합니다.
* 최종적으로 `05_universe_selection.ipynb`를 실행하면 `data/05_universe/` 디렉토리에 투자 의사결정을 위한 상세 엑셀 리포트가 생성됩니다.

---

**Maintained by**: SignalWeaver Team

**Schema Version**: 4.0.0 (Confirmed)

### 💡 주요 변경 포인트 설명
* **v3.9.1 -> v4.0.0 버전 상향**: 스키마 버전을 최신 "Confirmed" 상태로 반영했습니다.
* **핵심 기능 추가**: Seq Model(GRU) 지원과 On-the-fly 및 FastParquet 적용 등 대규모 최적화 성과를 강조했습니다.
* **파이프라인 및 디렉토리 개편**: 기존 `99_meta` 등 낡은 체계를 제거하고, `00a~00c` 준비 단계, `03a/03b/03c` 분기, `scripts` 폴더 신설 등 v4.0.0의 새로운 스키마를 시각적으로 완벽히 매핑했습니다.