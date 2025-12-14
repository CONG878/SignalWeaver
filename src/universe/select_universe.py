"""
Universe selection module

- build_train_universe  : 학습용 유니버스 (생존편향 허용)
- build_production_universe : 운영/예측용 유니버스 (실매매 가능성 기준)

이 모듈은 **데이터 스키마 계약**을 강제하는 역할을 한다.
LightGBM / GRU 파이프라인 어디에서도 동일한 인터페이스로 호출 가능해야 한다.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional
import pandas as pd

# ---------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """필수 컬럼 존재 여부 확인"""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Universe schema missing columns: {missing}")


# ---------------------------------------------------------------------
# 1. 학습용 유니버스
# ---------------------------------------------------------------------

def build_train_universe(
    df: pd.DataFrame,
    *,
    min_history: int = 120,
    allow_delisted: bool = True,
) -> pd.DataFrame:
    """
    학습용 유니버스 생성

    특징
    ------
    - 상장폐지 종목 포함 가능 (생존편향 제거 목적)
    - 거래정지/작전주 여부는 **라벨로만 유지**, 제거하지 않음

    Parameters
    ----------
    df : DataFrame
        일별 주가/지표 데이터 (long format)
    min_history : int
        최소 히스토리 길이 (일 단위)
    allow_delisted : bool
        상폐 종목 포함 여부

    Returns
    -------
    DataFrame
        학습용 유니버스 (원본 컬럼 + universe_flag_train)
    """

    required = [
        "date",
        "ticker",
        "close",
        "is_delisted",
    ]
    _validate_schema(df, required)

    work = df.copy()

    # (1) 히스토리 길이 필터
    hist_cnt = (
        work.groupby("ticker")["date"]
        .count()
        .rename("history_len")
    )
    work = work.merge(hist_cnt, on="ticker", how="left")
    work = work[work["history_len"] >= min_history]

    # (2) 상폐 종목 처리
    if not allow_delisted:
        work = work[work["is_delisted"] == False]

    # (3) 학습 유니버스 플래그
    work["universe_flag_train"] = True

    return work


# ---------------------------------------------------------------------
# 2. 운영/예측용 유니버스
# ---------------------------------------------------------------------

def build_production_universe(
    df: pd.DataFrame,
    *,
    min_liquidity: float,
    max_risk_score: float = 1.0,
    require_active: bool = True,
) -> pd.DataFrame:
    """
    운영(실제 예측/매매)용 유니버스 생성

    특징
    ------
    - 거래정지, 상폐 종목 **제외**
    - 유동성 및 리스크 기준을 만족하는 종목만 선택
    - LightGBM 후보선정 단계의 입력으로 사용

    Parameters
    ----------
    df : DataFrame
        피처/리스크 플래그가 포함된 데이터
    min_liquidity : float
        최소 유동성 점수 기준
    max_risk_score : float
        허용 가능한 최대 복합 리스크 점수
    require_active : bool
        거래 가능 종목만 허용할지 여부

    Returns
    -------
    DataFrame
        운영 유니버스 (원본 컬럼 + universe_flag_prod)
    """

    required = [
        "date",
        "ticker",
        "liquidity_score",
        "risk_composite",
        "is_suspended",
        "is_delisted",
    ]
    _validate_schema(df, required)

    work = df.copy()

    # (1) 거래 가능성 필터
    if require_active:
        work = work[(work["is_suspended"] == False) & (work["is_delisted"] == False)]

    # (2) 유동성 기준
    work = work[work["liquidity_score"] >= min_liquidity]

    # (3) 리스크 기준
    work = work[work["risk_composite"] <= max_risk_score]

    # (4) 운영 유니버스 플래그
    work["universe_flag_prod"] = True

    return work


# ---------------------------------------------------------------------
# 3. 후보 종목 추출 (LightGBM 결과 기반)
# ---------------------------------------------------------------------

def select_candidates_from_scores(
    df: pd.DataFrame,
    *,
    score_col: str = "score_lgbm",
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    LightGBM 예측 결과로부터 최종 후보 종목 선택

    Parameters
    ----------
    df : DataFrame
        LightGBM score가 포함된 데이터
    score_col : str
        스코어 컬럼명
    top_k : int, optional
        상위 K개 선택
    score_threshold : float, optional
        스코어 기준값

    Returns
    -------
    DataFrame
        후보 종목 DataFrame (rank, universe_flag_candidate 포함)
    """

    if score_col not in df.columns:
        raise ValueError(f"Missing score column: {score_col}")

    work = df.copy()
    work = work.sort_values(score_col, ascending=False)

    if score_threshold is not None:
        work = work[work[score_col] >= score_threshold]

    if top_k is not None:
        work = work.head(top_k)

    work["universe_rank"] = range(1, len(work) + 1)
    work["universe_flag_candidate"] = True

    return work


# ---------------------------------------------------------------------
# 사용 예시 (문서용)
# ---------------------------------------------------------------------
"""
# 학습용
train_df = build_train_universe(raw_df, min_history=120)

# 운영용
prod_df = build_production_universe(
    feature_df,
    min_liquidity=1e8,
    max_risk_score=0.7,
)

# 후보 선정
candidates = select_candidates_from_scores(prod_df, top_k=30)
"""