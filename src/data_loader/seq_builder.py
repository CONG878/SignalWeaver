"""
Sequence Builder — On-the-fly 시퀀스 생성

## 핵심 변경 (v4.0.0 rev2)
기존: build_sequences()가 전체 텐서 (N, seq_len, n_features)를 미리 RAM에 적재
      → 종목 수 × 시퀀스 수만큼 메모리 폭증 → 커널 사망

변경: SeqDataset이 (ticker, start_idx) 인덱스 목록만 보관하고
      학습 배치 요청 시 원본 DataFrame에서 슬라이스를 즉석 추출
      → 메모리 = 원본 dataset.parquet 크기 + 인덱스 목록(수 MB)

split_sequences_by_date()는 날짜 경계만 산출하므로 유지.
SeqTrainer는 SeqDataset을 받아 DataLoader에 넘깁니다.

## v4.0.0
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────
# On-the-fly Dataset
# ──────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """
    시퀀스를 즉석(on-the-fly)으로 추출하는 PyTorch Dataset.

    전체 텐서를 미리 만들지 않고, (ticker, start_row_in_ticker_df) 쌍의
    인덱스 목록만 보관합니다. __getitem__ 호출 시 해당 슬라이스를 추출합니다.

    메모리 사용량:
        기존: O(N_samples × seq_len × n_features) float32
        개선: O(N_tickers × N_days × n_features) — 원본 df 크기와 동일
              + O(N_samples × 2) int — 인덱스 목록 (수 MB)

    Parameters
    ----------
    df : pd.DataFrame
        ticker, date 정렬된 전체 데이터. feature_cols + target_col 포함.
    feature_cols : List[str]
    target_col : str
    seq_len : int
    forecast_horizon : int
    stride : int
    date_col : str
    ticker_col : str
    date_filter : (start, end) or None
        이 범위의 meta date(= 시퀀스 마지막 날짜)만 포함합니다.
        None이면 전체 사용.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int,
        forecast_horizon: int,
        stride: int = 1,
        date_col: str = "date",
        ticker_col: str = "ticker",
        date_filter: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ):
        self.df              = df.reset_index(drop=True)
        self.feature_cols    = feature_cols
        self.target_col      = target_col
        self.seq_len         = seq_len
        self.forecast_horizon = forecast_horizon
        self.stride          = stride
        self.date_col        = date_col
        self.ticker_col      = ticker_col

        # feature / target numpy 뷰 (복사 없음)
        self._feat_arr = self.df[feature_cols].values.astype(np.float32)
        self._tgt_arr  = self.df[target_col].values.astype(np.float32)
        self._dates    = pd.to_datetime(self.df[date_col].values)

        # 유효 인덱스 목록 구성
        self._indices: List[Tuple[int, int]] = []  # (global_start, global_end_exclusive)
        self._meta_dates: List[pd.Timestamp] = []  # 각 샘플의 meta date (시퀀스 마지막)
        self._meta_tickers: List[str] = []

        window = seq_len + forecast_horizon

        if date_filter is not None:
            filter_start, filter_end = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
        else:
            filter_start, filter_end = None, None

        for ticker, grp in self.df.groupby(ticker_col, sort=False):
            grp_idx = grp.index.values  # global row indices
            if len(grp_idx) < window:
                continue

            for i in range(0, len(grp_idx) - window + 1, stride):
                global_start = grp_idx[i]
                global_end   = grp_idx[i + window - 1] + 1  # 마지막 행 다음 위치

                # meta date = 시퀀스 마지막 날짜 (타겟의 기준일 t)
                meta_date = self._dates[grp_idx[i + seq_len - 1]]

                # date_filter 적용
                if filter_start is not None:
                    if meta_date < filter_start or meta_date > filter_end:
                        continue

                # NaN 검사 (빠른 범위 체크)
                x_block = self._feat_arr[global_start : global_start + seq_len]
                y_block = self._tgt_arr[global_start + seq_len : global_start + window]
                if np.isnan(x_block).any() or np.isnan(y_block).any():
                    continue

                self._indices.append((global_start, global_start + window))
                self._meta_dates.append(meta_date)
                self._meta_tickers.append(ticker)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        start, end = self._indices[idx]
        x = self._feat_arr[start : start + self.seq_len]           # (seq_len, n_feat)
        y = self._tgt_arr[start + self.seq_len : end]              # (forecast_horizon,)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    def get_meta(self) -> pd.DataFrame:
        """각 샘플의 (date, ticker) 메타 정보 반환."""
        return pd.DataFrame({
            "date":   self._meta_dates,
            "ticker": self._meta_tickers,
        })


# ──────────────────────────────────────────────────────────────
# 날짜 기준 분할
# ──────────────────────────────────────────────────────────────

def split_by_date(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    forecast_horizon: int,
    stride: int,
    train_end: str,
    valid_window_days: int,
    test_window_days: int,
    embargo_gap: int,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> dict:
    """
    df의 전체 거래일 달력을 기준으로 날짜 경계를 산출하고
    train / val / test 용 SeqDataset을 반환합니다.

    stride와 무관하게 실제 거래일 기준으로 창 크기를 정확히 적용합니다.

    Returns
    -------
    dict with keys:
        ds_train, ds_val, ds_test : SeqDataset
        dates : {val_start, val_end, test_start, test_end}
    """
    all_trading_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(df[date_col].unique()))
    )
    train_end_dt = pd.to_datetime(train_end)

    # train_end 이하 마지막 거래일 인덱스
    te_pos = int(np.searchsorted(all_trading_dates, train_end_dt, side="right")) - 1
    N = len(all_trading_dates)

    embargo_end_pos = max(0, te_pos - embargo_gap)
    val_start_pos   = te_pos + 1
    val_end_pos     = min(te_pos + valid_window_days, N - 1)
    test_start_pos  = val_end_pos + 1
    test_end_pos    = min(val_end_pos + test_window_days, N - 1)

    # 유효성 검증
    if val_start_pos > N - 1 or val_start_pos > val_end_pos:
        raise ValueError(
            f"검증 기간을 확보할 수 없습니다. "
            f"train_end='{train_end}', 데이터 마지막={all_trading_dates[-1].date()}, "
            f"valid_window_days={valid_window_days}\n"
            f"train_end 이후 가용 거래일: {N - 1 - te_pos}일"
        )
    if test_start_pos > N - 1 or test_start_pos > test_end_pos:
        raise ValueError(
            f"테스트 기간을 확보할 수 없습니다. "
            f"필요: embargo({embargo_gap})+val({valid_window_days})+test({test_window_days})"
            f"={embargo_gap+valid_window_days+test_window_days} 거래일, "
            f"train_end 이후 가용: {N - 1 - te_pos}일"
        )

    train_end_date  = all_trading_dates[embargo_end_pos]
    val_start_date  = all_trading_dates[val_start_pos]
    val_end_date    = all_trading_dates[val_end_pos]
    test_start_date = all_trading_dates[test_start_pos]
    test_end_date   = all_trading_dates[test_end_pos]

    # 공통 kwargs
    ds_kwargs = dict(
        df=df, feature_cols=feature_cols, target_col=target_col,
        seq_len=seq_len, forecast_horizon=forecast_horizon, stride=stride,
        date_col=date_col, ticker_col=ticker_col,
    )

    ds_train = SeqDataset(**ds_kwargs, date_filter=(all_trading_dates[0], train_end_date))
    ds_val   = SeqDataset(**ds_kwargs, date_filter=(val_start_date, val_end_date))
    ds_test  = SeqDataset(**ds_kwargs, date_filter=(test_start_date, test_end_date))

    return {
        "ds_train": ds_train,
        "ds_val":   ds_val,
        "ds_test":  ds_test,
        "dates": {
            "train_end":  str(train_end_date.date()),
            "val_start":  str(val_start_date.date()),
            "val_end":    str(val_end_date.date()),
            "test_start": str(test_start_date.date()),
            "test_end":   str(test_end_date.date()),
        },
    }
