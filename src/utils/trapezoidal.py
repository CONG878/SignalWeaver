"""
Trapezoidal Rule Log-Close Inverse Transformation

log_return_1d 모드에서 예측된 1일 등락률(Δy)을 로그 종가(log close)로 역산할 때
단순 누적합(cumsum)의 오차를 줄이기 위해 사다리꼴 적분 보정을 적용합니다.

v3.9.1 규격:
    y(t+h) = y(t) + Σ Δy_i  +  (Δy(t) − Δy(t+h)) / 2

기호 정의:
    log_close_base  : y(t)       — 청크 시작 기준 로그 가격
    cum_pred        : Σ Δy_i     — 예측 log return 누적합 (i=0..h-1)
    delta_y_t       : Δy(t)      — 앵커 (실측 or 직전 청크 마지막 예측값)
    delta_y_h       : Δy(t+h)    — 현재 h 시차의 예측 log return

사용 위치:
    - src/modeling/trainer.py        (_evaluate, _predict_with_metadata)
    - 03_train_predict.ipynb         (finalize 셀)
    - 03b_train_ensemble.ipynb       (val/test 저장 셀)
    - 04_forecast_future.ipynb       (Recursive Extension 루프)
    - src/universe/select_universe.py (evaluate_model_accuracy)
"""

from __future__ import annotations

import numpy as np


def trapezoid_log_close(
    log_close_base,   # y(t): 청크 시작 기준 로그 가격
    cum_pred,         # Σ Δy_i, i=0..h-1: 예측 log return 누적합
    delta_y_t,        # Δy(t): 앵커 (실측 또는 직전 청크 마지막 예측값)
    delta_y_h,        # Δy(t+h): 현재 h 시차 예측 log return
):
    """
    사다리꼴 적분 보정 기반 로그 종가 역산 (v3.9.1 규격).

    y(t+h) = y(t) + Σ Δy_i + (Δy(t) - Δy(t+h)) / 2

    Parameters
    ----------
    log_close_base : scalar or array-like
        청크 시작 기준 로그 가격 y(t).
    cum_pred : scalar or array-like
        예측 log return의 누적합 Σ Δy_i (i=0..h-1).
        h=1이면 raw_preds[0], h=2이면 raw_preds[0]+raw_preds[1], ...
    delta_y_t : scalar or array-like
        앵커값 Δy(t).
        - 첫 청크: 마지막 실측 target_log_return_1d
        - 이후 청크: 직전 청크의 raw_preds[-1]
    delta_y_h : scalar or array-like
        현재 h 시차의 예측 log return Δy(t+h).

    Returns
    -------
    scalar or np.ndarray
        역산된 로그 종가 y(t+h).

    Notes
    -----
    numpy 브로드캐스팅을 활용하므로 scalar, 1-D array, pandas Series
    어느 형태로 전달해도 동일하게 동작합니다.
    """
    return log_close_base + cum_pred + (delta_y_t - delta_y_h) / 2
