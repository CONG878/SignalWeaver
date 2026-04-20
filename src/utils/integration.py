"""
Numerical Integration for Log-Close Reconstruction

log_return_1d 모드에서 예측된 1일 등락률(Δy)을 로그 종가(log close)로
역산할 때 사용하는 수치 적분 보정 모듈입니다.

공통 수식:
    y(t+N) = y(t) + cum_pred + correction(order)

    여기서 cum_pred = Σ Δy(t+k), k=1..N  (직사각형 적분, order=0 보정항=0)

기호 정의:
    δ_t      = Δy(t)          — 앵커: 마지막 실측 log return
    δ_{t-1}  = Δy(t-1)        — order=2 전용 앵커: 그 전날 실측 log return
    δ_{t+N}  = Δy(t+N)        — 현재 h 시차의 예측 log return
    δ_{t+N-1}= Δy(t+N-1)      — order=2 전용: 직전 시차의 예측 log return

보정항:
    order=0:  0
    order=1:  (− δ_{t+N} + δ_t) / 2
    order=2:  (− 7δ_{t+N} + δ_{t+N-1} + 7δ_t − δ_{t-1}) / 12

수학적 배경:
    order=0: 0차(상수) 보간 — 직사각형 적분 (v3.9.0)
    order=1: 1차(선형) 보간 — 사다리꼴 적분 (v3.9.1)
    order=2: 2차 Lagrange 보간 — Adams-Moulton 2-step (v4.1.0)

사용 위치:
    - src/modeling/trainer.py        (_evaluate, _predict_with_metadata)
    - src/modeling/seq_trainer.py    (_to_log_close)
    - 03a_train_tabular.ipynb        (finalize 셀)
    - 03b_train_ensemble.ipynb       (val/test 저장 셀)
    - 04_forecast_future.ipynb       (Recursive Extension 루프)
    - src/universe/select_universe.py (evaluate_model_accuracy)

config.yaml:
    integration_order: 1   # 0 | 1 | 2

v4.1.0: trapezoidal.py → integration.py 대체
        order=2 (Adams-Moulton 2-step) 추가
        02단계 target_log_return_1d_lag1 컬럼 추가
"""

from __future__ import annotations

import numpy as np


def reconstruct_log_close(
    log_close_base,
    cum_pred,
    delta_y_t,
    delta_y_h,
    delta_y_t_minus_1=None,
    delta_y_h_minus_1=None,
    order: int = 1,
):
    """
    log-return 예측값으로부터 log-close를 역산합니다.

    Parameters
    ----------
    log_close_base : scalar or array-like
        청크 시작 기준 로그 가격 y(t).
    cum_pred : scalar or array-like
        예측 log return의 누적합 Σ Δy(t+k), k=1..N.
        h=1이면 raw_preds[0], h=2이면 raw_preds[0]+raw_preds[1], ...
    delta_y_t : scalar or array-like
        앵커 δ_t = Δy(t).
        - 첫 청크: 마지막 실측 target_log_return_1d
        - 이후 청크: 직전 청크의 raw_preds[-1]
    delta_y_h : scalar or array-like
        현재 h 시차의 예측 log return δ_{t+N} = Δy(t+N).
    delta_y_t_minus_1 : scalar or array-like, optional
        order=2 전용 앵커 δ_{t-1} = Δy(t-1).
        02단계의 target_log_return_1d_lag1 컬럼에서 조달.
    delta_y_h_minus_1 : scalar or array-like, optional
        order=2 전용. 직전 시차의 예측 log return δ_{t+N-1} = Δy(t+N-1).
        - h_idx > 0: raw_preds[h_idx - 1]
        - h_idx == 0 (청크 첫 스텝): 직전 청크의 raw_preds[-1]
    order : int, default=1
        수치 적분 보간 차수.
        0 — 직사각형 (v3.9.0): 보정 없음
        1 — 사다리꼴  (v3.9.1): 선형 보간
        2 — Adams-Moulton 2-step (v4.1.0): 2차 Lagrange 보간

    Returns
    -------
    scalar or np.ndarray
        역산된 로그 종가 y(t+N).

    Notes
    -----
    numpy 브로드캐스팅을 지원하므로 scalar, 1-D array, pandas Series
    어느 형태로 전달해도 동일하게 동작합니다.

    보정항 수식:
        order=0:  correction = 0
        order=1:  correction = (− δ_{t+N} + δ_t) / 2
        order=2:  correction = (− 7δ_{t+N} + δ_{t+N-1} + 7δ_t − δ_{t-1}) / 12

    Examples
    --------
    >>> reconstruct_log_close(7.5, 0.02, 0.01, 0.015, order=1)
    # order=0
    >>> reconstruct_log_close(7.5, 0.02, 0.01, 0.015, order=0)
    # order=2
    >>> reconstruct_log_close(
    ...     7.5, 0.02, 0.01, 0.015,
    ...     delta_y_t_minus_1=0.008, delta_y_h_minus_1=0.012,
    ...     order=2
    ... )
    """
    if order == 0:
        return log_close_base + cum_pred

    elif order == 1:
        correction = (- delta_y_h + delta_y_t) / 2
        return log_close_base + cum_pred + correction

    elif order == 2:
        if delta_y_t_minus_1 is None:
            raise ValueError(
                "order=2에는 delta_y_t_minus_1이 필요합니다.\n"
                "02단계의 target_log_return_1d_lag1 컬럼을 전달하세요."
            )
        if delta_y_h_minus_1 is None:
            raise ValueError(
                "order=2에는 delta_y_h_minus_1이 필요합니다.\n"
                "h_idx > 0이면 raw_preds[h_idx-1], "
                "h_idx == 0이면 직전 청크의 raw_preds[-1]을 전달하세요."
            )
        correction = (
            - 7 * delta_y_h
            + delta_y_h_minus_1
            + 7 * delta_y_t
            - delta_y_t_minus_1
        ) / 12
        return log_close_base + cum_pred + correction

    else:
        raise ValueError(
            f"order는 0, 1, 2 중 하나여야 합니다. 현재: {order}"
        )


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# (trapezoidal.py의 trapezoid_log_close를 그대로 호출하던 코드 대응)
# ---------------------------------------------------------------------------

def trapezoid_log_close(
    log_close_base,
    cum_pred,
    delta_y_t,
    delta_y_h,
):
    """
    .. deprecated::
        v4.1.0부터 ``reconstruct_log_close(..., order=1)`` 로 대체되었습니다.
        이 함수는 하위 호환성을 위해 유지됩니다.
    """
    import warnings
    warnings.warn(
        "trapezoid_log_close()는 deprecated입니다. "
        "reconstruct_log_close(..., order=1)을 사용하세요.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reconstruct_log_close(
        log_close_base, cum_pred, delta_y_t, delta_y_h, order=1
    )
