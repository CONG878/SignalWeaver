"""
Purpose:
    - PyTorch 기반 MLP Multi-output 예측 모델 구현체
    - 단일 네트워크로 h1~h5를 동시에 출력 (공유 잠재 표현)
    - MultiOutputRegressor 불필요 — 진정한 단일 모델 다중 출력
    - ModelBase 인터페이스 준수
    - 내부에서 StandardScaler를 관리 (fit/predict/save/load 일관성 보장)

Design notes:
    - fit()  : StandardScaler 학습 → _MLPNet 구성 → Adam + MSELoss + Early Stopping
    - predict(): scaler.transform → net.eval() forward → DataFrame 반환
    - save() : state_dict + scaler + 메타데이터를 단일 pickle로 직렬화
    - load() : pickle 역직렬화 → 네트워크 재구성 → state_dict 복원

Architecture:
    Input(feature_dim)
      → [Linear → BatchNorm1d → ReLU → Dropout(rate)] × len(hidden_dims)
      → Linear(output_dim)                          # output_dim = len(horizons)

    Dropout rate는 hidden_dims와 1:1 대응하는 리스트로 지정.
    마지막 히든 레이어의 rate=0.0 관례를 지원.

Trainer 호환:
    WalkForwardTrainer.run()의 model.fit() 호출 시그니처:
        fit(X, y, eval_set=[(X_val, y_val)], **fit_kwargs)
    → epochs, patience는 fit_kwargs를 통해 전달됨 (config.yaml mlp_params 기반)
    → eval_set을 이용한 early stopping 지원
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import ModelBase


# ──────────────────────────────────────────────────────────────
# 내부 네트워크 정의
# ──────────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    """
    MLP 네트워크 정의.

    Parameters
    ----------
    input_dim : int
        입력 피처 수
    hidden_dims : List[int]
        각 히든 레이어의 뉴런 수 (예: [128, 64, 32])
    dropout_rates : List[float]
        각 히든 레이어 이후 드롭아웃 비율.
        len(dropout_rates) == len(hidden_dims) 필수.
        rate=0.0이면 Dropout 레이어를 추가하지 않음.
    output_dim : int
        출력 뉴런 수 (= len(horizons), 보통 5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rates: List[float],
        output_dim: int,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim, dr in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dr > 0.0:
                layers.append(nn.Dropout(dr))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# 공개 모델 클래스
# ──────────────────────────────────────────────────────────────

class MLPModel(ModelBase):
    """
    PyTorch MLP Multi-output 회귀 모델 래퍼.

    특징:
    - 단일 forward pass로 모든 horizon을 동시에 예측 (공유 표현)
    - LightGBM(타겟별 독립 Booster)·RF(MultiOutputRegressor)와 달리
      출력 간 공분산 구조를 묵시적으로 학습
    - 내부 StandardScaler로 피처 정규화 캡슐화

    Parameters
    ----------
    model_version : str
        버전 식별자 (예: "v1_mlp_2025-08-13")
    params : Dict[str, Any]
        아키텍처 및 학습 파라미터. 필드:
            hidden_dims     : List[int]   — 기본 [128, 64, 32]
            dropout_rates   : List[float] — 기본 [0.2, 0.2, 0.0]
            learning_rate   : float       — 기본 0.001
            batch_size      : int         — 기본 1024
            weight_decay    : float       — 기본 0.0001
        런타임 파라미터 (fit_kwargs로 전달 권장):
            epochs   : int — 기본 200
            patience : int — 기본 10
    feature_list : List[str]
        학습/예측에 사용할 피처 컬럼명 목록
    task : str
        "regression" 고정 (확장 여지 확보용)
    """

    def __init__(
        self,
        model_version: str,
        params: Dict[str, Any],
        feature_list: List[str],
        task: str = "regression",
    ) -> None:
        super().__init__(model_name="mlp_multi", model_version=model_version)

        self.params = params
        self.feature_list = feature_list
        self.task = task
        self.target_columns: List[str] = []

        # 아키텍처 파라미터 추출
        self.hidden_dims: List[int] = params.get("hidden_dims", [128, 64, 32])
        self.dropout_rates: List[float] = params.get("dropout_rates", [0.2, 0.2, 0.0])
        self.learning_rate: float = params.get("learning_rate", 0.001)
        self.batch_size: int = params.get("batch_size", 1024)
        self.weight_decay: float = params.get("weight_decay", 0.0001)

        if len(self.hidden_dims) != len(self.dropout_rates):
            raise ValueError(
                f"hidden_dims 길이({len(self.hidden_dims)})와 "
                f"dropout_rates 길이({len(self.dropout_rates)})가 일치해야 합니다."
            )

        # 런타임에 초기화되는 내부 상태
        self.scaler: StandardScaler = StandardScaler()
        self.net: Optional[_MLPNet] = None
        self.device: torch.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        eval_set: Optional[List[Tuple]] = None,
        **kwargs,
    ) -> None:
        """
        MLP 학습.

        Parameters
        ----------
        X : DataFrame
            피처 행렬 (self.feature_list 컬럼 포함 필수)
        y : Series or DataFrame
            타깃. DataFrame일 경우 컬럼 수 = output_dim
        eval_set : List[Tuple], optional
            [(X_val, y_val)]. early stopping 기준.
            미전달 시 전체 epochs 학습 후 최종 가중치 사용.
        **kwargs
            epochs   : int — 최대 학습 에폭 (기본 200)
            patience : int — early stopping 인내 에폭 (기본 15)
        """
        # fit_kwargs에서 런타임 파라미터 추출
        epochs: int = kwargs.pop("epochs", self.params.get("epochs", 200))
        patience: int = kwargs.pop("patience", self.params.get("patience", 15))

        # y 정규화
        if isinstance(y, pd.Series):
            y = y.to_frame()
        self.target_columns = y.columns.tolist()

        input_dim = len(self.feature_list)
        output_dim = len(self.target_columns)

        # 네트워크 초기화
        self.net = _MLPNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rates=self.dropout_rates,
            output_dim=output_dim,
        ).to(self.device)

        # ── 훈련 데이터 전처리 ──
        X_train_np = X[self.feature_list].values.astype(np.float32)
        X_train_scaled = self.scaler.fit_transform(X_train_np)
        y_train_np = y.values.astype(np.float32)

        # NaN 행 제거 (Trainer가 dropna를 선행하지만 이중 방어)
        train_mask = ~(
            np.isnan(X_train_scaled).any(axis=1)
            | np.isnan(y_train_np).any(axis=1)
        )
        X_train_scaled = X_train_scaled[train_mask]
        y_train_np = y_train_np[train_mask]

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train_np, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,   # BatchNorm1d는 배치 크기 1에서 오류 발생 방지
        )

        # ── 검증 데이터 전처리 ──
        val_loader: Optional[DataLoader] = None
        if eval_set:
            X_val, y_val = eval_set[0]
            if isinstance(y_val, pd.Series):
                y_val = y_val.to_frame()
            X_val_np = X_val[self.feature_list].values.astype(np.float32)
            X_val_scaled = self.scaler.transform(X_val_np)   # 훈련 scaler 재사용
            y_val_np = y_val.values.astype(np.float32)

            val_mask = ~(
                np.isnan(X_val_scaled).any(axis=1)
                | np.isnan(y_val_np).any(axis=1)
            )
            X_val_scaled = X_val_scaled[val_mask]
            y_val_np = y_val_np[val_mask]

            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val_scaled, dtype=torch.float32),
                    torch.tensor(y_val_np, dtype=torch.float32),
                ),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,  # 검증은 전체 샘플 평가
            )

        # ── 학습 루프 ──
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        best_val_loss: float = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        no_improve: int = 0

        for epoch in range(1, epochs + 1):
            # 훈련 페이즈
            self.net.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            # 검증 페이즈 (eval_set이 있을 때만)
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(
                            f"   [MLP] Early stopping — epoch {epoch} | "
                            f"best val_loss: {best_val_loss:.6f}"
                        )
                        break
            else:
                # eval_set 없음: 마지막 가중치를 best로 간주
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

        # best 가중치 복원
        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.net.eval()
        self.is_fitted = True

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        target_name: Optional[str] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        예측 수행.

        Parameters
        ----------
        X : DataFrame
        target_name : str, optional
            지정 시 해당 컬럼만 Series로 반환 (Trainer 내부 단일 타겟 조회용)

        Returns
        -------
        DataFrame (전체 horizon) 또는 Series (target_name 지정 시)
        """
        self.validate_fitted()
        self.net.eval()

        X_np = X[self.feature_list].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_np)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds_np = self.net(X_tensor).cpu().numpy()

        pred_df = pd.DataFrame(
            preds_np, index=X.index, columns=self.target_columns
        )

        if target_name is not None:
            if target_name not in self.target_columns:
                raise ValueError(
                    f"Target '{target_name}'이 학습된 컬럼 목록에 없습니다. "
                    f"가용 컬럼: {self.target_columns}"
                )
            return pred_df[target_name]

        return pred_df

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        모델 전체를 단일 pickle 파일로 저장.
        저장 대상: state_dict, scaler, 아키텍처 메타데이터
        """
        self.validate_fitted()

        payload = {
            "model_name":     self.model_name,
            "model_version":  self.model_version,
            "params":         self.params,
            "feature_list":   self.feature_list,
            "target_columns": self.target_columns,
            "hidden_dims":    self.hidden_dims,
            "dropout_rates":  self.dropout_rates,
            "scaler":         self.scaler,
            "net_state_dict": self.net.state_dict(),
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        """
        저장된 pickle 파일에서 모델 복원.
        네트워크 구조는 저장된 메타데이터로 재구성.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            model_version=payload["model_version"],
            params=payload["params"],
            feature_list=payload["feature_list"],
        )

        instance.target_columns = payload["target_columns"]
        instance.hidden_dims    = payload["hidden_dims"]
        instance.dropout_rates  = payload["dropout_rates"]
        instance.scaler         = payload["scaler"]

        input_dim  = len(payload["feature_list"])
        output_dim = len(payload["target_columns"])

        instance.net = _MLPNet(
            input_dim=input_dim,
            hidden_dims=instance.hidden_dims,
            dropout_rates=instance.dropout_rates,
            output_dim=output_dim,
        )
        instance.net.load_state_dict(payload["net_state_dict"])
        instance.net.eval()
        instance.is_fitted = True

        return instance

    def get_meta(self) -> Dict[str, Any]:
        return {
            "model_name":     self.model_name,
            "model_version":  self.model_version,
            "params":         self.params,
            "feature_list":   self.feature_list,
            "target_columns": self.target_columns,
            "hidden_dims":    self.hidden_dims,
            "dropout_rates":  self.dropout_rates,
            "is_fitted":      self.is_fitted,
        }

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _evaluate_loss(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """검증 DataLoader 전체에 대한 평균 MSE 손실 계산."""
        self.net.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.net(X_batch)
                loss = criterion(pred, y_batch)
                n = len(X_batch)
                total_loss += loss.item() * n
                total_samples += n

        return total_loss / total_samples if total_samples > 0 else float("inf")
