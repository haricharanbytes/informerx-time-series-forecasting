"""
Metrics: MSE, MAE, RMSE (+ MAPE as a bonus), computed on numpy arrays in the ORIGINAL (inverse-transformed).
"""
from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-7) -> float:
    """Mean absolute percentage error. Guards against division by ~0."""
    return float(np.mean(np.abs((pred - true) / (true + eps)))) * 100.0


def compute_all_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "MSE": mse(pred, true),
        "MAE": mae(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true),
    }