"""
Experiment tracking wrapper around wandb.

Usage:
    logger = ExperimentLogger(cfg)
    logger.log({"train/loss": 0.42, "epoch": 3})
    logger.log_predictions_chart(y_true, y_pred, step=3)  # line chart
    logger.finish()
"""
from __future__ import annotations

from typing import Any, Optional


class ExperimentLogger:
    def __init__(self, cfg, run_name: Optional[str] = None):
        self.enabled = bool(cfg.logging.use_wandb)
        self._wandb = None

        if self.enabled:
            try:
                import wandb

                self._wandb = wandb
                self._wandb.init(
                    project=cfg.logging.wandb_project,
                    entity=cfg.logging.wandb_entity,
                    name=run_name or cfg.logging.run_name,
                    config=cfg.to_dict(),
                )
            except Exception as e:
                print(f"[logger] wandb init failed ({e}); continuing without experiment tracking.")
                self.enabled = False
                self._wandb = None

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self._wandb.log(metrics, step=step)

    def log_predictions_chart(self, y_true, y_pred, title: str = "Predictions vs Actual", step: Optional[int] = None) -> None:
        """Log a line chart comparing predicted vs actual values for one window."""
        if not self.enabled:
            return
        import numpy as np

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        table = self._wandb.Table(
            data=[[i, float(t), float(p)] for i, (t, p) in enumerate(zip(y_true, y_pred))],
            columns=["timestep", "actual", "predicted"],
        )
        chart = self._wandb.plot.line_series(
            xs=list(range(len(y_true))),
            ys=[y_true.tolist(), y_pred.tolist()],
            keys=["actual", "predicted"],
            title=title,
            xname="timestep",
        )
        self.log({title: chart}, step=step)

    def log_comparison_table(self, rows: list[dict], title: str = "model_comparison") -> None:
        """Log a table comparing metrics across models (e.g. Informer vs LSTM vs Transformer)."""
        if not self.enabled or not rows:
            return
        columns = list(rows[0].keys())
        table = self._wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])
        self.log({title: table})

    def watch(self, model) -> None:
        if not self.enabled:
            return
        self._wandb.watch(model, log="all", log_freq=100)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._wandb.finish()