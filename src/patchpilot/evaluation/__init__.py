"""Evaluation metrics and statistical analysis for PatchPilot runs."""

from patchpilot.evaluation.metrics import (
    RunMetricRow,
    SummaryMetricRow,
    collect_run_metrics,
    summarise_runs,
)
from patchpilot.evaluation.statistics import McNemarResult, exact_mcnemar_test

__all__ = [
    "McNemarResult",
    "RunMetricRow",
    "SummaryMetricRow",
    "collect_run_metrics",
    "exact_mcnemar_test",
    "summarise_runs",
]
