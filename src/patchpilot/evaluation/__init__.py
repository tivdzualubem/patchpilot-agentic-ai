"""Evaluation conditions, metrics, and statistical analysis."""

from patchpilot.evaluation.conditions import (
    CONDITION_SPECS,
    PRIMARY_CONDITIONS,
    ConditionSpec,
    ConfiguredCondition,
    EvaluationCondition,
    build_condition,
    condition_values,
    get_condition_spec,
    parse_condition,
)
from patchpilot.evaluation.metrics import (
    RunMetricRow,
    SummaryMetricRow,
    collect_run_metrics,
    summarise_runs,
)
from patchpilot.evaluation.statistics import (
    McNemarResult,
    exact_mcnemar_test,
)

__all__ = [
    "CONDITION_SPECS",
    "PRIMARY_CONDITIONS",
    "ConditionSpec",
    "ConfiguredCondition",
    "EvaluationCondition",
    "McNemarResult",
    "RunMetricRow",
    "SummaryMetricRow",
    "build_condition",
    "collect_run_metrics",
    "condition_values",
    "exact_mcnemar_test",
    "get_condition_spec",
    "parse_condition",
    "summarise_runs",
]
