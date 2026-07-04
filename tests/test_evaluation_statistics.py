import pytest

from patchpilot.evaluation.statistics import exact_mcnemar_test


def test_exact_mcnemar_counts_paired_outcomes() -> None:
    result = exact_mcnemar_test(
        full_success=[True] * 12,
        baseline_success=[True] * 8 + [False] * 4,
    )

    assert result.both_success == 8
    assert result.full_only_success == 4
    assert result.baseline_only_success == 0
    assert result.both_failure == 0
    assert result.discordant == 4
    assert result.p_value == pytest.approx(0.125)


def test_exact_mcnemar_requires_paired_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        exact_mcnemar_test(
            full_success=[True],
            baseline_success=[True, False],
        )
