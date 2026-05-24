"""OutageEvaluator — reusable evaluation harness for any NavPipeline.

Wraps the ``NavPipeline.run_outage`` call with multi-outage / multi-pipeline
convenience methods and a stable result schema used by figure scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from .data.euroc import EuRoCSequence
from .pipeline import NavPipeline, OutageResult


@dataclass
class OutageMetrics:
    """Headline metrics for a single outage run (matches what the README reports)."""
    final_velocity_error: float  # m/s
    mean_velocity_error: float   # m/s
    final_position_drift: float  # m
    mean_position_drift: float   # m

    @classmethod
    def from_result(cls, result: OutageResult) -> "OutageMetrics":
        return cls(
            final_velocity_error=result.final_error_norm,
            mean_velocity_error=result.mean_error_norm,
            final_position_drift=result.final_position_drift,
            mean_position_drift=float(result.position_error.mean()),
        )

    def as_dict(self) -> dict:
        return {
            "final_velocity_error": self.final_velocity_error,
            "mean_velocity_error": self.mean_velocity_error,
            "final_position_drift": self.final_position_drift,
            "mean_position_drift": self.mean_position_drift,
        }


class OutageEvaluator:
    """Run one or more pipelines through one or more outage windows.

    Parameters
    ----------
    sequence:
        The test sequence (e.g., ``EuRoCSequence.load("MH_05_difficult", ...)``).
    outage_start_frac:
        Start of the simulated outage as a fraction of the sequence duration.
        Default 0.4 (matches what every nav-eval script uses).
    """

    def __init__(
        self,
        sequence: EuRoCSequence,
        outage_start_frac: float = 0.4,
    ) -> None:
        self.sequence = sequence
        self.outage_start_frac = outage_start_frac

    def evaluate(
        self,
        pipeline: NavPipeline,
        outage_duration_s: float = 30.0,
    ) -> tuple[OutageResult, OutageMetrics]:
        """Run one pipeline through one outage duration. Returns (result, metrics)."""
        start, end = self.sequence.outage_window(self.outage_start_frac, outage_duration_s)
        result = pipeline.run_outage(self.sequence, start, end)
        return result, OutageMetrics.from_result(result)

    def sweep_durations(
        self,
        pipeline: NavPipeline,
        durations_s: Iterable[float] = (5.0, 10.0, 30.0, 60.0),
    ) -> Dict[float, OutageMetrics]:
        """Run one pipeline across multiple outage durations."""
        out: Dict[float, OutageMetrics] = {}
        for d in durations_s:
            _, metrics = self.evaluate(pipeline, outage_duration_s=d)
            out[d] = metrics
        return out

    def compare(
        self,
        pipelines: Dict[str, NavPipeline],
        outage_duration_s: float = 30.0,
    ) -> Dict[str, OutageMetrics]:
        """Run multiple pipelines through the same outage duration."""
        out: Dict[str, OutageMetrics] = {}
        for name, p in pipelines.items():
            _, metrics = self.evaluate(p, outage_duration_s=outage_duration_s)
            out[name] = metrics
        return out
