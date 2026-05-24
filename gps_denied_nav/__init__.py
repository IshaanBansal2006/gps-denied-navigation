"""gps_denied_nav — neural-aided IMU navigation for GPS-denied UAV flight.

Top-level convenience re-exports::

    from gps_denied_nav import NavPipeline, OutageEvaluator, EuRoCSequence
    from gps_denied_nav.models import LSTMRegressor, load_lstm_checkpoint
    from gps_denied_nav.adaptation import RLSHead
    from gps_denied_nav.filters import VelocityOnlyFilter

See the README for an end-to-end usage example on EuRoC MH_05.
"""
from .data.euroc import EuRoCSequence
from .eval import OutageEvaluator, OutageMetrics
from .pipeline import NavPipeline, OutageResult

__version__ = "0.1.0"

__all__ = [
    "EuRoCSequence",
    "NavPipeline",
    "OutageEvaluator",
    "OutageMetrics",
    "OutageResult",
    "__version__",
]
