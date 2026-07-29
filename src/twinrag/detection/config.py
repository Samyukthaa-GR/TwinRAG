"""Configuration for statistical anomaly detection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterThreshold:
    """Residual thresholds for a monitored parameter."""

    anomaly: float
    warning: float
    critical: float


# ---------------------------------------------------------------------
# Residual thresholds
# ---------------------------------------------------------------------

PRESSURE_THRESHOLD = ParameterThreshold(
    anomaly=0.50,
    warning=1.50,
    critical=5.00,
)

FLOWRATE_THRESHOLD = ParameterThreshold(
    anomaly=0.005,
    warning=0.020,
    critical=0.100,
)


THRESHOLDS = {
    "pressure": PRESSURE_THRESHOLD,
    "flowrate": FLOWRATE_THRESHOLD,
}