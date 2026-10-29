"""Placeholder for validated risk thresholds.

Do not choose arbitrary thresholds for production. Agree on and validate
probability cutoffs with the team using validation data and the false-alarm target.
"""

def classify_risk(delay_probability: float, at_risk_threshold: float, critical_threshold: float) -> str:
    if not 0 <= delay_probability <= 1:
        raise ValueError("delay_probability must be between 0 and 1")
    if not 0 <= at_risk_threshold <= critical_threshold <= 1:
        raise ValueError("Require 0 <= at_risk_threshold <= critical_threshold <= 1")
    if delay_probability < at_risk_threshold:
        return "On-Time"
    if delay_probability < critical_threshold:
        return "At-Risk"
    return "Critical Delay"
