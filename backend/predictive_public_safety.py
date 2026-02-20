"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""
from __future__ import annotations

"""Predictive Public Safety AI module (prototype).

This module provides a rule-based, environment-focused risk assessment for
night-time public areas. It uses only coarse signals like estimated crowd
counts and time of day. It does NOT identify individuals and does NOT
predict criminal intent.

The logic is suitable for hackathon demos and can later be replaced with
real crowd-estimation models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


@dataclass
class CrowdEstimate:
    """Simple data container for crowd estimation inputs."""

    people_count: int
    is_night: bool


def _classify_crowd_density(people_count: int) -> str:
    """Classify crowd density into Low / Medium / High.

    Thresholds are arbitrary and only for prototype purposes.
    """

    if people_count <= 3:
        return "Low"
    if people_count <= 15:
        return "Medium"
    return "High"


def run_predictive_public_safety_detection(
    people_count: int | None = None,
    is_night: bool | None = None,
) -> Dict[str, Any]:
    """Run a simple, rule-based predictive public safety assessment.

    Args:
        people_count: Optional simulated count of visible people in the frame.
            If None, we use a fixed demo value.
        is_night: Optional flag indicating night-time. If None, we infer from
            the current hour (consider 20:00-05:59 as night).

    Returns:
        A dictionary describing the area risk and suggested preventive actions.
    """

    now = datetime.now()
    human_time = now.strftime("%Y-%m-%d %H:%M:%S")

    if is_night is None:
        hour = now.hour
        is_night = hour >= 20 or hour <= 5

    # For hackathon demo, if people_count is not provided, pick a small number
    # to demonstrate the "High Risk Zone (Preventive)" behaviour.
    if people_count is None:
        people_count = 2

    density = _classify_crowd_density(people_count)

    if is_night and density == "Low":
        risk_level = "High Risk Zone (Preventive)"
        patrol_suggestion = "Preventive Patrol Suggested due to Low Night-Time Activity"
        monitoring = "Increase monitoring frequency for this area."
    elif density == "Medium":
        risk_level = "Moderate Risk Zone"
        patrol_suggestion = "Optional patrol – normal awareness recommended."
        monitoring = "Standard monitoring interval."
    else:
        risk_level = "Normal Activity Zone"
        patrol_suggestion = "No preventive patrol required at this time."
        monitoring = "Standard monitoring interval."

    result = {
        "time": human_time,
        "is_night": bool(is_night),
        "people_count": people_count,
        "crowd_density": density,
        "risk_level": risk_level,
        "patrol_suggestion": patrol_suggestion,
        "monitoring_note": monitoring,
    }

    return result
