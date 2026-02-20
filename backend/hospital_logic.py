"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""Hospital selection logic for Lifeline AI prototype.

Reads a simple JSON "database" of nearby hospitals and selects the nearest
one that currently has at least one ICU bed available.

============================================================================
[PROPRIETARY MONETIZATION FEATURE - Hospital Routing Algorithm]

This module implements the intelligent hospital allocation system which is
a key revenue driver for the Lifeline AI SaaS platform:

BUSINESS VALUE:
- Hospitals pay for efficient patient routing (reduces wait times)
- Insurance companies benefit from smart allocation (cost optimization)
- Emergency responders save time with pre-selected destination
- Direct revenue model: $10-50 per incident routed successfully

REVENUE PROTECTION:
- Hospital ranking algorithm is proprietary
- API access to routing should be behind authentication wall
- Advanced hospital networks (partnerships) get premium placement
- Usage tracking enables per-incident billing

FUTURE ENHANCEMENTS:
- Real-time hospital API integration (vs static JSON)
- Predictive capacity modeling (estimate future bed availability)
- Dynamic pricing based on incident severity + rush level
- Hospital partnership tiers (bronze/silver/gold placement)

============================================================================
"""

import json
import os
from typing import Optional, Dict, Any, List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOSPITALS_PATH = os.path.join(BASE_DIR, "hospitals.json")


def _load_hospitals():
    """Load hospital records from the local JSON file."""
    if not os.path.exists(HOSPITALS_PATH):
        # For robustness in a demo environment, provide a fallback.
        return [
            {"name": "Fallback City Hospital", "icu_beds": 2, "distance_km": 2.0},
        ]

    with open(HOSPITALS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _rush_penalty(rush_level: str) -> float:
    """Convert a textual rush level into a numeric penalty for scoring.

    Higher rush means a larger penalty so hospitals with heavy crowding are
    less likely to be selected when a lower-rush option exists.
    """

    level = (rush_level or "").lower()
    if level == "low":
        return 0.0
    if level == "medium":
        return 1.0
    if level == "high":
        return 3.0
    return 1.5


def _alertness_bonus(alertness: str) -> float:
    """Convert traffic police alertness into a small bonus.

    Higher alertness slightly improves the score (reduces the effective time),
    modelling smoother traffic management on the route.
    """

    level = (alertness or "").lower()
    if level == "high":
        return 1.0
    if level == "medium":
        return 0.5
    if level == "low":
        return 0.0
    return 0.0


def rank_hospitals() -> List[Dict[str, Any]]:
    """Return all hospitals with ICU beds, scored and sorted by priority.

    The first element in the returned list is the hospital that will start
    treatment first according to the scoring logic.
    """

    hospitals = _load_hospitals()

    available = [h for h in hospitals if h.get("icu_beds", 0) > 0]
    if not available:
        return []

    def sort_key(h: Dict[str, Any]):
        # Priority logic aligned with described behaviour:
        # 1) Lower rush_level is best (avoid very crowded hospitals)
        # 2) Then nearer distance_km (closest among similar rush)
        # 3) Then lower ambulance_time_min (ETA)
        # 4) Then more ICU beds

        rush = _rush_penalty(h.get("rush_level", ""))
        distance = float(h.get("distance_km", 999))
        eta = float(h.get("ambulance_time_min", 20))
        icu_beds = int(h.get("icu_beds", 0))

        # Negative beds so that more beds come first when sorting ascending.
        return (rush, distance, eta, -icu_beds)

    ranked = sorted(available, key=sort_key)
    for idx, h in enumerate(ranked, start=1):
        h.setdefault("priority_rank", idx)
        # Simple model of when treatment can actually start at hospital:
        # ambulance_time_min + small triage/entry buffer (e.g. 2 minutes).
        eta_val = float(h.get("ambulance_time_min", 20))
        h.setdefault("treatment_start_min", eta_val + 2.0)
    return ranked


def select_hospital() -> Optional[Dict[str, Any]]:
    """Select the best hospital based on distance, rush, ETA and alertness.

    Returns the *full* hospital record (dict), or a placeholder dict with a
    "name" field set to "No hospital available" if none have ICU capacity.
    """

    ranked = rank_hospitals()
    if not ranked:
        return {"name": "No hospital available"}
    return ranked[0]
