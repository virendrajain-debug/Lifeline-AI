"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""
from __future__ import annotations

"""Public Safety AI module for Lifeline AI prototype.

This module simulates detection of unusual or potentially dangerous public
behaviour at night, with a focus on women safety and public protection.

For the hackathon prototype we use rule-based / simulated outputs instead of
heavy real models. The interface is simple so it can later be replaced with
actual tracking / behaviour analysis.
"""

from datetime import datetime
from typing import Dict, Any


def run_public_safety_detection() -> Dict[str, Any]:
    """Simulate detection of a public safety threat from CCTV/video.

    The logic is intentionally simplified and rule-based:
    - Assume we are processing night-time footage in a low-crowd area.
    - Assume the AI has detected a continuous chasing behaviour where one
      person aggressively follows another and keeps a close distance.

    We **do not** label anything as rape or assault. We only classify as a
    "Public Safety Threat – Unusual Behavior" and trigger a safety workflow.
    """

    now = datetime.now()
    human_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Mock GPS location and context for demo. This can be wired to real
    # metadata if available from the CCTV system.
    mock_location = "Mock GPS: 22.7196, 75.8577 (Central City Square)"

    incident_type = "Public Safety Threat – Unusual Behavior"

    # Simple status progression for the frontend to visualise.
    status_timeline = [
        {"label": "Detected", "state": "done"},
        {"label": "Police Alerted", "state": "in_progress"},
        {"label": "Patrol Dispatched", "state": "pending"},
    ]

    alert = {
        "threat_detected": True,
        "incident_type": incident_type,
        "time": human_time,
        "location": mock_location,
        "night_time": True,
        "police_alert_status": "Nearby police control room alerted (simulated).",
        "patrol_status": "Nearest patrol unit being dispatched (simulated).",
        "evidence_clip_path": "evidence/public_safety_clip.mp4",  # mock path
        "status_timeline": status_timeline,
    }

    return alert
