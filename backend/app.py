"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""Flask application for Lifeline AI hackathon prototype.

This app exposes:
- "/"         -> redirects to "/dashboard"
- "/detect"   -> runs emergency detection + hospital allocation, returns JSON
- "/dashboard"-> renders a simple dashboard with the latest detection result

Run locally (from the backend folder):
    pip install -r requirements.txt
    python app.py
"""

import os
from datetime import datetime
from flask import Flask, jsonify, redirect, render_template, url_for, send_from_directory, request
from werkzeug.utils import secure_filename

from detector import run_emergency_detection
from hospital_logic import select_hospital, rank_hospitals
from public_safety import run_public_safety_detection
from predictive_public_safety import run_predictive_public_safety_detection


# Configure Flask to look for templates in ../frontend/templates relative to this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend/templates"))

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ============================================================================
# [FUTURE: API KEY AUTHENTICATION]
# 
# When moving to production/SaaS, uncomment and implement authentication:
# 
# import os
# from functools import wraps
# 
# API_KEYS = {
#     os.getenv('LIFELINE_API_KEY'): {'tier': 'enterprise', 'monthly_quota': None},
#     os.getenv('LIFELINE_DEMO_KEY'): {'tier': 'demo', 'monthly_quota': 100},
# }
#
# def require_api_key(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         api_key = request.headers.get('X-API-Key')
#         if not api_key or api_key not in API_KEYS:
#             return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
#         request.api_tier = API_KEYS[api_key]['tier']
#         return f(*args, **kwargs)
#     return decorated_function
#
# Then apply @require_api_key to /detect and /camera/* endpoints
# ============================================================================


# In-memory store for the last detection result so the dashboard can show it.
last_detection_result = {
    "emergency": False,
    "reason": "No analysis has been run yet.",
    "selected_hospital": None,
}

# In-memory store for the last public safety alert.
last_public_safety_alert = {
    "threat_detected": False,
    "incident_type": None,
    "time": None,
    "location": None,
    "night_time": False,
    "police_alert_status": None,
    "patrol_status": None,
    "evidence_clip_path": None,
    "status_timeline": [],
}

# In-memory store for the last predictive public safety assessment.
last_predictive_public_safety = {
    "time": None,
    "is_night": False,
    "people_count": None,
    "crowd_density": None,
    "risk_level": None,
    "patrol_suggestion": None,
    "monitoring_note": None,
}

# Simple in-memory record of the last operation (default or uploaded).
last_operation_meta = {
    "label": None,
    "time": None,
    "filename": None,
    "type": None,
}

# In-memory list of uploaded feeds so the dashboard can draw a camera card for
# each admin-added CCTV/video clip.
uploaded_feeds = []


@app.route("/")
def index() -> str:
    """Redirect to the dashboard for convenience."""
    return redirect(url_for("dashboard"))


@app.route("/feed/<int:feed_id>")
def feed_detail(feed_id: int):
    """Detail view for an uploaded camera feed.

    Reuses the generic camera_detail.html template to show the uploaded
    footage, AI reason, and hospital routing in a consistent layout.
    """

    feed = next((f for f in uploaded_feeds if f.get("id") == feed_id), None)
    if feed is None:
        return redirect(url_for("dashboard"))

    # Build a simple hospital table from the ranked hospitals so the view
    # matches the other camera detail pages.
    hospitals_ranked = rank_hospitals()
    hospitals_table = []
    for h in hospitals_ranked:
        hospitals_table.append(
            {
                "name": h.get("name", "Unknown"),
                "icu_beds": h.get("icu_beds", 0),
                "distance_km": h.get("distance_km", "-"),
                "rush_level": (h.get("rush_level") or "-").title(),
                "ambulance_eta": f"{h.get('ambulance_time_min')} min" if h.get("ambulance_time_min") is not None else "-",
                "police_alertness": (h.get("police_alertness") or "-").title(),
                "priority_rank": h.get("priority_rank", "-"),
                "treatment_start": f"{h.get('treatment_start_min')} min" if h.get("treatment_start_min") is not None else "-",
            }
        )

    status = "emergency" if feed.get("emergency") else "normal"

    return render_template(
        "camera_detail.html",
        camera_id=f"U{feed_id:02d}",
        area_name=feed.get("label") or f"Uploaded Feed {feed_id}",
        gps="-",
        description=f"Uploaded CCTV/video feed file: {feed.get('filename')}",
        status=status,
        incident_type="Uploaded Feed Analysis",
        analysis_summary=feed.get("reason") or "No reason available.",
        recommended_action="Review this feed together with the hospital routing table below.",
        selected_hospital=feed.get("selected_hospital") or "No hospital selected",
        hospitals_table=hospitals_table,
        timeline=feed.get("timeline") or [],
        video_url=url_for("uploaded_video", filename=feed.get("filename")),
    )


@app.route("/uploaded_video/<path:filename>")
def uploaded_video(filename: str):
    """Serve uploaded video files for dynamic camera cards."""

    video_dir = os.path.join(BASE_DIR, "video")
    return send_from_directory(video_dir, filename)


@app.route("/delete_feed/<int:feed_id>", methods=["POST"])
def delete_feed(feed_id: int):
    """Allow admin to delete an uploaded camera feed card (and its file)."""

    global uploaded_feeds

    feed = next((f for f in uploaded_feeds if f.get("id") == feed_id), None)
    if feed is not None:
        filepath = os.path.join(BASE_DIR, "video", feed.get("filename", ""))
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            # For a demo we silently ignore file removal errors.
            pass

        uploaded_feeds = [f for f in uploaded_feeds if f.get("id") != feed_id]

    return redirect(url_for("dashboard"))


def _run_detection_and_update_state(video_path: str | None = None, label: str | None = None):
    """Internal helper to run detection and update in-memory state.

    This lets both /detect and /dashboard trigger the same logic so that
    operations can start automatically without manually opening /detect.
    
    VIDEO MODE RULE: Do NOT select hospital for uploaded/recorded videos.
    Hospitals should only be selected for LIVE camera feeds with location metadata.
    """

    global last_detection_result, last_operation_meta

    emergency_detected, reason, timeline = run_emergency_detection(video_path=video_path, quiet=True)

    # CRITICAL FIX: Only select hospital for LIVE feeds, NOT for uploaded videos
    selected_hospital = None
    is_video_mode = video_path is not None
    
    if emergency_detected and not is_video_mode:
        # LIVE mode: Select hospital if location metadata exists
        selected_hospital = select_hospital()
    elif emergency_detected and is_video_mode:
        # VIDEO mode (uploaded/recorded): Do NOT select hospital
        # This is for human/operator review only
        selected_hospital = None

    operation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    last_detection_result = {
        "emergency": emergency_detected,
        "reason": reason,
        "selected_hospital": selected_hospital,
        "operation_time": operation_time,
        "timeline": timeline,
    }

    # Update high-level operation meta so the dashboard can show when and for
    # which source the latest analysis was run.
    last_operation_meta = {
        "label": label or "Default accident_sample feed",
        "time": operation_time,
        "filename": os.path.basename(video_path) if video_path else "accident_sample.mp4",
        "type": "Uploaded Video" if video_path else "Default Demo",
    }

    return last_detection_result


@app.route("/detect")
def detect():
    """Run emergency detection and hospital allocation and return JSON."""

    result = _run_detection_and_update_state()
    return jsonify(result)


@app.route("/upload_feed", methods=["POST"])
def upload_feed():
    """Admin endpoint to upload CCTV/video footage and run detection on it.

    The uploaded file is stored under the backend video folder and analysed
    immediately so that the dashboard reflects this new operation.
    """

    global uploaded_feeds

    file = request.files.get("video_file")
    label = request.form.get("label") or "Uploaded Feed"

    if not file or file.filename == "":
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    video_dir = os.path.join(BASE_DIR, "video")
    os.makedirs(video_dir, exist_ok=True)
    save_path = os.path.join(video_dir, filename)
    file.save(save_path)

    # Run detection on the newly uploaded footage and update state/operation
    # metadata so the dashboard shows this operation.
    result = _run_detection_and_update_state(video_path=save_path, label=label)

    # Record this feed so the dashboard can render a dedicated camera card.
    # VIDEO MODE: Do NOT include hospital selection, tag to camera only
    # Format: "Incident detected on Camera X" or "No incident on Camera X"
    
    # Assign to next available camera slot (camera_1, camera_2, camera_3, camera_4)
    camera_id = min(len(uploaded_feeds) % 4 + 1, 4)  # Cycle through 1-4
    incident_tag = f"Camera {camera_id}"
    
    if result.get("emergency", False):
        incident_status = f"Incident detected on {incident_tag}"
    else:
        incident_status = f"No incident on {incident_tag}"

    uploaded_feeds.append(
        {
            "id": len(uploaded_feeds) + 1,
            "label": label,
            "filename": filename,
            "time": result.get("operation_time"),
            "type": "Uploaded Video",
            "emergency": bool(result.get("emergency", False)),
            "reason": result.get("reason", "-"),
            "selected_hospital": incident_status,  # Tag to camera instead of hospital
            "timeline": result.get("timeline") or [],
        }
    )

    return redirect(url_for("dashboard"))


@app.route("/public_safety_detect")
def public_safety_detect():
    """Run the Public Safety AI simulated analysis.

    This triggers a rule-based detection of unusual public behaviour at night
    (e.g. one person aggressively chasing another in a low-crowd area) and
    returns a structured alert object. Existing accident detection is not
    affected by this endpoint.
    """

    global last_public_safety_alert

    alert = run_public_safety_detection()
    last_public_safety_alert = alert

    return jsonify(alert)


@app.route("/predictive_public_safety")
def predictive_public_safety():
    """Run Predictive Public Safety AI (environment-based risk assessment).

    This uses only environmental signals (time of day, crowd density) to mark
    areas as Normal / Moderate / High Risk (Preventive). It does not identify
    individuals or predict intent.
    """

    global last_predictive_public_safety

    result = run_predictive_public_safety_detection()
    last_predictive_public_safety = result

    return jsonify(result)


@app.route("/camera1_video")
def camera1_video():
    """Serve the demo accident video for Camera 01 preview.

    This does not affect the detection logic, it is purely for a small
    front-end preview of the accident_sample.mp4 clip.
    """

    video_dir = os.path.join(BASE_DIR, "video")
    return send_from_directory(video_dir, "accident_sample.mp4")


@app.route("/camera2_video")
def camera2_video():
    """Serve demo crowd video for Camera 02 preview (high crowd area)."""

    video_dir = os.path.join(BASE_DIR, "video")
    return send_from_directory(video_dir, "camera2_demo.mp4")


@app.route("/camera3_video")
def camera3_video():
    """Serve demo low-crowd video for Camera 03 preview (silent area)."""

    video_dir = os.path.join(BASE_DIR, "video")
    return send_from_directory(video_dir, "camera3_demo.mp4")


@app.route("/camera4_video")
def camera4_video():
    """Serve demo high-alert dark area video for Camera 04 preview."""

    video_dir = os.path.join(BASE_DIR, "video")
    return send_from_directory(video_dir, "camera4_demo.mp4")


@app.route("/camera/1")
def camera_1_detail():
    # Reuse hospital ranking logic so the camera view can show the same
    # hospital priority table as the main dashboard.
    hospitals_ranked = rank_hospitals()
    hospitals_table = []
    for h in hospitals_ranked:
        hospitals_table.append(
            {
                "name": h.get("name", "Unknown"),
                "icu_beds": h.get("icu_beds", 0),
                "distance_km": h.get("distance_km", "-"),
                "rush_level": (h.get("rush_level") or "-").title(),
                "ambulance_eta": f"{h.get('ambulance_time_min')} min" if h.get("ambulance_time_min") is not None else "-",
                "police_alertness": (h.get("police_alertness") or "-").title(),
                "priority_rank": h.get("priority_rank", "-"),
                "treatment_start": f"{h.get('treatment_start_min')} min" if h.get("treatment_start_min") is not None else "-",
            }
        )

    selected = last_detection_result.get("selected_hospital")
    hospital_name = "No hospital selected"
    if isinstance(selected, dict) and selected:
        hospital_name = selected.get("name", hospital_name)
    elif isinstance(selected, str):
        hospital_name = selected

    return render_template(
        "camera_detail.html",
        camera_id="01",
        area_name="Jan Marg – Intersection A",
        gps="22.7196, 75.8577",
        description="Night-time CCTV at a busy intersection where a collision has been detected.",
        status="emergency",
        incident_type="Road Accident",
        analysis_summary="Collision detected on the carriageway with a person-vehicle interaction.",
        recommended_action="Ambulance dispatched, nearest hospital selected, and traffic police alerted.",
        selected_hospital=hospital_name,
        hospitals_table=hospitals_table,
    )


@app.route("/camera/2")
def camera_2_detail():
    # For Bengaluru camera, show a simplified two-hospital table local to this
    # location while keeping the global hospital logic unchanged elsewhere.
    hospitals_table = [
        {
            "name": "Sanskars Hospital",
            "icu_beds": 4,
            "distance_km": 1.2,
            "rush_level": "Low",
            "ambulance_eta": "5 min",
            "police_alertness": "High",
            "priority_rank": 1,
            "treatment_start": "~10 min",
        },
        {
            "name": "Bengaluru Clinic",
            "icu_beds": 2,
            "distance_km": 3.5,
            "rush_level": "Medium",
            "ambulance_eta": "9 min",
            "police_alertness": "Medium",
            "priority_rank": 2,
            "treatment_start": "~15 min",
        },
    ]

    # For this camera view, highlight the top-priority hospital directly.
    hospital_name = hospitals_table[0]["name"] + " (in case of some emergency this is more suitable)"

    return render_template(
        "camera_detail.html",
        camera_id="02",
        area_name="Bengaluru – Market Zone",
        gps="22.7199, 75.8582",
        description="Night-time CCTV covering a high crowd market area.",
        status="normal",
        incident_type="Crowd Monitoring",
        analysis_summary="High crowd density observed, but behaviour appears normal and non-threatening.",
        recommended_action="No immediate action required. Continue routine monitoring.",
        selected_hospital=hospital_name,
        hospitals_table=hospitals_table,
    )


@app.route("/camera/3")
def camera_3_detail():
    hospitals_ranked = rank_hospitals()
    hospitals_table = []
    for h in hospitals_ranked:
        hospitals_table.append(
            {
                "name": h.get("name", "Unknown"),
                "icu_beds": h.get("icu_beds", 0),
                "distance_km": h.get("distance_km", "-"),
                "rush_level": (h.get("rush_level") or "-").title(),
                "ambulance_eta": f"{h.get('ambulance_time_min')} min" if h.get("ambulance_time_min") is not None else "-",
                "police_alertness": (h.get("police_alertness") or "-").title(),
                "priority_rank": h.get("priority_rank", "-"),
                "treatment_start": f"{h.get('treatment_start_min')} min" if h.get("treatment_start_min") is not None else "-",
            }
        )

    selected = last_detection_result.get("selected_hospital")
    hospital_name = "No hospital selected"
    if isinstance(selected, dict) and selected:
        hospital_name = selected.get("name", hospital_name)
    elif isinstance(selected, str):
        hospital_name = selected

    return render_template(
        "camera_detail.html",
        camera_id="03",
        area_name="Jan Marg – Silent Stretch",
        gps="22.7203, 75.8565",
        description="Night-time CCTV monitoring a low-crowd, isolated road segment.",
        status="preventive",
        incident_type="Predictive Public Safety",
        analysis_summary="Low crowd density at night indicates a silent, isolated stretch with higher preventive risk.",
        recommended_action="Suggest preventive police patrol and increased monitoring frequency.",
        selected_hospital=hospital_name,
        hospitals_table=hospitals_table,
    )


@app.route("/camera/4")
def camera_4_detail():
    hospitals_ranked = rank_hospitals()
    hospitals_table = []
    for h in hospitals_ranked:
        hospitals_table.append(
            {
                "name": h.get("name", "Unknown"),
                "icu_beds": h.get("icu_beds", 0),
                "distance_km": h.get("distance_km", "-"),
                "rush_level": (h.get("rush_level") or "-").title(),
                "ambulance_eta": f"{h.get('ambulance_time_min')} min" if h.get("ambulance_time_min") is not None else "-",
                "police_alertness": (h.get("police_alertness") or "-").title(),
                "priority_rank": h.get("priority_rank", "-"),
                "treatment_start": f"{h.get('treatment_start_min')} min" if h.get("treatment_start_min") is not None else "-",
            }
        )

    selected = last_detection_result.get("selected_hospital")
    hospital_name = "No hospital selected"
    if isinstance(selected, dict) and selected:
        hospital_name = selected.get("name", hospital_name)
    elif isinstance(selected, str):
        hospital_name = selected

    return render_template(
        "camera_detail.html",
        camera_id="04",
        area_name="Jan Marg – Dark Alley",
        gps="22.7208, 75.8559",
        description="Very dark alley with a single entity detected; high alert for public safety.",
        status="preventive",
        incident_type="High-Alert Night-Time Monitoring",
        analysis_summary="Only one moving entity detected in a very dark zone; increased preventive monitoring recommended.",
        recommended_action="Keep patrol units on high alert and increase camera monitoring frequency.",
        nearest_police_station="Jantroops PS (Nearest Police Station)",
        selected_hospital=hospital_name,
        hospitals_table=hospitals_table,
    )


@app.route("/dashboard")
def dashboard():
    """Render a simple HTML dashboard showing the latest detection result.

    Note: The dashboard only *displays* the latest result. To trigger a fresh
    detection, call the /detect endpoint (e.g. open it in a browser tab or
    use curl/Postman) and then refresh this page.
    """
    # VIDEO MODE FIX: DO NOT auto-run detection on every dashboard load.
    # This prevents repeated video re-scanning (LIVE mode behavior).
    # Dashboard shows cached results only. User must explicitly visit /detect.
    # (Removed: _run_detection_and_update_state())

    emergency = last_detection_result.get("emergency", False)
    reason = last_detection_result.get("reason", "No data")
    selected = last_detection_result.get("selected_hospital")

    hospital_name = "No hospital selected"
    rush_level = "-"
    ambulance_eta = "-"
    police_alertness = "-"

    if isinstance(selected, dict) and selected:
        hospital_name = selected.get("name", hospital_name)
        rush_level = (selected.get("rush_level") or "-").title()
        eta_val = selected.get("ambulance_time_min")
        ambulance_eta = f"{eta_val} min" if eta_val is not None else "-"
        police_alertness = (selected.get("police_alertness") or "-").title()
    elif isinstance(selected, str):
        hospital_name = selected

    # Build a table of hospitals with their properties and priority.
    hospitals_ranked = rank_hospitals()
    hospitals_table = []
    for h in hospitals_ranked:
        hospitals_table.append(
            {
                "name": h.get("name", "Unknown"),
                "icu_beds": h.get("icu_beds", 0),
                "distance_km": h.get("distance_km", "-"),
                "rush_level": (h.get("rush_level") or "-").title(),
                "ambulance_eta": f"{h.get('ambulance_time_min')} min" if h.get("ambulance_time_min") is not None else "-",
                "police_alertness": (h.get("police_alertness") or "-").title(),
                "priority_rank": h.get("priority_rank", "-"),
                "treatment_start": f"{h.get('treatment_start_min')} min" if h.get("treatment_start_min") is not None else "-",
                "is_selected": h.get("name") == hospital_name,
            }
        )

    # Public Safety AI: prepare alert info for the dashboard.
    ps = last_public_safety_alert or {}
    ps_threat = ps.get("threat_detected", False)
    ps_incident_type = ps.get("incident_type") or "No alerts yet"
    ps_time = ps.get("time") or "-"
    ps_location = ps.get("location") or "-"
    ps_night = bool(ps.get("night_time", False))
    ps_police_status = ps.get("police_alert_status") or "-"
    ps_patrol_status = ps.get("patrol_status") or "-"
    ps_timeline = ps.get("status_timeline") or []

    # Predictive Public Safety AI context.
    pps = last_predictive_public_safety or {}
    pps_time = pps.get("time") or "-"
    pps_is_night = bool(pps.get("is_night", False))
    pps_density = pps.get("crowd_density") or "Unknown"
    pps_risk_level = pps.get("risk_level") or "No assessment yet"
    pps_patrol_suggestion = pps.get("patrol_suggestion") or "-"
    pps_monitoring_note = pps.get("monitoring_note") or "-"

    ambulance_status = "Dispatched" if emergency else "On standby"

    op_label = last_operation_meta.get("label") or "-"
    op_time = last_operation_meta.get("time") or "-"
    op_type = last_operation_meta.get("type") or "-"

    return render_template(
        "dashboard.html",
        emergency="YES" if emergency else "NO",
        emergency_bool=emergency,
        reason=reason,
        selected_hospital=hospital_name,
        hospital_rush_level=rush_level,
        hospital_ambulance_eta=ambulance_eta,
        hospital_police_alertness=police_alertness,
        ambulance_status=ambulance_status,
        hospitals_table=hospitals_table,
        # Public Safety AI context
        ps_threat_detected=ps_threat,
        ps_incident_type=ps_incident_type,
        ps_time=ps_time,
        ps_location=ps_location,
        ps_night_time=ps_night,
        ps_police_status=ps_police_status,
        ps_patrol_status=ps_patrol_status,
        ps_status_timeline=ps_timeline,
        # Predictive Public Safety context
        pps_time=pps_time,
        pps_is_night=pps_is_night,
        pps_density=pps_density,
        pps_risk_level=pps_risk_level,
        pps_patrol_suggestion=pps_patrol_suggestion,
        pps_monitoring_note=pps_monitoring_note,
        op_label=op_label,
        op_time=op_time,
        op_type=op_type,
        uploaded_feeds=uploaded_feeds,
    )


if __name__ == "__main__":
    # For hackathon demo purposes, enable debug mode for quick iteration.
    app.run(host="0.0.0.0", port=5000, debug=True)
