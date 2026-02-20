## Quick Start Guide

### 1. Prerequisites

Ensure Python 3.7+ is installed with required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Demo (No Video File Needed)

Test the complete perception → reasoning pipeline:

```bash
python demo_integrated_pipeline.py
```

**What you'll see:**
- Frame-by-frame motion analysis (velocity, acceleration)
- Temporal events (acceleration, object appearance/disappearance)
- Causal inference pipeline initialization
- Summary statistics (frames processed, events, causal chains)

**Expected output:**
```
✓ TemporalTracker initialized
✓ CausalReasoner initialized
Frame  0 (t=0.000s): Vehicle(v=   0.0px/s) | Events: [...]
...
[SUCCESS] Perception → Reasoning pipeline working correctly! ✓
```

### 3. Run the Web Server

Start the Flask backend:

```bash
python app.py
```

Access at: `http://127.0.0.1:5000`

**Features:**
- Upload or select video for analysis
- View detection results
- See incident reports with timestamps
- Access hospital routing information

### 4. Run All Tests

Verify the system components:

```bash
# Temporal tracking tests (7 tests)
python test_temporal_tracker.py

# Causal reasoning tests (11 tests)
python test_causal_reasoner.py

# Integration tests (8 tests)
python test_integration.py

# Verification checks (5 checks)
python verify_temporal_tracker.py
```

**Expected result:** All 31 tests passing ✓

### 5. Run Detection on a Video

To analyze a specific video file:

```python
from detector import run_emergency_detection

# Analyze video file
emergency, reason, timeline = run_emergency_detection(
    max_seconds=30,
    video_path="path/to/your/video.mp4"
)

print(f"Emergency detected: {emergency}")
print(f"Reason: {reason}")
print(f"Timeline events: {len(timeline)}")
```

### Key Files

| File | Purpose |
|------|---------|
| `demo_integrated_pipeline.py` | Full pipeline demo (no video needed) |
| `app.py` | Flask web server |
| `detector.py` | Main detection engine + adapters |
| `temporal_tracker.py` | Motion tracking module |
| `causal_reasoner.py` | Causal reasoning module |
| `test_*.py` | Test suites |
| `README.md` | Full documentation |

### Common Issues

**"Model not found"**
- First run downloads YOLOv8 model (~140MB)
- Internet connection required for first run only
- Model is cached locally afterward

**"Video file not found"**
- Place video in `video/` directory
- Or use absolute path: `/path/to/video.mp4`
- Demo works without video file

**Tests failing**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: 3.7+
- Run from backend directory: `cd backend/`

### Next Steps

- Review `README.md` for architecture details
- Explore test files to understand module APIs
- Read module docstrings for implementation details
- Check inline comments for algorithm explanations

---

**Status:** ✓ Ready to run locally

**Test Coverage:** 31/31 tests passing
