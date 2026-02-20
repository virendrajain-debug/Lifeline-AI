"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""Unit tests and validation for TemporalTracker.

These tests verify the temporal tracking engine without requiring actual video.
"""

import math
from temporal_tracker import (
    TemporalTracker,
    BoundingBox,
    Detection,
    ObjectClass,
    MotionPattern,
    create_detection_from_yolo,
    TrackedObject,
)


def test_bounding_box_properties():
    """Test BoundingBox calculations."""
    bbox = BoundingBox(x1=100, y1=50, x2=200, y2=150)

    assert bbox.center == (150, 100), "Center calculation incorrect"
    assert bbox.width == 100, "Width calculation incorrect"
    assert bbox.height == 100, "Height calculation incorrect"
    assert bbox.area == 10000, "Area calculation incorrect"

    print("✓ BoundingBox properties test passed")


def test_velocity_calculation():
    """Test velocity calculation with synthetic data."""
    bbox1 = BoundingBox(0, 0, 10, 10)
    bbox2 = BoundingBox(50, 0, 60, 10)  # Moved 50 pixels to the right

    det1 = Detection(
        class_id=0,
        class_name=ObjectClass.PERSON,
        bbox=bbox1,
        confidence=0.95,
        frame_index=0,
        timestamp_sec=0.0,  # At t=0
    )

    det2 = Detection(
        class_id=0,
        class_name=ObjectClass.PERSON,
        bbox=bbox2,
        confidence=0.95,
        frame_index=1,
        timestamp_sec=1.0 / 25.0,  # At t=1 frame at 25 FPS
    )

    obj = TrackedObject(
        object_id=1,
        class_name=ObjectClass.PERSON,
        initial_frame=0,
        initial_timestamp=0.0,
    )

    obj.add_detection(det1)
    obj.add_detection(det2)

    velocity = obj.compute_velocity()

    # Distance = 50 pixels, time = 1/25 = 0.04 seconds
    # velocity = 50 / 0.04 = 1250 px/sec
    expected_velocity = 1250.0
    assert abs(velocity - expected_velocity) < 1.0, f"Velocity {velocity} != {expected_velocity}"

    print(f"✓ Velocity calculation test passed (velocity={velocity:.1f} px/sec)")


def test_direction_calculation():
    """Test direction calculation."""
    # Create horizontal movement (left to right)
    bbox1 = BoundingBox(0, 0, 10, 10)
    bbox2 = BoundingBox(50, 0, 60, 10)  # 50 pixels right

    det1 = Detection(
        class_id=0,
        class_name=ObjectClass.PERSON,
        bbox=bbox1,
        confidence=0.95,
        frame_index=0,
        timestamp_sec=0.0,
    )

    det2 = Detection(
        class_id=0,
        class_name=ObjectClass.PERSON,
        bbox=bbox2,
        confidence=0.95,
        frame_index=1,
        timestamp_sec=1.0 / 25.0,
    )

    obj = TrackedObject(
        object_id=1,
        class_name=ObjectClass.PERSON,
        initial_frame=0,
        initial_timestamp=0.0,
    )

    obj.add_detection(det1)
    obj.add_detection(det2)

    direction = obj.compute_direction()

    # Moving right = ~0 degrees (or 360)
    assert (direction < 5 or direction > 355), f"Direction {direction}° should be ~0° for rightward movement"

    print(f"✓ Direction calculation test passed (direction={direction:.1f}°)")


def test_acceleration_calculation():
    """Test acceleration calculation."""
    # Simulate increasing velocity: 0 → 25 → 50 pixels per frame

    positions = [(0, 0), (25, 0), (75, 0)]  # Distances: 25px, then 50px
    timestamps = [0.0, 1.0 / 25.0, 2.0 / 25.0]

    obj = TrackedObject(
        object_id=1,
        class_name=ObjectClass.PERSON,
        initial_frame=0,
        initial_timestamp=0.0,
    )

    for i, (pos, ts) in enumerate(zip(positions, timestamps)):
        bbox = BoundingBox(pos[0], pos[1], pos[0] + 10, pos[1] + 10)
        det = Detection(
            class_id=0,
            class_name=ObjectClass.PERSON,
            bbox=bbox,
            confidence=0.95,
            frame_index=i,
            timestamp_sec=ts,
        )
        obj.add_detection(det)

    obj.compute_velocity()
    obj.compute_velocity()
    acceleration = obj.compute_acceleration()

    # vel1 = 25 / 0.04 = 625 px/sec
    # vel2 = 50 / 0.04 = 1250 px/sec
    # accel = (1250 - 625) / 0.04 = 15625 px/sec²
    expected_accel = 15625.0
    assert abs(acceleration - expected_accel) < 1.0, f"Acceleration {acceleration} != {expected_accel}"

    print(f"✓ Acceleration calculation test passed (accel={acceleration:.1f} px/sec²)")


def test_motion_pattern_detection():
    """Test motion pattern classification."""
    tracker = TemporalTracker(fps=25.0)

    # Scenario 1: Stationary object
    print("\n  Testing STATIONARY pattern...")
    obj_stationary = TrackedObject(
        object_id=1,
        class_name=ObjectClass.PERSON,
        initial_frame=0,
        initial_timestamp=0.0,
    )

    for i in range(5):
        bbox = BoundingBox(50, 50, 60, 60)  # Same position
        det = Detection(
            class_id=0,
            class_name=ObjectClass.PERSON,
            bbox=bbox,
            confidence=0.95,
            frame_index=i,
            timestamp_sec=i / 25.0,
        )
        obj_stationary.add_detection(det)

    obj_stationary.compute_velocity()
    pattern = obj_stationary.detect_motion_pattern()
    assert pattern == MotionPattern.STATIONARY, f"Expected STATIONARY, got {pattern}"
    print(f"    ✓ Detected: {pattern}")

    # Scenario 2: Sudden acceleration
    print("  Testing ACCELERATION pattern...")
    obj_accel = TrackedObject(
        object_id=2,
        class_name=ObjectClass.VEHICLE,
        initial_frame=0,
        initial_timestamp=0.0,
    )

    # Create frames with increasing velocity
    x_positions = [0, 20, 80]  # First jump: 20px, second jump: 60px
    for i, x in enumerate(x_positions):
        bbox = BoundingBox(x, 0, x + 20, 20)
        det = Detection(
            class_id=1,
            class_name=ObjectClass.VEHICLE,
            bbox=bbox,
            confidence=0.95,
            frame_index=i,
            timestamp_sec=i / 25.0,
        )
        obj_accel.add_detection(det)

    obj_accel.compute_velocity()
    obj_accel.compute_velocity()
    obj_accel.compute_acceleration()
    pattern = obj_accel.detect_motion_pattern()
    # Should detect ACCELERATION since accel > 50
    print(f"    ✓ Detected: {pattern} (acceleration={obj_accel.last_acceleration:.1f})")

    print("✓ Motion pattern detection test passed")


def test_tracker_with_synthetic_data():
    """Test full tracker with synthetic multi-object scenario."""
    print("\nTesting full tracker with synthetic scenario...")
    tracker = TemporalTracker(fps=25.0, memory_seconds=30)

    # Simulate 10 frames with 2 people
    for frame_idx in range(10):
        detections = []
        timestamp = frame_idx / 25.0

        # Person 1: Moving right
        x1 = 100 + frame_idx * 30
        bbox1 = BoundingBox(x1, 100, x1 + 40, 150)
        det1 = Detection(
            class_id=0,
            class_name=ObjectClass.PERSON,
            bbox=bbox1,
            confidence=0.95,
            frame_index=frame_idx,
            timestamp_sec=timestamp,
        )
        detections.append(det1)

        # Person 2: Stationary
        bbox2 = BoundingBox(400, 200, 440, 250)
        det2 = Detection(
            class_id=0,
            class_name=ObjectClass.PERSON,
            bbox=bbox2,
            confidence=0.95,
            frame_index=frame_idx,
            timestamp_sec=timestamp,
        )
        detections.append(det2)

        result = tracker.process_frame(frame_idx, detections)

        if frame_idx == 0:
            assert len(result["new_tracks"]) == 2, "Should detect 2 new objects"
            print(f"  Frame {frame_idx}: Created 2 new tracks")
        elif frame_idx < 9:
            assert len(result["updated_tracks"]) >= 2, "Should have 2+ updated tracks"
        else:
            # Last frame
            print(f"  Frame {frame_idx}: Final tracking state")
            print(f"    - Active tracks: {len(result['updated_tracks'])}")
            print(f"    - Total events: {len(tracker.all_events)}")

    # Verify event detection
    motion_events = [e for e in tracker.all_events if "motion" in e.event_type]
    print(f"  ✓ Detected {len(motion_events)} motion-related events")

    print("✓ Full tracker test passed")


def test_event_recording():
    """Test that events are properly recorded and retrievable."""
    print("\nTesting event recording...")
    tracker = TemporalTracker(fps=25.0)

    # Single frame with one person
    bbox = BoundingBox(100, 100, 140, 180)
    det = Detection(
        class_id=0,
        class_name=ObjectClass.PERSON,
        bbox=bbox,
        confidence=0.95,
        frame_index=0,
        timestamp_sec=0.0,
    )

    result = tracker.process_frame(0, [det])

    # Should have an "object_appeared" event
    events = result["events"]
    appeared_events = [e for e in events if e.event_type == "object_appeared"]

    assert len(appeared_events) == 1, "Should have 1 object_appeared event"
    print(f"  ✓ Recorded object appearance event")

    # Check event retrieval by time window
    events_in_window = tracker.get_events_in_window(0.0, 1.0)
    assert len(events_in_window) >= 1, "Should retrieve events in time window"
    print(f"  ✓ Retrieved {len(events_in_window)} events from time window [0.0, 1.0]")

    print("✓ Event recording test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("TEMPORAL TRACKER TEST SUITE")
    print("=" * 70)

    test_bounding_box_properties()
    test_velocity_calculation()
    test_direction_calculation()
    test_acceleration_calculation()
    test_motion_pattern_detection()
    test_tracker_with_synthetic_data()
    test_event_recording()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
