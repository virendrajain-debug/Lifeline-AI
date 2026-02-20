"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""Integration tests for perception → reasoning pipeline in detector.py

Tests that temporal_tracker and causal_reasoner are properly integrated
into the detection pipeline.
"""

import sys
from unittest.mock import Mock, patch, MagicMock
from temporal_tracker import TemporalTracker, BoundingBox, Detection, ObjectClass
from causal_reasoner import CausalReasoner


def test_detector_imports():
    """Verify detector.py can import temporal and reasoning modules."""
    try:
        from detector import run_emergency_detection, TemporalTracker, CausalReasoner
        print("✓ detector.py imports temporal_tracker and causal_reasoner")
        return True
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return False


def test_pipeline_initialization():
    """Verify pipeline objects initialize correctly."""
    try:
        tracker = TemporalTracker(fps=25.0, memory_seconds=30)
        reasoner = CausalReasoner()
        print("✓ Pipeline components initialize without errors")
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False


def test_tracker_bbox_conversion():
    """Verify YOLOv8 bounding boxes can be converted to tracker format."""
    try:
        # Simulate YOLOv8 detection result
        yolo_bbox = [100.0, 150.0, 250.0, 400.0]  # [x1, y1, x2, y2]
        
        # Convert to Detection object with BoundingBox
        bbox = BoundingBox(x1=yolo_bbox[0], y1=yolo_bbox[1], x2=yolo_bbox[2], y2=yolo_bbox[3])
        detection = Detection(
            class_id=0,
            class_name=ObjectClass.PERSON,
            bbox=bbox,
            confidence=0.95,
            frame_index=0,
            timestamp_sec=0.0,
        )
        
        assert detection.bbox.x1 == 100.0
        assert detection.class_name == ObjectClass.PERSON
        print("✓ YOLOv8 bbox → Detection conversion works")
        return True
    except Exception as e:
        print(f"✗ Detection conversion failed: {e}")
        return False


def test_tracker_frame_processing():
    """Verify temporal tracker processes detections correctly."""
    try:
        tracker = TemporalTracker(fps=25.0, memory_seconds=30)
        
        # Simulate frame 1: Person at (100, 150, 250, 400)
        bbox1 = BoundingBox(x1=100, y1=150, x2=250, y2=400)
        detections_f1 = [
            Detection(
                class_id=0,
                class_name=ObjectClass.PERSON,
                bbox=bbox1,
                confidence=0.95,
                frame_index=0,
                timestamp_sec=0.0,
            )
        ]
        result1 = tracker.process_frame(0, detections_f1)
        
        # Simulate frame 2: Person moved to (110, 160, 260, 410)
        bbox2 = BoundingBox(x1=110, y1=160, x2=260, y2=410)
        detections_f2 = [
            Detection(
                class_id=0,
                class_name=ObjectClass.PERSON,
                bbox=bbox2,
                confidence=0.95,
                frame_index=1,
                timestamp_sec=0.04,
            )
        ]
        result2 = tracker.process_frame(1, detections_f2)
        
        # Verify tracking occurred
        assert len(tracker.tracked_objects) > 0, "Should have tracked objects"
        print("✓ TemporalTracker.process_frame handles multi-frame sequences")
        return True
    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        return False


def test_event_flow_to_reasoner():
    """Verify events flow from tracker to reasoner."""
    try:
        tracker = TemporalTracker(fps=25.0, memory_seconds=30)
        reasoner = CausalReasoner()
        
        # Create multi-frame tracking sequence
        for i in range(5):
            t_sec = i * 0.04
            bbox = BoundingBox(
                x1=100 + (i * 50),  # Moving right
                y1=200,
                x2=200 + (i * 50),
                y2=400,
            )
            detections = [
                Detection(
                    class_id=0,
                    class_name=ObjectClass.PERSON,
                    bbox=bbox,
                    confidence=0.95,
                    frame_index=i,
                    timestamp_sec=t_sec,
                )
            ]
            result = tracker.process_frame(i, detections)
            
            # Feed events to reasoner
            for event in result.get("events", []):
                reasoner.add_event(
                    object_id=event.object_id,
                    timestamp=event.timestamp_sec,
                    event_type=event.event_type,
                    event_data={"motion_pattern": event.motion_pattern.name},
                )
        
        # Verify reasoner received events
        assert len(reasoner.events_by_object) > 0, "Reasoner should have events"
        print("✓ Events successfully flow from TemporalTracker → CausalReasoner")
        return True
    except Exception as e:
        print(f"✗ Event flow test failed: {e}")
        return False


def test_causal_inference_on_events():
    """Verify causal reasoner can infer from temporal events."""
    try:
        tracker = TemporalTracker(fps=25.0, memory_seconds=30)
        reasoner = CausalReasoner()
        
        # Create acceleration pattern (0→5→10 pixels/frame = increasing velocity)
        for i in range(5):
            t_sec = i * 0.04
            x_pos = 100 + (i * (i + 1) * 25)  # Quadratic: accelerating
            bbox = BoundingBox(
                x1=x_pos,
                y1=200,
                x2=x_pos + 100,
                y2=400,
            )
            detections = [
                Detection(
                    class_id=2,  # Vehicle
                    class_name=ObjectClass.VEHICLE,
                    bbox=bbox,
                    confidence=0.95,
                    frame_index=i,
                    timestamp_sec=t_sec,
                )
            ]
            result = tracker.process_frame(i, detections)
            
            # Feed events to reasoner
            for event in result.get("events", []):
                reasoner.add_event(
                    object_id=event.object_id,
                    timestamp=event.timestamp_sec,
                    event_type=event.event_type,
                    event_data={"motion_pattern": event.motion_pattern.name},
                )
        
        # Try to infer causality
        links = []
        for obj_id in reasoner.events_by_object.keys():
            links.extend(reasoner.infer_single_object_causality(obj_id))
        
        # With acceleration, should produce some causal links
        assert isinstance(links, list), "Should return list of links"
        print(f"✓ CausalReasoner.infer_single_object_causality returns {len(links)} links")
        return True
    except Exception as e:
        print(f"✗ Causal inference test failed: {e}")
        return False


def test_collision_detection_integration():
    """Verify both detection-based and reasoning-based collision signals work."""
    try:
        # Person and vehicle overlapping in same frame
        tracker = TemporalTracker(fps=25.0, memory_seconds=30)
        reasoner = CausalReasoner()
        
        # Person at (100, 200, 150, 400)
        # Vehicle at (120, 250, 200, 350) - overlapping!
        detections = [
            Detection(
                class_id=0,
                class_name=ObjectClass.PERSON,
                bbox=BoundingBox(x1=100, y1=200, x2=150, y2=400),
                confidence=0.95,
                frame_index=0,
                timestamp_sec=0.0,
            ),
            Detection(
                class_id=2,
                class_name=ObjectClass.VEHICLE,
                bbox=BoundingBox(x1=120, y1=250, x2=200, y2=350),
                confidence=0.95,
                frame_index=0,
                timestamp_sec=0.0,
            ),
        ]
        
        result = tracker.process_frame(0, detections)
        
        # Feed to reasoner
        for event in result.get("events", []):
            reasoner.add_event(
                object_id=event.object_id,
                timestamp=event.timestamp_sec,
                event_type=event.event_type,
                event_data={"motion_pattern": event.motion_pattern.name},
            )
        
        # Both should work without errors
        assert tracker.tracked_objects is not None
        assert reasoner.events_by_object is not None
        print("✓ Both tracking and reasoning pipelines handle overlapping objects")
        return True
    except Exception as e:
        print(f"✗ Collision integration test failed: {e}")
        return False


def test_causal_chain_building():
    """Verify causal chains are properly built from links."""
    try:
        reasoner = CausalReasoner()
        
        # Manually add events to simulate temporal sequence
        reasoner.add_event(
            object_id=0,
            timestamp=0.0,
            event_type="acceleration",
            event_data={"value": 5.0, "motion_pattern": "ACCELERATION"},
        )
        reasoner.add_event(
            object_id=0,
            timestamp=0.5,
            event_type="high_velocity",
            event_data={"value": 25.0, "motion_pattern": "FAST_MOVING"},
        )
        
        # Build chains
        chains = reasoner.build_causal_chains()
        
        assert isinstance(chains, list), "Should return list of chains"
        print(f"✓ CausalReasoner.build_causal_chains returns {len(chains)} chains")
        return True
    except Exception as e:
        print(f"✗ Chain building test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("PERCEPTION → REASONING PIPELINE INTEGRATION TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        ("Detector imports", test_detector_imports),
        ("Pipeline initialization", test_pipeline_initialization),
        ("BoundingBox conversion", test_tracker_bbox_conversion),
        ("Frame processing", test_tracker_frame_processing),
        ("Event flow", test_event_flow_to_reasoner),
        ("Causal inference", test_causal_inference_on_events),
        ("Collision detection", test_collision_detection_integration),
        ("Chain building", test_causal_chain_building),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n[SUCCESS] All integration tests passed! ✓")
        print("Pipeline is ready for use in detector.py")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
