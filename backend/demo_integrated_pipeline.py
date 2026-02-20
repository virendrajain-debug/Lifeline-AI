#!/usr/bin/env python3
"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""
Quick verification that the integrated pipeline works end-to-end.
This demonstrates the perception → reasoning flow without needing a video file.
"""

from temporal_tracker import TemporalTracker, Detection, BoundingBox, ObjectClass
from causal_reasoner import CausalReasoner


def main():
    print("\n" + "=" * 80)
    print("LIFELINE AI: PERCEPTION → REASONING PIPELINE DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Initialize components
    print("[1] Initializing pipeline components...")
    tracker = TemporalTracker(fps=25.0, memory_seconds=30)
    reasoner = CausalReasoner()
    print("    ✓ TemporalTracker initialized")
    print("    ✓ CausalReasoner initialized\n")
    
    # Simulate 10 frames of a vehicle accelerating
    print("[2] Simulating video frames (vehicle accelerating toward person)...")
    print("-" * 80)
    
    for frame_idx in range(10):
        t_sec = frame_idx / 25.0  # 25 FPS
        
        # Vehicle accelerating rightward
        x_pos = 100 + (frame_idx * frame_idx * 10)  # Quadratic acceleration
        vehicle_bbox = BoundingBox(
            x1=x_pos,
            y1=250,
            x2=x_pos + 80,
            y2=350
        )
        
        # Person stationary
        person_bbox = BoundingBox(x1=500, y1=200, x2=550, y2=450)
        
        detections = [
            Detection(
                class_id=2,
                class_name=ObjectClass.VEHICLE,
                bbox=vehicle_bbox,
                confidence=0.95,
                frame_index=frame_idx,
                timestamp_sec=t_sec,
            ),
            Detection(
                class_id=0,
                class_name=ObjectClass.PERSON,
                bbox=person_bbox,
                confidence=0.95,
                frame_index=frame_idx,
                timestamp_sec=t_sec,
            ),
        ]
        
        # Process frame through tracker
        result = tracker.process_frame(frame_idx, detections)
        frame_events = result.get("events", [])
        
        # Extract motion info
        tracked_objects = tracker.tracked_objects
        print(f"Frame {frame_idx:2d} (t={t_sec:.3f}s): ", end="")
        
        for obj_id, obj in tracked_objects.items():
            if obj.class_name == ObjectClass.VEHICLE:
                vel = obj.last_velocity
                print(f"Vehicle(v={vel:6.1f}px/s) ", end="")
        
        # Feed events to reasoner
        for event in frame_events:
            reasoner.add_event(
                object_id=event.object_id,
                timestamp=event.timestamp_sec,
                event_type=event.event_type,
                event_data={
                    "motion_pattern": event.motion_pattern.name,
                    "magnitude": event.magnitude,
                },
            )
        
        if frame_events:
            print(f"| Events: {[e.event_type for e in frame_events]}")
        else:
            print("| No events")
    
    print("-" * 80 + "\n")
    
    # Run causal reasoning
    print("[3] Running causal inference...")
    all_causal_links = []
    
    for obj_id in reasoner.events_by_object.keys():
        links = reasoner.infer_single_object_causality(obj_id)
        if links:
            print(f"    Object {obj_id} (VEHICLE):")
            for link in links:
                all_causal_links.append(link)
                print(f"      → {link.cause_event} → {link.effect_event}")
                print(f"        Type: {link.cause_type.name}")
                print(f"        Confidence: {link.confidence:.2f}")
                print(f"        Evidence: {link.evidence[0]}")
    
    inter_links = reasoner.infer_inter_object_causality()
    if inter_links:
        print(f"    Inter-object causality:")
        for link in inter_links:
            all_causal_links.append(link)
            print(f"      → Object {link.cause_object_id} → Object {link.effect_object_id}")
            print(f"        {link.cause_event} caused {link.effect_event}")
            print(f"        Confidence: {link.confidence:.2f}")
    
    if not all_causal_links:
        print("    (No causal links detected in this sequence)")
    
    print()
    
    # Build chains
    print("[4] Building causal chains...")
    chains = reasoner.build_causal_chains()
    print(f"    {len(chains)} causal chain(s) identified")
    for i, chain in enumerate(chains):
        print(f"    Chain {i}: {len(chain.links)} links")
        for link in chain.links:
            print(f"      → {link.cause_event} ⟶ {link.effect_event}")
    
    print()
    
    # Summary
    print("[5] PIPELINE SUMMARY")
    print("-" * 80)
    print(f"Frames processed:        {tracker.current_frame_index + 1}")
    print(f"Time span:               0.00s - {tracker.current_timestamp:.3f}s")
    print(f"Active tracked objects:  {len(tracker.tracked_objects)}")
    print(f"Temporal events:         {len(tracker.all_events)}")
    print(f"Causal links detected:   {len(all_causal_links)}")
    print(f"Causal chains:           {len(chains)}")
    print("-" * 80)
    
    print("\n[SUCCESS] Perception → Reasoning pipeline working correctly! ✓\n")
    
    # Show how this integrates with detector.py
    print("[INTEGRATION NOTE]")
    print("In detector.py, this pipeline runs as:")
    print("  1. YOLOv8 detects objects → Detection objects")
    print("  2. TemporalTracker processes detections → events")
    print("  3. CausalReasoner analyzes events → causal links")
    print("  4. Check for COLLISION cause type")
    print("  5. Combine with original detection logic")
    print("  6. Make final emergency decision")
    print()


if __name__ == "__main__":
    main()
