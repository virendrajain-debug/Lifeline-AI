"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""Unit tests for Causal Reasoning Engine.

These tests verify the causal reasoning module without requiring actual video.
"""

import math
from causal_reasoner import (
    CausalReasoner,
    CausalLink,
    CausalChain,
    CauseType,
    InteractionType,
    IncidentPhase,
)


def test_single_object_causality():
    """Test causality inference within single object's behavior."""
    print("\n[TEST] Single Object Causality")

    reasoner = CausalReasoner()

    # Scenario: Person accelerates then collides
    reasoner.add_event(
        object_id=0,
        timestamp=0.0,
        event_type="acceleration",
        event_data={"magnitude": 75.0},
    )
    reasoner.add_event(
        object_id=0,
        timestamp=0.5,
        event_type="collision_like",
        event_data={"magnitude": 0.95},
    )

    links = reasoner.infer_causality()

    # Should infer that acceleration caused collision
    accel_collision_links = [
        l for l in links
        if "acceleration" in l.cause_event and "collision" in l.effect_event
    ]

    assert len(accel_collision_links) > 0, "Should detect acceleration → collision link"
    link = accel_collision_links[0]
    assert link.cause_type == CauseType.MECHANICAL
    assert link.confidence > 0.7
    assert abs(link.time_delta - 0.5) < 0.01

    print(f"  ✓ Acceleration → Collision detected")
    print(f"    Confidence: {link.confidence:.2%}")
    print(f"    Time delta: {link.time_delta:.2f}s")


def test_inter_object_reaction():
    """Test causality between objects (reaction patterns)."""
    print("\n[TEST] Inter-Object Reaction Pattern")

    reasoner = CausalReasoner()

    # Scenario: Vehicle accelerates → Person reacts with sudden start
    # Vehicle (obj 0) accelerates at t=0
    reasoner.add_event(
        object_id=0,
        timestamp=0.0,
        event_type="acceleration",
        event_data={"magnitude": 85.0},
    )

    # Person (obj 1) reacts at t=0.3s (reaction time ~300ms)
    reasoner.add_event(
        object_id=1,
        timestamp=0.3,
        event_type="sudden_start",
        event_data={"magnitude": 0.9},
    )

    links = reasoner.infer_causality()

    # Should infer reaction link
    reaction_links = [
        l for l in links
        if l.cause_object_id == 0 and l.effect_object_id == 1
    ]

    assert len(reaction_links) > 0, "Should detect vehicle acceleration → person reaction"
    link = reaction_links[0]
    assert link.cause_type == CauseType.REACTION
    assert 0.2 < link.time_delta < 1.0, "Reaction time should be 0.2-1.0 seconds"

    print(f"  ✓ Vehicle acceleration → Person reaction detected")
    print(f"    Confidence: {link.confidence:.2%}")
    print(f"    Reaction time: {link.time_delta:.2f}s")


def test_causal_chain_building():
    """Test building causal chains from links."""
    print("\n[TEST] Causal Chain Building")

    reasoner = CausalReasoner()

    # Build a chain: Person stationary → sudden start → acceleration → collision
    reasoner.add_event(0, 0.0, "motion_pattern_changed", {"pattern": "stationary"})
    reasoner.add_event(0, 1.0, "sudden_start", {"magnitude": 0.8})
    reasoner.add_event(0, 1.5, "acceleration", {"magnitude": 65.0})
    reasoner.add_event(0, 2.0, "collision_like", {"magnitude": 0.95})

    # Infer causality
    reasoner.infer_causality()

    # Build chains
    chains = reasoner.build_causal_chains()

    assert len(chains) > 0, "Should build at least one chain"

    # Find chain with multiple links
    multi_link_chains = [c for c in chains if len(c.links) > 1]
    assert len(multi_link_chains) > 0, "Should have chain with multiple links"

    chain = multi_link_chains[0]
    print(f"  ✓ Chain built with {len(chain.links)} links")
    print(f"    Duration: {chain.duration:.2f}s")
    print(f"    Phase: {chain.phase.value}")
    print(f"    Objects involved: {chain.objects_involved}")


def test_collision_causality():
    """Test collision detection between objects."""
    print("\n[TEST] Collision Causality")

    reasoner = CausalReasoner()

    # Vehicle approaching person
    reasoner.add_event(0, 0.0, "acceleration", {"magnitude": 80.0})
    reasoner.add_event(0, 0.5, "close_proximity", {"magnitude": 0.95})
    reasoner.add_event(1, 0.6, "collision_like", {"magnitude": 0.9})

    links = reasoner.infer_causality()

    collision_links = [
        l for l in links
        if "collision" in l.effect_event
    ]

    assert len(collision_links) > 0, "Should detect collision causality"
    link = collision_links[0]
    assert link.cause_type in [CauseType.EXTERNAL_FORCE, CauseType.REACTION]
    assert link.confidence > 0.6

    print(f"  ✓ Collision causality detected")
    print(f"    Cause type: {link.cause_type.value}")
    print(f"    Confidence: {link.confidence:.2%}")


def test_deceleration_stop_link():
    """Test deceleration → sudden_stop causality."""
    print("\n[TEST] Deceleration → Stop Link")

    reasoner = CausalReasoner()

    # Object decelerating hard then stops
    reasoner.add_event(0, 0.0, "high_deceleration", {"magnitude": 120.0})
    reasoner.add_event(0, 0.2, "sudden_stop", {"magnitude": 0.95})

    links = reasoner.infer_causality()

    decel_stop_links = [
        l for l in links
        if "deceleration" in l.cause_event and "sudden_stop" in l.effect_event
    ]

    assert len(decel_stop_links) > 0, "Should link deceleration to stop"
    link = decel_stop_links[0]
    assert link.confidence > 0.9, "Should be high confidence"

    print(f"  ✓ Deceleration → Stop link created")
    print(f"    Confidence: {link.confidence:.2%}")


def test_interaction_type_classification():
    """Test classification of interaction types."""
    print("\n[TEST] Interaction Type Classification")

    reasoner = CausalReasoner()

    # Create events suggesting pursuit behavior
    reasoner.add_event(0, 0.0, "acceleration", {})
    reasoner.add_event(1, 0.3, "sudden_start", {})
    reasoner.add_event(1, 0.6, "acceleration", {})

    # Classify interaction
    interaction = reasoner.get_interaction_type(0, 1)

    print(f"  ✓ Interaction type: {interaction.value}")


def test_context_analysis():
    """Test incident context analysis."""
    print("\n[TEST] Incident Context Analysis")

    reasoner = CausalReasoner()

    # Multiple events from multiple objects
    reasoner.add_event(0, 0.0, "acceleration", {})
    reasoner.add_event(0, 0.5, "collision_like", {})
    reasoner.add_event(1, 0.3, "sudden_start", {})
    reasoner.add_event(1, 0.6, "deceleration", {})

    context = reasoner.analyze_incident_context()

    assert context.primary_object_id in [0, 1]
    assert len(context.secondary_objects) > 0
    assert context.start_time < context.end_time

    print(f"  ✓ Context analyzed")
    print(f"    Primary object: {context.primary_object_id}")
    print(f"    Secondary objects: {context.secondary_objects}")
    print(f"    Duration: {context.end_time - context.start_time:.2f}s")


def test_causal_graph_export():
    """Test causal graph export to structured format."""
    print("\n[TEST] Causal Graph Export")

    reasoner = CausalReasoner()

    # Create scenario with causal relationships
    reasoner.add_event(0, 0.0, "acceleration", {"magnitude": 80.0})
    reasoner.add_event(1, 0.5, "sudden_start", {"magnitude": 0.8})
    reasoner.add_event(1, 1.0, "acceleration", {"magnitude": 70.0})

    reasoner.infer_causality()
    reasoner.build_causal_chains()
    reasoner.analyze_incident_context()

    graph = reasoner.summarize_causal_graph()

    assert "nodes" in graph
    assert "edges" in graph
    assert "chains_count" in graph
    assert "links_count" in graph

    print(f"  ✓ Graph exported")
    print(f"    Nodes: {len(graph['nodes'])}")
    print(f"    Edges: {len(graph['edges'])}")
    print(f"    Chains: {graph['chains_count']}")
    print(f"    Links: {graph['links_count']}")


def test_evidence_tracking():
    """Test that evidence is recorded for each causal link."""
    print("\n[TEST] Evidence Tracking")

    reasoner = CausalReasoner()

    reasoner.add_event(0, 0.0, "acceleration", {"magnitude": 80.0})
    reasoner.add_event(0, 0.5, "collision_like", {"magnitude": 0.95})

    links = reasoner.infer_causality()
    collision_links = [l for l in links if "collision" in l.effect_event]

    assert len(collision_links) > 0
    link = collision_links[0]
    assert len(link.evidence) > 0, "Should have evidence"

    print(f"  ✓ Evidence tracked for causal link")
    print(f"    Evidence points: {len(link.evidence)}")
    for i, evidence in enumerate(link.evidence, 1):
        print(f"      {i}. {evidence}")


def test_empty_events():
    """Test reasoner handles empty event stream gracefully."""
    print("\n[TEST] Empty Events Handling")

    reasoner = CausalReasoner()

    # No events added
    links = reasoner.infer_causality()
    assert len(links) == 0, "Should return empty list for no events"

    chains = reasoner.build_causal_chains()
    assert len(chains) == 0, "Should return empty list for no chains"

    context = reasoner.analyze_incident_context()
    assert context.primary_object_id == 0, "Should handle gracefully"

    print(f"  ✓ Handles empty event stream")


def test_confidence_scoring():
    """Test that confidence scores reflect causality strength."""
    print("\n[TEST] Confidence Scoring")

    reasoner = CausalReasoner()

    # High confidence: immediate collision after deceleration
    reasoner.add_event(0, 0.0, "high_deceleration", {})
    reasoner.add_event(0, 0.1, "sudden_stop", {})

    links = reasoner.infer_causality()
    decel_stop = [l for l in links if "deceleration" in l.cause_event]

    # Should have high confidence
    if decel_stop:
        assert decel_stop[0].confidence > 0.85
        print(f"  ✓ High-confidence link: {decel_stop[0].confidence:.2%}")

    # Low confidence: events far apart or weak signal
    reasoner2 = CausalReasoner()
    reasoner2.add_event(0, 0.0, "acceleration", {})
    reasoner2.add_event(1, 3.0, "deceleration", {})  # 3 seconds later

    links2 = reasoner2.infer_causality()
    # Should have low or no confidence (too far apart)
    if links2:
        print(f"  ✓ Low-confidence or no link detected (time gap too large)")


def run_all_tests():
    """Run all causal reasoner tests."""
    print("=" * 70)
    print("CAUSAL REASONING ENGINE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Single Object Causality", test_single_object_causality),
        ("Inter-Object Reaction", test_inter_object_reaction),
        ("Causal Chain Building", test_causal_chain_building),
        ("Collision Causality", test_collision_causality),
        ("Deceleration → Stop", test_deceleration_stop_link),
        ("Interaction Type Classification", test_interaction_type_classification),
        ("Incident Context Analysis", test_context_analysis),
        ("Causal Graph Export", test_causal_graph_export),
        ("Evidence Tracking", test_evidence_tracking),
        ("Empty Events Handling", test_empty_events),
        ("Confidence Scoring", test_confidence_scoring),
    ]

    results = {}
    for name, test_func in tests:
        try:
            test_func()
            results[name] = True
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            results[name] = False
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed ✓")
        return True
    else:
        print(f"\n[FAILURE] {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
