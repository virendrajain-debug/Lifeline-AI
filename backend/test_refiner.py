"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""

"""Test suite for LIFELINE AI PIPELINE REFINER MODE.

Tests:
1. EventFilter - Low-confidence events are filtered
2. Context awareness - Normal activity in crowds doesn't trigger false emergencies
3. EmergencyValidator - Multi-cue validation prevents false positives
4. Integration - Detector with confidence filtering works end-to-end
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, '/'.join(__file__.split('/')[:-1]))

from detector import EventFilter, EmergencyValidator, run_emergency_detection


class TestEventFilter(unittest.TestCase):
    """Test EventFilter confidence filtering."""
    
    def setUp(self):
        self.filter = EventFilter(confidence_threshold=0.7)
    
    def test_high_confidence_event_forwarded(self):
        """High-confidence event should pass filter."""
        event = MagicMock()
        event.magnitude = 80.0
        event.motion_pattern = MagicMock()
        event.motion_pattern.name = 'ACCELERATION'
        
        context = {
            'location_type': 'road',
            'crowd_density': 'low',
            'lighting': 'night',
            'isolation_level': 'isolated',
        }
        
        # High detection confidence (0.95) should pass
        should_forward, reason = self.filter.should_forward_event(event, context)
        self.assertTrue(should_forward, f"Should forward high-confidence event, got: {reason}")
    
    def test_low_confidence_event_filtered(self):
        """Low-confidence event should be filtered."""
        event = MagicMock()
        event.magnitude = 10.0  # Very low magnitude
        event.motion_pattern = MagicMock()
        event.motion_pattern.name = 'NORMAL'
        
        context = {'crowd_density': 'high'}
        
        should_forward, reason = self.filter.should_forward_event(event, context)
        self.assertFalse(should_forward, f"Should filter low-confidence event, got: {reason}")
    
    def test_normal_motion_in_crowd_filtered(self):
        """Normal motion in crowd should be filtered."""
        event = MagicMock()
        event.magnitude = 50.0
        event.motion_pattern = MagicMock()
        event.motion_pattern.name = 'NORMAL'
        
        context = {
            'location_type': 'market',
            'crowd_density': 'high',
            'lighting': 'day',
        }
        
        should_forward, reason = self.filter.should_forward_event(event, context)
        self.assertFalse(should_forward, "Normal motion in crowd should be filtered")


class TestEmergencyValidator(unittest.TestCase):
    """Test EmergencyValidator multi-cue validation."""
    
    def setUp(self):
        self.validator = EmergencyValidator()
    
    def test_high_confidence_collision_triggers_emergency(self):
        """High-confidence collision should trigger emergency."""
        context = {
            'location_type': 'road',
            'crowd_density': 'low',
            'isolation_level': 'isolated',
        }
        
        is_emergency, reason = self.validator.assess_emergency(
            collision_detected=True,
            lying_detected=False,
            event_type='collision',
            event_confidence=0.90,
            context=context,
        )
        
        self.assertTrue(is_emergency, f"High-confidence collision should trigger emergency: {reason}")
    
    def test_low_confidence_collision_no_emergency(self):
        """Low-confidence collision should not trigger emergency."""
        context = {
            'location_type': 'market',
            'crowd_density': 'high',
        }
        
        is_emergency, reason = self.validator.assess_emergency(
            collision_detected=True,
            lying_detected=False,
            event_type='collision',
            event_confidence=0.50,
            context=context,
        )
        
        self.assertFalse(is_emergency, "Low-confidence collision should not trigger emergency")
    
    def test_lying_in_isolated_area_triggers_emergency(self):
        """Lying in isolated area with high confidence should trigger emergency."""
        context = {
            'location_type': 'isolated_road',
            'lighting': 'night',
            'isolation_level': 'isolated',
            'crowd_density': 'low',
        }
        
        is_emergency, reason = self.validator.assess_emergency(
            collision_detected=False,
            lying_detected=True,
            event_type='motion_stop',
            event_confidence=0.85,
            context=context,
        )
        
        self.assertTrue(is_emergency, f"Lying in isolated/night context should trigger: {reason}")
    
    def test_lying_in_crowded_market_no_emergency(self):
        """Lying in crowded market (normal context) should NOT trigger emergency."""
        context = {
            'location_type': 'market',
            'lighting': 'day',
            'isolation_level': 'urban',
            'crowd_density': 'high',
        }
        
        is_emergency, reason = self.validator.assess_emergency(
            collision_detected=False,
            lying_detected=True,
            event_type='motion_stop',
            event_confidence=0.75,
            context=context,
        )
        
        # In a crowded market, lying down might be someone resting - less likely to be emergency
        self.assertFalse(is_emergency, f"Lying in crowded market should NOT trigger: {reason}")
    
    def test_multiple_signals_with_high_confidence(self):
        """Multiple danger signals + high confidence should trigger emergency."""
        context = {
            'location_type': 'street',
            'crowd_density': 'low',
        }
        
        is_emergency, reason = self.validator.assess_emergency(
            collision_detected=True,
            lying_detected=True,
            event_type='collision',
            event_confidence=0.82,
            context=context,
        )
        
        self.assertTrue(is_emergency, f"Multiple signals should trigger: {reason}")


class TestDetectorIntegration(unittest.TestCase):
    """Test detector integration with refiner mode."""
    
    def test_detector_accepts_context_parameter(self):
        """Detector should accept context parameter."""
        context = {
            'location_type': 'market',
            'crowd_density': 'high',
            'lighting': 'day',
            'isolation_level': 'urban',
        }
        
        # Call with context - should not raise error
        try:
            emergency, reason, timeline = run_emergency_detection(
                max_seconds=1,
                video_path=None,
                context=context,
            )
            # Success - no exception
            self.assertTrue(isinstance(emergency, bool))
            self.assertTrue(isinstance(reason, str))
        except TypeError as e:
            self.fail(f"Detector should accept context parameter: {e}")
    
    def test_detector_default_context(self):
        """Detector should use default context when not provided."""
        # Call without context - should use default
        try:
            emergency, reason, timeline = run_emergency_detection(
                max_seconds=1,
                video_path=None,
            )
            # Success - no exception
            self.assertTrue(isinstance(emergency, bool))
        except TypeError as e:
            self.fail(f"Detector should work without context: {e}")


class TestContextAwareFiltering(unittest.TestCase):
    """Test context-aware event filtering scenarios."""
    
    def test_market_daytime_normal_activity_not_emergency(self):
        """Normal walking in crowded market during day = no emergency."""
        filter = EventFilter(confidence_threshold=0.7)
        
        event = MagicMock()
        event.magnitude = 30.0  # Moderate movement
        event.motion_pattern = MagicMock()
        event.motion_pattern.name = 'NORMAL'
        
        context = {
            'location_type': 'market',
            'crowd_density': 'high',
            'lighting': 'day',
            'isolation_level': 'urban',
        }
        
        should_forward, reason = filter.should_forward_event(event, context)
        self.assertFalse(should_forward, "Normal activity in crowded market should be filtered")
    
    def test_isolated_road_night_motion_pattern_concerning(self):
        """Abnormal motion in isolated road at night = concerning."""
        filter = EventFilter(confidence_threshold=0.7)
        
        event = MagicMock()
        event.magnitude = 70.0  # High magnitude
        event.motion_pattern = MagicMock()
        event.motion_pattern.name = 'ACCELERATION'
        
        context = {
            'location_type': 'isolated_road',
            'crowd_density': 'low',
            'lighting': 'night',
            'isolation_level': 'isolated',
        }
        
        should_forward, reason = filter.should_forward_event(event, context)
        # Should forward high-magnitude acceleration event
        self.assertTrue(should_forward, f"High-magnitude acceleration should be forwarded: {reason}")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEventFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestEmergencyValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestContextAwareFiltering))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED - REFINER MODE WORKING CORRECTLY")
        print("="*70)
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
