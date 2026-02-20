"""
Lifeline AI
Copyright © 2026 Virendra Jain
All Rights Reserved.
Unauthorized commercial use is prohibited.
"""
from __future__ import annotations

"""Causal Reasoning Engine for LIFELINE AI.

This module consumes temporal events from temporal_tracker.py and infers
cause→effect relationships to build causal chains explaining incident dynamics.

It does NOT generate natural language explanations.
It does NOT classify incidents.
It ONLY answers: "What caused what, and why?"

Output is structured causal graphs and temporal chains that other modules
(forensic explanation, confidence scoring) can build upon.

============================================================================
[CORE PROPRIETARY LOGIC - Causal Reasoning & Incident Explanation]

The causal inference engine is the "reasoning layer" of Lifeline AI and
represents the most significant intellectual property in the platform.

KEY INNOVATIONS:
- Structured causal chain inference from temporal events
- Evidence-based confidence scoring (0.0-1.0 transparent metrics)
- Single-object causality (acceleration → collision patterns)
- Inter-object reaction detection (vehicle approach → defensive response)
- Temporal windowing for causality constraints

COMPETITIVE ADVANTAGES vs ALTERNATIVES:
- Not just detecting "something happened" - explaining WHY it happened
- Structured outputs enable downstream decision-making (hospital routing)
- Evidence tracking provides audit trail for liability protection
- Extensible design supports future reasoning modules

MONETIZATION STRATEGY:
- Core inference logic is SaaS-gated (API authentication required)
- Advanced reasoning modes (forensic narrative generation) = premium tier
- API access to causal chains = per-call billing ($0.10-$1.00 per query)
- Enterprise SaaS = unlimited reasoning queries

FUTURE ENHANCEMENTS:
- Multi-hypothesis reasoning (multiple competing causal explanations)
- Confidence aggregation across layers (detection → motion → causality)
- Forensic narrative generation (Phase 4)
- Q&A system: "What happened and why?" natural language interface

============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import math


class CauseType(str, Enum):
    """Classification of what caused an event."""

    HUMAN_ACTION = "human_action"  # Person initiated movement
    EXTERNAL_FORCE = "external_force"  # Vehicle, object collision
    REACTION = "reaction"  # Response to external stimulus
    ENVIRONMENTAL = "environmental"  # Weather, terrain, natural
    MECHANICAL = "mechanical"  # System failure, equipment
    UNKNOWN = "unknown"  # Cannot determine


class InteractionType(str, Enum):
    """Type of interaction between two objects."""

    APPROACH = "approach"  # Objects moving toward each other
    AVOIDANCE = "avoidance"  # Object moving away
    COLLISION = "collision"  # High-speed contact
    CONTACT = "contact"  # Touching but not high-speed
    PURSUIT = "pursuit"  # One following another
    FLEEING = "fleeing"  # One escaping from another
    PARALLEL = "parallel"  # Moving together
    CROSSING = "crossing"  # Paths crossing


class IncidentPhase(str, Enum):
    """Phase of incident development."""

    PRE_INCIDENT = "pre_incident"  # Setup/preconditions
    TRIGGER = "trigger"  # Initiating event
    ESCALATION = "escalation"  # Rapid development
    PEAK = "peak"  # Maximum impact
    RESOLUTION = "resolution"  # Conclusion/outcome


@dataclass
class CausalLink:
    """Single causal relationship: A caused B."""

    cause_object_id: int  # Object that initiated action
    effect_object_id: int  # Object affected
    cause_event: str  # What happened (event_type from temporal_tracker)
    effect_event: str  # What resulted
    cause_type: CauseType  # Type of causation
    time_cause: float  # Timestamp of cause event
    time_effect: float  # Timestamp of effect event
    time_delta: float  # Delay between cause and effect (seconds)
    confidence: float  # 0.0-1.0, how certain is this link?
    evidence: List[str] = field(default_factory=list)  # Reasoning steps

    def __hash__(self) -> int:
        """Make hashable for set operations."""
        return hash((self.cause_object_id, self.effect_object_id, self.time_cause, self.time_effect))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, CausalLink):
            return False
        return (
            self.cause_object_id == other.cause_object_id
            and self.effect_object_id == other.effect_object_id
            and abs(self.time_cause - other.time_cause) < 0.01
            and abs(self.time_effect - other.time_effect) < 0.01
        )


@dataclass
class CausalChain:
    """Ordered sequence of causal events building toward incident."""

    chain_id: int
    phase: IncidentPhase
    start_time: float
    end_time: float
    links: List[CausalLink] = field(default_factory=list)
    objects_involved: Set[int] = field(default_factory=set)
    description_key: str = ""  # Reference for incident type (not full description)

    @property
    def duration(self) -> float:
        """Duration of this phase in seconds."""
        return self.end_time - self.start_time

    @property
    def link_count(self) -> int:
        """Number of causal links in chain."""
        return len(self.links)


@dataclass
class ObjectBehavior:
    """Behavioral profile of tracked object over time interval."""

    object_id: int
    time_start: float
    time_end: float
    max_velocity: float
    mean_velocity: float
    acceleration_events: int  # Count of acceleration events
    deceleration_events: int  # Count of deceleration events
    direction_changes: int  # Count of significant direction changes
    motion_patterns: List[str] = field(default_factory=list)  # List of patterns observed
    is_stationary_at_end: bool = False


@dataclass
class IncidentContext:
    """Context around an incident."""

    start_time: float
    end_time: float
    primary_object_id: int  # Person with injury/incident
    secondary_objects: List[int] = field(default_factory=list)  # Vehicles, obstacles, etc.
    environment_type: str = "unknown"  # street, building, vehicle, etc.
    time_of_day: str = "unknown"  # day, night, dusk, etc.
    crowd_density: str = "unknown"  # low, medium, high


class CausalReasoner:
    """Main causal reasoning engine."""

    def __init__(self, time_window_seconds: float = 30.0):
        """Initialize reasoner.

        Args:
            time_window_seconds: Look-back window for causal analysis
        """
        self.time_window = time_window_seconds

        # All temporal events, indexed by object and time
        self.events_by_object: Dict[int, List[Dict]] = {}
        self.events_by_time: Dict[float, List[Dict]] = {}
        self.all_events: List[Dict] = []

        # Extracted causal links
        self.causal_links: Set[CausalLink] = set()

        # Built causal chains
        self.causal_chains: List[CausalChain] = []

        # Object behavior profiles
        self.object_behaviors: Dict[int, List[ObjectBehavior]] = {}

        # Incident context
        self.incident_context: Optional[IncidentContext] = None

        # Next available chain ID
        self.next_chain_id = 0

    def add_event(
        self,
        object_id: int,
        timestamp: float,
        event_type: str,
        event_data: Optional[Dict] = None,
    ) -> None:
        """Add a temporal event for causal analysis.

        Args:
            object_id: Which object this event concerns
            timestamp: When it happened
            event_type: Type of event (motion_pattern_changed, etc.)
            event_data: Additional context (magnitude, pattern, etc.)
        """
        event = {
            "object_id": object_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "data": event_data or {},
        }

        self.all_events.append(event)

        # Index by object
        if object_id not in self.events_by_object:
            self.events_by_object[object_id] = []
        self.events_by_object[object_id].append(event)

        # Index by time (rounded to nearest 0.1s to group simultaneous events)
        time_bucket = round(timestamp, 1)
        if time_bucket not in self.events_by_time:
            self.events_by_time[time_bucket] = []
        self.events_by_time[time_bucket].append(event)

    def infer_single_object_causality(self, object_id: int) -> List[CausalLink]:
        """Infer causality within single object's behavior (internal causes).

        Example: Person was stationary → accelerated → collision
        Causality: Person's acceleration caused their collision
        """
        if object_id not in self.events_by_object:
            return []

        events = sorted(self.events_by_object[object_id], key=lambda e: e["timestamp"])
        links = []

        for i in range(len(events) - 1):
            curr_event = events[i]
            next_event = events[i + 1]

            time_delta = next_event["timestamp"] - curr_event["timestamp"]

            # Pattern 1: Acceleration → Collision
            if (
                "acceleration" in curr_event["event_type"].lower()
                and "collision" in next_event["event_type"].lower()
            ):
                link = CausalLink(
                    cause_object_id=object_id,
                    effect_object_id=object_id,
                    cause_event=curr_event["event_type"],
                    effect_event=next_event["event_type"],
                    cause_type=CauseType.MECHANICAL,
                    time_cause=curr_event["timestamp"],
                    time_effect=next_event["timestamp"],
                    time_delta=time_delta,
                    confidence=0.85,
                    evidence=[
                        f"Object {object_id} showed acceleration",
                        f"Shortly after (Δt={time_delta:.2f}s), collision detected",
                        "Physical sequence: speed up → impact",
                    ],
                )
                links.append(link)

            # Pattern 2: Sudden Start → Motion
            if (
                "sudden_start" in curr_event["event_type"].lower()
                and ("acceleration" in next_event["event_type"].lower()
                     or "motion" in next_event["event_type"].lower())
            ):
                link = CausalLink(
                    cause_object_id=object_id,
                    effect_object_id=object_id,
                    cause_event=curr_event["event_type"],
                    effect_event=next_event["event_type"],
                    cause_type=CauseType.REACTION,
                    time_cause=curr_event["timestamp"],
                    time_effect=next_event["timestamp"],
                    time_delta=time_delta,
                    confidence=0.90,
                    evidence=[
                        f"Object {object_id} initiated motion suddenly",
                        f"Transition to sustained motion: Δt={time_delta:.2f}s",
                    ],
                )
                links.append(link)

            # Pattern 3: High Deceleration → Sudden Stop
            if (
                "deceleration" in curr_event["event_type"].lower()
                and "sudden_stop" in next_event["event_type"].lower()
            ):
                link = CausalLink(
                    cause_object_id=object_id,
                    effect_object_id=object_id,
                    cause_event=curr_event["event_type"],
                    effect_event=next_event["event_type"],
                    cause_type=CauseType.MECHANICAL,
                    time_cause=curr_event["timestamp"],
                    time_effect=next_event["timestamp"],
                    time_delta=time_delta,
                    confidence=0.95,
                    evidence=[
                        f"High deceleration event detected",
                        f"Object came to complete stop: Δt={time_delta:.2f}s",
                        "Mechanical causality: deceleration → stop",
                    ],
                )
                links.append(link)

        return links

    def infer_inter_object_causality(self) -> List[CausalLink]:
        """Infer causality between different objects.

        Example: Vehicle approaching → Person accelerating
        Causality: Vehicle approach caused person to panic/react
        """
        links = []
        objects = list(self.events_by_object.keys())

        if len(objects) < 2:
            return links

        # For each pair of objects (check both directions)
        for obj_a in objects:
            for obj_b in objects:
                if obj_a == obj_b:
                    continue

                events_a = sorted(self.events_by_object[obj_a], key=lambda e: e["timestamp"])
                events_b = sorted(self.events_by_object[obj_b], key=lambda e: e["timestamp"])

                # Check if A's event precedes and influences B's event
                for event_a in events_a:
                    for event_b in events_b:
                        time_delta = event_b["timestamp"] - event_a["timestamp"]

                        # Only consider causal if B happens shortly after A (0.05 - 5.0 seconds)
                        if not (0.05 < time_delta < 5.0):
                            continue

                        # Pattern 1: Acceleration (any object) → Person sudden_start
                        # This models threat detection and reaction
                        if (
                            "acceleration" in event_a["event_type"].lower()
                            and "sudden_start" in event_b["event_type"].lower()
                        ):
                            link = CausalLink(
                                cause_object_id=obj_a,
                                effect_object_id=obj_b,
                                cause_event=event_a["event_type"],
                                effect_event=event_b["event_type"],
                                cause_type=CauseType.REACTION,
                                time_cause=event_a["timestamp"],
                                time_effect=event_b["timestamp"],
                                time_delta=time_delta,
                                confidence=0.75,
                                evidence=[
                                    f"Object {obj_a} showed acceleration",
                                    f"Object {obj_b} reacted {time_delta:.2f}s later",
                                    "Temporal proximity suggests cause-effect",
                                    "Typical reaction pattern: threat detection → evasion",
                                ],
                            )
                            links.append(link)

                        # Pattern 2: Proximity + High velocity → Collision
                        if (
                            "close_proximity" in event_a["event_type"].lower()
                            and event_a["data"].get("magnitude", 0) > 0.8
                            and "collision" in event_b["event_type"].lower()
                        ):
                            link = CausalLink(
                                cause_object_id=obj_a,
                                effect_object_id=obj_b,
                                cause_event=event_a["event_type"],
                                effect_event=event_b["event_type"],
                                cause_type=CauseType.EXTERNAL_FORCE,
                                time_cause=event_a["timestamp"],
                                time_effect=event_b["timestamp"],
                                time_delta=time_delta,
                                confidence=0.90,
                                evidence=[
                                    f"Objects in extreme proximity (mag={event_a['data'].get('magnitude', 0):.2f})",
                                    f"Contact resulted in collision: Δt={time_delta:.2f}s",
                                    "Physical causality: proximity + contact force",
                                ],
                            )
                            links.append(link)

                        # Pattern 3: Object approach → Person deceleration
                        if (
                            "acceleration" in event_a["event_type"].lower()
                            and "deceleration" in event_b["event_type"].lower()
                        ):
                            # Could indicate person braking to avoid collision
                            link = CausalLink(
                                cause_object_id=obj_a,
                                effect_object_id=obj_b,
                                cause_event=event_a["event_type"],
                                effect_event=event_b["event_type"],
                                cause_type=CauseType.REACTION,
                                time_cause=event_a["timestamp"],
                                time_effect=event_b["timestamp"],
                                time_delta=time_delta,
                                confidence=0.65,
                                evidence=[
                                    f"Object {obj_a} accelerated",
                                    f"Object {obj_b} decelerated {time_delta:.2f}s later",
                                    "Possible evasive maneuver in response",
                                ],
                            )
                            links.append(link)

        return links

    def infer_causality(self) -> List[CausalLink]:
        """Infer all causal links from temporal events.

        Returns:
            List of CausalLink objects representing cause→effect relationships
        """
        links = []

        # Single-object causality (internal mechanical/behavioral)
        for obj_id in self.events_by_object.keys():
            links.extend(self.infer_single_object_causality(obj_id))

        # Inter-object causality (external forces, reactions)
        links.extend(self.infer_inter_object_causality())

        # Deduplicate and store
        self.causal_links = set(links)

        return list(self.causal_links)

    def build_causal_chains(self) -> List[CausalChain]:
        """Build ordered causal chains from links.

        A causal chain is a sequence of links where each effect becomes
        the cause for the next link.

        Returns:
            List of CausalChain objects
        """
        if not self.causal_links:
            return []

        # Sort links by time
        sorted_links = sorted(self.causal_links, key=lambda l: l.time_cause)

        chains = []
        used_links = set()

        # Build chains by following causal threads
        for start_link in sorted_links:
            if id(start_link) in used_links:
                continue

            chain = CausalChain(
                chain_id=self.next_chain_id,
                phase=IncidentPhase.TRIGGER,  # Will update based on analysis
                start_time=start_link.time_cause,
                end_time=start_link.time_effect,
                links=[start_link],
                objects_involved={start_link.cause_object_id, start_link.effect_object_id},
            )
            used_links.add(id(start_link))
            self.next_chain_id += 1

            # Try to extend chain: find next link where effect_object of current link
            # is cause_object of next link
            extended = True
            while extended:
                extended = False
                for next_link in sorted_links:
                    if id(next_link) in used_links:
                        continue

                    # Check if next_link follows logically from last link in chain
                    last_link = chain.links[-1]
                    if (
                        next_link.cause_object_id == last_link.effect_object_id
                        and next_link.time_cause <= last_link.time_effect + 2.0
                    ):
                        chain.links.append(next_link)
                        chain.end_time = next_link.time_effect
                        chain.objects_involved.add(next_link.effect_object_id)
                        used_links.add(id(next_link))
                        extended = True
                        break

            # Classify chain phase based on duration and intensity
            chain.phase = self._classify_chain_phase(chain)

            chains.append(chain)

        self.causal_chains = chains
        return chains

    def _classify_chain_phase(self, chain: CausalChain) -> IncidentPhase:
        """Classify incident phase based on causal chain characteristics."""
        if len(chain.links) == 0:
            return IncidentPhase.PRE_INCIDENT

        if len(chain.links) == 1:
            # Single link: either trigger or peak depending on impact
            link = chain.links[0]
            if "collision" in link.effect_event.lower() or "high_" in link.effect_event.lower():
                return IncidentPhase.PEAK
            else:
                return IncidentPhase.TRIGGER

        # Multiple links: sequence of events
        if len(chain.links) <= 2:
            return IncidentPhase.ESCALATION

        # Long chains indicate ongoing incident
        return IncidentPhase.RESOLUTION

    def get_interaction_type(self, obj_a_id: int, obj_b_id: int, time_window: float = 2.0) -> InteractionType:
        """Classify type of interaction between two objects.

        Args:
            obj_a_id: First object
            obj_b_id: Second object
            time_window: Look at events within this window

        Returns:
            InteractionType classification
        """
        if obj_a_id not in self.events_by_object or obj_b_id not in self.events_by_object:
            return InteractionType.PARALLEL

        events_a = self.events_by_object[obj_a_id]
        events_b = self.events_by_object[obj_b_id]

        # Simple heuristic: check event types
        event_types_a = set(e["event_type"] for e in events_a)
        event_types_b = set(e["event_type"] for e in events_b)

        # If one object shows sudden_start and the other shows acceleration,
        # could be pursuit
        if (
            "sudden_start" in event_types_a or "acceleration" in event_types_a
        ) and "motion_pattern" in event_types_b:
            # Check if velocities suggest one catching another
            return InteractionType.PURSUIT

        # If both decelerate simultaneously, could be avoiding
        if "deceleration" in event_types_a and "deceleration" in event_types_b:
            return InteractionType.AVOIDANCE

        # If collision events, direct contact
        if ("collision" in event_types_a or "close_proximity" in event_types_a) and (
            "collision" in event_types_b or "close_proximity" in event_types_b
        ):
            return InteractionType.CONTACT

        return InteractionType.PARALLEL

    def analyze_incident_context(self) -> IncidentContext:
        """Analyze and determine context of incident.

        Returns:
            IncidentContext with incident details
        """
        if not self.all_events:
            return IncidentContext(
                start_time=0.0,
                end_time=0.0,
                primary_object_id=0,
            )

        # Find time bounds
        start_time = min(e["timestamp"] for e in self.all_events)
        end_time = max(e["timestamp"] for e in self.all_events)

        # Find primary object (most events)
        object_event_counts = {}
        for event in self.all_events:
            obj_id = event["object_id"]
            object_event_counts[obj_id] = object_event_counts.get(obj_id, 0) + 1

        primary_object = max(object_event_counts, key=object_event_counts.get) if object_event_counts else 0
        secondary_objects = [
            obj_id for obj_id in object_event_counts.keys()
            if obj_id != primary_object
        ]

        context = IncidentContext(
            start_time=start_time,
            end_time=end_time,
            primary_object_id=primary_object,
            secondary_objects=secondary_objects,
        )

        self.incident_context = context
        return context

    def get_causal_chains(self) -> List[CausalChain]:
        """Return all causal chains (must call infer_causality and build_causal_chains first)."""
        return self.causal_chains

    def get_causal_links(self) -> List[CausalLink]:
        """Return all causal links."""
        return list(self.causal_links)

    def summarize_causal_graph(self) -> Dict:
        """Export causal graph structure (no natural language).

        Returns:
            Dictionary with nodes and edges for visualization/export
        """
        nodes = []
        edges = []

        # Create nodes for each link
        for i, link in enumerate(self.causal_links):
            nodes.append({
                "id": f"link_{i}",
                "cause_object": link.cause_object_id,
                "effect_object": link.effect_object_id,
                "cause_event": link.cause_event,
                "effect_event": link.effect_event,
                "cause_type": link.cause_type.value,
                "confidence": link.confidence,
                "time_delta": link.time_delta,
            })

        # Create edges for chains
        for chain in self.causal_chains:
            for i in range(len(chain.links) - 1):
                link_a = chain.links[i]
                link_b = chain.links[i + 1]

                edges.append({
                    "from_cause_object": link_a.cause_object_id,
                    "from_effect_object": link_a.effect_object_id,
                    "to_cause_object": link_b.cause_object_id,
                    "to_effect_object": link_b.effect_object_id,
                    "phase": chain.phase.value,
                    "time_gap": link_b.time_cause - link_a.time_effect,
                })

        return {
            "incident_context": self.incident_context.__dict__ if self.incident_context else None,
            "nodes": nodes,
            "edges": edges,
            "chains_count": len(self.causal_chains),
            "links_count": len(self.causal_links),
        }

    def reset(self) -> None:
        """Reset reasoner state (for new video)."""
        self.events_by_object.clear()
        self.events_by_time.clear()
        self.all_events.clear()
        self.causal_links.clear()
        self.causal_chains.clear()
        self.object_behaviors.clear()
        self.incident_context = None
        self.next_chain_id = 0
