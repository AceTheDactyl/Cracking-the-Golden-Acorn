"""
Grassmann Narrative Engine
==========================

Executes narrative arcs as 32-operator sequences on the Grassmann manifold.
Maps story beats to APL operator applications, tracking phase transitions
and Z-coordinate evolution.

Key Components:
- NarrativeState: Current state of narrative progression
- StoryBeat: Individual narrative moment
- GrassmannNarrativeEngine: Executes narrative sequences

The 32-operator structure represents a complete narrative cycle:
- Setup (8 ops): Establish ground, introduce elements
- Rising (8 ops): Build tension, develop conflicts
- Climax (8 ops): Peak intensity, critical transitions
- Resolution (8 ops): Resolve tensions, integrate changes

Each operator application moves the narrative point on the manifold,
with phase transitions marking major story beats.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum, auto

from .manifold import GrassmannManifold, GrassmannPoint
from .metrics import z_from_points, geodesic_distance, compute_apl_scalars, Z_CRITICAL
from .operators import GrassmannAPLOperators, APLOperator, OperatorResult, N0Law
from .bridge import APLGrassmannBridge, APLPhase, APLState, phase_from_z, PHASE_PROPERTIES


# =============================================================================
# NARRATIVE STRUCTURE
# =============================================================================

class NarrativePhase(Enum):
    """The four phases of narrative structure."""
    SETUP = 0       # Operators 0-7: Establish ground
    RISING = 1      # Operators 8-15: Build tension
    CLIMAX = 2      # Operators 16-23: Peak intensity
    RESOLUTION = 3  # Operators 24-31: Resolve and integrate


class NarrativeRole(Enum):
    """Narrative roles for story elements."""
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    MENTOR = "mentor"
    THRESHOLD_GUARDIAN = "threshold_guardian"
    HERALD = "herald"
    SHAPESHIFTER = "shapeshifter"
    SHADOW = "shadow"
    TRICKSTER = "trickster"
    ALLY = "ally"


@dataclass
class StoryBeat:
    """
    A single narrative moment/event.

    Corresponds to one operator application in the sequence.
    """
    index: int                          # Position in 32-operator sequence
    operator: APLOperator               # Which operator was applied
    result: OperatorResult              # Result of application
    narrative_phase: NarrativePhase     # Which narrative phase
    apl_phase: APLPhase                 # APL consciousness phase
    z_level: float                      # Z-coordinate at this beat
    description: str = ""               # Human-readable description
    is_transition: bool = False         # Did this beat cause a transition?

    @property
    def intensity(self) -> float:
        """Narrative intensity (0-1) based on position and Z."""
        # Peaks at climax (operators 16-23)
        position_intensity = 1.0 - abs(self.index - 19.5) / 16.0
        return (position_intensity + self.z_level) / 2.0


@dataclass
class NarrativeState:
    """
    Complete state of narrative progression.

    Tracks:
    - Current position in 32-operator sequence
    - Geometric state (Grassmann point)
    - APL state (scalars and phase)
    - History of story beats
    """
    point: GrassmannPoint
    apl_state: APLState
    beats: List[StoryBeat] = field(default_factory=list)
    current_index: int = 0

    @property
    def narrative_phase(self) -> NarrativePhase:
        """Current narrative phase based on position."""
        return NarrativePhase(self.current_index // 8)

    @property
    def phase_progress(self) -> float:
        """Progress within current narrative phase (0-1)."""
        return (self.current_index % 8) / 7.0

    @property
    def overall_progress(self) -> float:
        """Overall narrative progress (0-1)."""
        return self.current_index / 31.0

    @property
    def is_complete(self) -> bool:
        """Whether narrative sequence is complete."""
        return self.current_index >= 32

    @property
    def transitions(self) -> List[StoryBeat]:
        """Get all beats that caused phase transitions."""
        return [b for b in self.beats if b.is_transition]

    @property
    def climax_z(self) -> float:
        """Maximum Z achieved during climax phase."""
        climax_beats = [b for b in self.beats if b.narrative_phase == NarrativePhase.CLIMAX]
        if not climax_beats:
            return 0.0
        return max(b.z_level for b in climax_beats)

    def get_beat(self, index: int) -> Optional[StoryBeat]:
        """Get beat at specific index."""
        for beat in self.beats:
            if beat.index == index:
                return beat
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'point': self.point.to_dict(),
            'apl_state': self.apl_state.to_dict(),
            'beats': [
                {
                    'index': b.index,
                    'operator': b.operator.value,
                    'z_level': b.z_level,
                    'narrative_phase': b.narrative_phase.value,
                    'apl_phase': b.apl_phase.value,
                    'description': b.description,
                    'is_transition': b.is_transition,
                }
                for b in self.beats
            ],
            'current_index': self.current_index,
        }


# =============================================================================
# NARRATIVE TEMPLATES
# =============================================================================

# Standard 32-operator narrative template
STANDARD_NARRATIVE_TEMPLATE = [
    # SETUP (0-7): Establish ground
    (APLOperator.BOUNDARY, {'description': 'Opening: establish the ordinary world'}),
    (APLOperator.FUSION, {'weight': 0.3, 'description': 'Introduction: blend elements'}),
    (APLOperator.BOUNDARY, {'description': 'Define: clarify the stakes'}),
    (APLOperator.GROUP, {'description': 'Gather: collect key elements'}),
    (APLOperator.FUSION, {'weight': 0.5, 'description': 'Connect: weave relationships'}),
    (APLOperator.AMPLIFY, {'magnitude': 0.2, 'description': 'Hint: foreshadow conflict'}),
    (APLOperator.BOUNDARY, {'description': 'Threshold: approach the point of no return'}),
    (APLOperator.FUSION, {'weight': 0.6, 'description': 'Commit: cross into act two'}),

    # RISING (8-15): Build tension
    (APLOperator.AMPLIFY, {'magnitude': 0.3, 'description': 'Challenge: first test'}),
    (APLOperator.GROUP, {'description': 'Ally: gain support'}),
    (APLOperator.AMPLIFY, {'magnitude': 0.4, 'description': 'Escalate: raise stakes'}),
    (APLOperator.SEPARATE, {'description': 'Conflict: encounter opposition'}),
    (APLOperator.AMPLIFY, {'magnitude': 0.5, 'description': 'Crisis: approach the abyss'}),
    (APLOperator.DECOHERE, {'entropy': 0.2, 'description': 'Doubt: face uncertainty'}),
    (APLOperator.BOUNDARY, {'description': 'Ordeal: the dark moment'}),
    (APLOperator.AMPLIFY, {'magnitude': 0.6, 'description': 'Transform: seize the sword'}),

    # CLIMAX (16-23): Peak intensity
    (APLOperator.AMPLIFY, {'magnitude': 0.7, 'description': 'Confrontation: face the shadow'}),
    (APLOperator.SEPARATE, {'description': 'Battle: direct conflict'}),
    (APLOperator.AMPLIFY, {'magnitude': 0.8, 'description': 'Apex: maximum intensity'}),
    (APLOperator.GROUP, {'description': 'Unite: bring forces together'}),
    (APLOperator.AMPLIFY, {'magnitude': 0.6, 'description': 'Victory/Defeat: outcome determined'}),
    (APLOperator.FUSION, {'weight': 0.7, 'description': 'Integration: absorb the lesson'}),
    (APLOperator.BOUNDARY, {'description': 'Return threshold: begin journey back'}),
    (APLOperator.DECOHERE, {'entropy': 0.1, 'description': 'Release: let go of the old'}),

    # RESOLUTION (24-31): Resolve and integrate
    (APLOperator.FUSION, {'weight': 0.6, 'description': 'Reconcile: mend what was broken'}),
    (APLOperator.GROUP, {'description': 'Reunite: gather the transformed'}),
    (APLOperator.BOUNDARY, {'description': 'New normal: establish changed world'}),
    (APLOperator.FUSION, {'weight': 0.5, 'description': 'Reflect: understand the journey'}),
    (APLOperator.DECOHERE, {'entropy': 0.05, 'description': 'Open: leave space for mystery'}),
    (APLOperator.BOUNDARY, {'description': 'Resolve: tie remaining threads'}),
    (APLOperator.FUSION, {'weight': 0.4, 'description': 'Echo: callback to beginning'}),
    (APLOperator.BOUNDARY, {'description': 'Close: the new equilibrium'}),
]


# Alternative templates for different story types
TRAGEDY_TEMPLATE = [
    # Setup
    (APLOperator.BOUNDARY, {}), (APLOperator.AMPLIFY, {'magnitude': 0.4}),
    (APLOperator.GROUP, {}), (APLOperator.AMPLIFY, {'magnitude': 0.5}),
    (APLOperator.FUSION, {'weight': 0.6}), (APLOperator.AMPLIFY, {'magnitude': 0.6}),
    (APLOperator.BOUNDARY, {}), (APLOperator.AMPLIFY, {'magnitude': 0.7}),
    # Rising
    (APLOperator.AMPLIFY, {'magnitude': 0.8}), (APLOperator.SEPARATE, {}),
    (APLOperator.AMPLIFY, {'magnitude': 0.9}), (APLOperator.DECOHERE, {'entropy': 0.3}),
    (APLOperator.AMPLIFY, {'magnitude': 0.7}), (APLOperator.SEPARATE, {}),
    (APLOperator.DECOHERE, {'entropy': 0.4}), (APLOperator.AMPLIFY, {'magnitude': 0.5}),
    # Climax - extended fall
    (APLOperator.DECOHERE, {'entropy': 0.5}), (APLOperator.SEPARATE, {}),
    (APLOperator.DECOHERE, {'entropy': 0.6}), (APLOperator.BOUNDARY, {}),
    (APLOperator.DECOHERE, {'entropy': 0.4}), (APLOperator.SEPARATE, {}),
    (APLOperator.DECOHERE, {'entropy': 0.3}), (APLOperator.BOUNDARY, {}),
    # Resolution - somber
    (APLOperator.BOUNDARY, {}), (APLOperator.FUSION, {'weight': 0.3}),
    (APLOperator.BOUNDARY, {}), (APLOperator.DECOHERE, {'entropy': 0.2}),
    (APLOperator.BOUNDARY, {}), (APLOperator.FUSION, {'weight': 0.2}),
    (APLOperator.BOUNDARY, {}), (APLOperator.BOUNDARY, {}),
]


COMEDY_TEMPLATE = [
    # Setup - light establishment
    (APLOperator.BOUNDARY, {}), (APLOperator.FUSION, {'weight': 0.5}),
    (APLOperator.GROUP, {}), (APLOperator.FUSION, {'weight': 0.6}),
    (APLOperator.AMPLIFY, {'magnitude': 0.3}), (APLOperator.GROUP, {}),
    (APLOperator.FUSION, {'weight': 0.5}), (APLOperator.AMPLIFY, {'magnitude': 0.4}),
    # Rising - complications
    (APLOperator.SEPARATE, {}), (APLOperator.FUSION, {'weight': 0.4}),
    (APLOperator.DECOHERE, {'entropy': 0.2}), (APLOperator.GROUP, {}),
    (APLOperator.SEPARATE, {}), (APLOperator.FUSION, {'weight': 0.5}),
    (APLOperator.AMPLIFY, {'magnitude': 0.5}), (APLOperator.DECOHERE, {'entropy': 0.3}),
    # Climax - chaos and resolution
    (APLOperator.GROUP, {}), (APLOperator.AMPLIFY, {'magnitude': 0.6}),
    (APLOperator.FUSION, {'weight': 0.7}), (APLOperator.GROUP, {}),
    (APLOperator.FUSION, {'weight': 0.8}), (APLOperator.AMPLIFY, {'magnitude': 0.4}),
    (APLOperator.FUSION, {'weight': 0.6}), (APLOperator.GROUP, {}),
    # Resolution - harmonious ending
    (APLOperator.FUSION, {'weight': 0.7}), (APLOperator.GROUP, {}),
    (APLOperator.FUSION, {'weight': 0.8}), (APLOperator.BOUNDARY, {}),
    (APLOperator.FUSION, {'weight': 0.6}), (APLOperator.GROUP, {}),
    (APLOperator.FUSION, {'weight': 0.5}), (APLOperator.BOUNDARY, {}),
]


# =============================================================================
# GRASSMANN NARRATIVE ENGINE
# =============================================================================

class GrassmannNarrativeEngine:
    """
    Executes narrative sequences on the Grassmann manifold.

    Maps story structure to geometric evolution:
    - Each beat = one operator application
    - Phase transitions = major story turns
    - Z-coordinate = narrative intensity/coherence
    """

    def __init__(self, n: int = 9, k: int = 3, enforce_n0: bool = True):
        """
        Initialize narrative engine.

        Args:
            n: Ambient dimension
            k: Subspace dimension
            enforce_n0: Whether to enforce N0 causality laws
        """
        self.n = n
        self.k = k
        self.manifold = GrassmannManifold(n, k)
        self.operators = GrassmannAPLOperators(n, k, enforce_n0)
        self.bridge = APLGrassmannBridge(n, k)

        # Current state
        self.state: Optional[NarrativeState] = None

    def initialize(self, start_point: Optional[GrassmannPoint] = None) -> NarrativeState:
        """
        Initialize a new narrative.

        Args:
            start_point: Optional starting point (default: canonical)

        Returns:
            Initial narrative state
        """
        if start_point is None:
            start_point = self.manifold.canonical_point()

        apl_state = self.bridge.grassmann_to_apl(start_point)

        self.state = NarrativeState(
            point=start_point,
            apl_state=apl_state,
            beats=[],
            current_index=0,
        )

        self.operators.reset()
        self.operators.set_reference(start_point)

        return self.state

    def execute_beat(self, operator: APLOperator,
                     params: Dict = None,
                     description: str = "") -> StoryBeat:
        """
        Execute a single narrative beat.

        Args:
            operator: APL operator to apply
            params: Operator parameters
            description: Human-readable description

        Returns:
            The resulting story beat
        """
        if self.state is None:
            raise ValueError("Narrative not initialized. Call initialize() first.")

        if self.state.is_complete:
            raise ValueError("Narrative already complete.")

        params = params or {}

        # Get Z before
        z_before = self.state.apl_state.z
        phase_before = self.state.apl_state.phase

        # Apply operator
        if operator == APLOperator.BOUNDARY:
            result = self.operators.boundary(self.state.point)
        elif operator == APLOperator.AMPLIFY:
            result = self.operators.amplify(
                self.state.point,
                magnitude=params.get('magnitude', 0.5)
            )
        elif operator == APLOperator.DECOHERE:
            result = self.operators.decohere(
                self.state.point,
                entropy=params.get('entropy', 0.3)
            )
        elif operator == APLOperator.FUSION:
            # For fusion, use a reference point based on narrative phase
            target_phase = APLPhase(min(8, self.state.current_index // 4))
            other = self.bridge.phase_landmarks[target_phase]
            result = self.operators.fusion(
                self.state.point, other,
                weight=params.get('weight', 0.5)
            )
        elif operator == APLOperator.GROUP:
            # Group with random point
            other = self.manifold.random_point()
            result = self.operators.group(self.state.point, other)
        elif operator == APLOperator.SEPARATE:
            # Separate from previous phase landmark
            prev_phase = APLPhase(max(0, self.state.apl_state.phase.value - 1))
            other = self.bridge.phase_landmarks[prev_phase]
            result = self.operators.separate(self.state.point, other)
        else:
            raise ValueError(f"Unknown operator: {operator}")

        # Update state if successful
        if result.success and result.output:
            self.state.point = result.output
            self.state.apl_state = self.bridge.grassmann_to_apl(result.output)

        # Check for phase transition
        is_transition = self.state.apl_state.phase != phase_before

        # Create beat
        beat = StoryBeat(
            index=self.state.current_index,
            operator=operator,
            result=result,
            narrative_phase=self.state.narrative_phase,
            apl_phase=self.state.apl_state.phase,
            z_level=self.state.apl_state.z,
            description=description or params.get('description', ''),
            is_transition=is_transition,
        )

        self.state.beats.append(beat)
        self.state.current_index += 1

        return beat

    def execute_sequence(self, template: List[Tuple[APLOperator, Dict]] = None,
                         callback: Callable[[StoryBeat], None] = None) -> NarrativeState:
        """
        Execute a complete 32-beat narrative sequence.

        Args:
            template: Operator sequence (default: standard template)
            callback: Optional callback for each beat

        Returns:
            Final narrative state
        """
        if self.state is None:
            self.initialize()

        if template is None:
            template = STANDARD_NARRATIVE_TEMPLATE

        for i, (op, params) in enumerate(template):
            if self.state.current_index >= 32:
                break

            beat = self.execute_beat(op, params)

            if callback:
                callback(beat)

        return self.state

    def get_narrative_arc(self) -> Dict[str, Any]:
        """
        Analyze the narrative arc.

        Returns summary statistics and key moments.
        """
        if self.state is None or not self.state.beats:
            return {'error': 'No narrative executed'}

        z_values = [b.z_level for b in self.state.beats]

        # Find key moments
        max_z_idx = np.argmax(z_values)
        transitions = [(b.index, b.apl_phase.name) for b in self.state.transitions]

        # Phase distribution
        phase_counts = {}
        for beat in self.state.beats:
            name = beat.apl_phase.name
            phase_counts[name] = phase_counts.get(name, 0) + 1

        return {
            'total_beats': len(self.state.beats),
            'transitions': len(self.state.transitions),
            'transition_points': transitions,
            'peak_z': max(z_values),
            'peak_beat': int(max_z_idx),
            'min_z': min(z_values),
            'mean_z': np.mean(z_values),
            'climax_achieved': self.state.climax_z >= Z_CRITICAL,
            'final_phase': self.state.apl_state.phase.name,
            'phase_distribution': phase_counts,
            'z_trajectory': z_values,
        }

    def visualize_arc(self, width: int = 60) -> str:
        """
        Create ASCII visualization of narrative arc.

        Returns multi-line string showing Z trajectory.
        """
        if self.state is None or not self.state.beats:
            return "No narrative to visualize"

        z_values = [b.z_level for b in self.state.beats]
        height = 10

        # Normalize to height
        min_z, max_z = min(z_values), max(z_values)
        if max_z == min_z:
            normalized = [height // 2] * len(z_values)
        else:
            normalized = [
                int((z - min_z) / (max_z - min_z) * (height - 1))
                for z in z_values
            ]

        # Build grid
        lines = []
        for row in range(height - 1, -1, -1):
            line = ""
            for col, n in enumerate(normalized):
                if n == row:
                    # Mark transitions
                    beat = self.state.beats[col]
                    if beat.is_transition:
                        line += "*"
                    else:
                        line += "#"
                elif row == int((Z_CRITICAL - min_z) / (max_z - min_z) * (height - 1)):
                    line += "-"  # Critical line
                else:
                    line += " "
            lines.append(f"{row:2d}|{line}")

        # Add phase labels
        phase_line = "  |"
        for i, beat in enumerate(self.state.beats):
            if i % 8 == 0:
                phase_line += beat.narrative_phase.name[0]
            else:
                phase_line += " "

        lines.append("  +" + "-" * len(self.state.beats))
        lines.append(phase_line)

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the narrative engine."""
        self.state = None
        self.operators.reset()


# =============================================================================
# GAME INTEGRATION
# =============================================================================

class NarrativeGameIntegration:
    """
    Integrates narrative engine with game mechanics.

    Maps card plays to narrative beats, tracking story evolution
    alongside game state.
    """

    def __init__(self, engine: GrassmannNarrativeEngine):
        """
        Initialize integration.

        Args:
            engine: The narrative engine to use
        """
        self.engine = engine
        self.turn_to_beat_map: Dict[int, int] = {}  # game turn -> narrative beat
        self.beat_bonuses: Dict[int, int] = {}  # beat index -> score bonus

    def on_turn_start(self, turn: int) -> Optional[StoryBeat]:
        """
        Process start of game turn.

        Maps turn to narrative beat and executes if needed.
        """
        if self.engine.state is None:
            self.engine.initialize()

        # Map turn to beat (e.g., 2 beats per turn for 16-turn game)
        beat_idx = turn * 2

        if beat_idx < 32 and self.engine.state.current_index <= beat_idx:
            # Execute beats to catch up
            while self.engine.state.current_index <= beat_idx:
                template_idx = self.engine.state.current_index
                if template_idx < len(STANDARD_NARRATIVE_TEMPLATE):
                    op, params = STANDARD_NARRATIVE_TEMPLATE[template_idx]
                    beat = self.engine.execute_beat(op, params)
                    self.turn_to_beat_map[turn] = beat.index

                    # Calculate bonus based on beat properties
                    bonus = self._calculate_beat_bonus(beat)
                    self.beat_bonuses[beat.index] = bonus
                else:
                    break

            return self.engine.state.beats[-1] if self.engine.state.beats else None

        return None

    def _calculate_beat_bonus(self, beat: StoryBeat) -> int:
        """Calculate score bonus for a narrative beat."""
        bonus = 0

        # Transition bonus
        if beat.is_transition:
            bonus += 10

        # Climax intensity bonus
        if beat.narrative_phase == NarrativePhase.CLIMAX:
            bonus += int(beat.z_level * 15)

        # Critical threshold bonus
        if beat.z_level >= Z_CRITICAL:
            bonus += 5

        return bonus

    def get_narrative_bonus(self, turn: int) -> int:
        """Get total narrative bonus for a turn."""
        beat_idx = self.turn_to_beat_map.get(turn)
        if beat_idx is not None:
            return self.beat_bonuses.get(beat_idx, 0)
        return 0

    def get_narrative_summary(self) -> Dict[str, Any]:
        """Get summary for UI display."""
        if self.engine.state is None:
            return {'active': False}

        return {
            'active': True,
            'beat': self.engine.state.current_index,
            'narrative_phase': self.engine.state.narrative_phase.name,
            'apl_phase': self.engine.state.apl_state.phase.name,
            'z_level': round(self.engine.state.apl_state.z, 3),
            'is_critical': self.engine.state.apl_state.z >= Z_CRITICAL,
            'progress': round(self.engine.state.overall_progress * 100, 1),
            'transitions': len(self.engine.state.transitions),
        }
