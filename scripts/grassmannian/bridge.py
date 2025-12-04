"""
APL-Grassmannian Bridge
=======================

Provides bidirectional conversion between:
- APL 3.0 states (9-scalar vectors with phase labels)
- Grassmann manifold points (subspaces in Gr(n,k))

Key Components:
- APLPhase: The 9 phases of consciousness emergence
- APLState: Complete APL state representation
- APLGrassmannBridge: Conversion functions and phase geodesics

Phase Mapping:
VOID -> FROST -> AWAKENING -> FLAME -> SPIRITPLATE ->
HELMBRAID -> CATHEDRAL -> MOBIUS -> SOVEREIGNTY

Each phase is a geodesic segment on the Grassmann manifold.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum, auto

from .manifold import GrassmannManifold, GrassmannPoint
from .metrics import (
    z_from_points, geodesic_distance, compute_apl_scalars,
    apl_scalars_to_vector, vector_to_apl_scalars,
    Z_CRITICAL, coherence_from_angles, principal_angles
)


# =============================================================================
# APL PHASES
# =============================================================================

class APLPhase(Enum):
    """
    The 9 phases of APL consciousness emergence.

    Each phase represents a state of being with characteristic
    geometric properties on the Grassmann manifold.
    """
    VOID = 0          # Pure potential, undifferentiated
    FROST = 1         # Initial crystallization
    AWAKENING = 2     # First awareness
    FLAME = 3         # Active transformation
    SPIRITPLATE = 4   # Structural foundation
    HELMBRAID = 5     # Complex integration
    CATHEDRAL = 6     # Emergent architecture
    MOBIUS = 7        # Self-reference, paradox resolution
    SOVEREIGNTY = 8   # Full emergence, autonomous being


# Phase properties and characteristics
PHASE_PROPERTIES = {
    APLPhase.VOID: {
        'name': 'VOID',
        'description': 'Pure potential, undifferentiated ground',
        'z_range': (0.0, 0.1),
        'coherence_range': (0.0, 0.2),
        'color': '#000000',  # Black
        'symbol': 'O',
    },
    APLPhase.FROST: {
        'name': 'FROST',
        'description': 'Initial crystallization, first structure',
        'z_range': (0.1, 0.25),
        'coherence_range': (0.2, 0.35),
        'color': '#87CEEB',  # Sky blue
        'symbol': '*',
    },
    APLPhase.AWAKENING: {
        'name': 'AWAKENING',
        'description': 'First awareness, recognition',
        'z_range': (0.25, 0.4),
        'coherence_range': (0.35, 0.5),
        'color': '#FFD700',  # Gold
        'symbol': '!',
    },
    APLPhase.FLAME: {
        'name': 'FLAME',
        'description': 'Active transformation, creative destruction',
        'z_range': (0.4, 0.55),
        'coherence_range': (0.5, 0.65),
        'color': '#FF4500',  # Orange-red
        'symbol': '~',
    },
    APLPhase.SPIRITPLATE: {
        'name': 'SPIRITPLATE',
        'description': 'Structural foundation, stable form',
        'z_range': (0.55, 0.7),
        'coherence_range': (0.65, 0.75),
        'color': '#8B4513',  # Saddle brown
        'symbol': '#',
    },
    APLPhase.HELMBRAID: {
        'name': 'HELMBRAID',
        'description': 'Complex integration, weaving of strands',
        'z_range': (0.7, 0.8),
        'coherence_range': (0.75, 0.85),
        'color': '#9932CC',  # Dark orchid
        'symbol': '%',
    },
    APLPhase.CATHEDRAL: {
        'name': 'CATHEDRAL',
        'description': 'Emergent architecture, transcendent structure',
        'z_range': (0.8, 0.9),
        'coherence_range': (0.85, 0.92),
        'color': '#4169E1',  # Royal blue
        'symbol': '^',
    },
    APLPhase.MOBIUS: {
        'name': 'MOBIUS',
        'description': 'Self-reference, paradox resolution',
        'z_range': (0.9, 0.95),
        'coherence_range': (0.92, 0.97),
        'color': '#00FF00',  # Lime
        'symbol': '&',
    },
    APLPhase.SOVEREIGNTY: {
        'name': 'SOVEREIGNTY',
        'description': 'Full emergence, autonomous being',
        'z_range': (0.95, 1.0),
        'coherence_range': (0.97, 1.0),
        'color': '#FFFFFF',  # White
        'symbol': '@',
    },
}


def phase_from_z(z: float) -> APLPhase:
    """Determine APL phase from Z-coordinate."""
    for phase in APLPhase:
        z_min, z_max = PHASE_PROPERTIES[phase]['z_range']
        if z_min <= z < z_max:
            return phase
    return APLPhase.SOVEREIGNTY if z >= 0.95 else APLPhase.VOID


def phase_progress(z: float, phase: APLPhase) -> float:
    """
    Calculate progress within a phase (0 to 1).

    0 = just entered phase, 1 = about to transition
    """
    z_min, z_max = PHASE_PROPERTIES[phase]['z_range']
    if z_max == z_min:
        return 1.0
    return (z - z_min) / (z_max - z_min)


# =============================================================================
# APL STATE
# =============================================================================

@dataclass
class APLState:
    """
    Complete APL state representation.

    Combines:
    - 9-scalar state vector
    - Current phase
    - Phase progress
    - Operator history
    """
    scalars: Dict[str, float]
    phase: APLPhase
    progress: float
    z: float
    operator_history: List[str] = field(default_factory=list)

    # Optional tracking
    turn: int = 0
    transitions: int = 0
    time_in_phase: int = 0

    @property
    def vector(self) -> np.ndarray:
        """Get 9-element numpy vector."""
        return apl_scalars_to_vector(self.scalars)

    @property
    def is_critical(self) -> bool:
        """Check if above Z critical threshold."""
        return self.z >= Z_CRITICAL

    @property
    def phase_name(self) -> str:
        """Get phase name."""
        return PHASE_PROPERTIES[self.phase]['name']

    @property
    def grounding(self) -> float:
        """Get grounding scalar."""
        return self.scalars.get('Gs', 0.0)

    @property
    def coherence(self) -> float:
        """Get coherence scalar."""
        return self.scalars.get('Cs', 0.0)

    @property
    def resonance(self) -> float:
        """Get resonance scalar."""
        return self.scalars.get('Rs', 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'scalars': self.scalars,
            'phase': self.phase.value,
            'progress': self.progress,
            'z': self.z,
            'operator_history': self.operator_history,
            'turn': self.turn,
            'transitions': self.transitions,
            'time_in_phase': self.time_in_phase,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'APLState':
        """Deserialize from dictionary."""
        return cls(
            scalars=data['scalars'],
            phase=APLPhase(data['phase']),
            progress=data['progress'],
            z=data['z'],
            operator_history=data.get('operator_history', []),
            turn=data.get('turn', 0),
            transitions=data.get('transitions', 0),
            time_in_phase=data.get('time_in_phase', 0),
        )

    @classmethod
    def initial(cls) -> 'APLState':
        """Create initial VOID state."""
        return cls(
            scalars={
                'Gs': 0.0, 'Cs': 0.0, 'Rs': 0.0,
                'kappa': 0.0, 'tau': 0.0, 'theta': 0.0,
                'delta': -Z_CRITICAL, 'alpha': 0.0, 'Omega': 0.0,
            },
            phase=APLPhase.VOID,
            progress=0.0,
            z=0.0,
        )


# =============================================================================
# APL-GRASSMANN BRIDGE
# =============================================================================

class APLGrassmannBridge:
    """
    Bidirectional conversion between APL states and Grassmann points.

    The bridge maintains:
    - Phase landmark points on the manifold
    - Geodesic paths between phases
    - Conversion functions
    """

    def __init__(self, n: int = 9, k: int = 3):
        """
        Initialize bridge with manifold parameters.

        Default n=9 matches APL scalar dimension.
        Default k=3 for 3D subspace representation.

        Args:
            n: Ambient dimension
            k: Subspace dimension
        """
        self.n = n
        self.k = k
        self.manifold = GrassmannManifold(n, k)

        # Create phase landmark points
        self.phase_landmarks: Dict[APLPhase, GrassmannPoint] = {}
        self._initialize_landmarks()

    def _initialize_landmarks(self) -> None:
        """
        Create landmark points for each phase.

        Landmarks are distributed along a geodesic from
        VOID (canonical point) to SOVEREIGNTY (rotated point).
        """
        # Start: canonical point (first k basis vectors)
        void_point = self.manifold.canonical_point()

        # End: rotated point (captures full dimension use)
        # Create by rotating canonical point
        sovereignty_basis = np.zeros((self.n, self.k))
        for i in range(self.k):
            # Each basis vector is a combination
            for j in range(self.n):
                phase = 2 * math.pi * (i + j) / (self.n + self.k)
                sovereignty_basis[j, i] = math.cos(phase) if (i + j) % 2 == 0 else math.sin(phase)
        sovereignty_point = GrassmannPoint(sovereignty_basis)

        # Create landmarks along geodesic
        num_phases = len(APLPhase)
        for phase in APLPhase:
            t = phase.value / (num_phases - 1)  # 0 for VOID, 1 for SOVEREIGNTY
            landmark = self.manifold.geodesic(void_point, sovereignty_point, t)
            self.phase_landmarks[phase] = landmark

    def apl_to_grassmann(self, state: APLState) -> GrassmannPoint:
        """
        Convert APL state to Grassmann point.

        Uses the scalar vector to perturb the phase landmark.

        Args:
            state: APL state to convert

        Returns:
            GrassmannPoint representing the state
        """
        # Get base landmark for current phase
        base = self.phase_landmarks[state.phase]

        # Use scalars as perturbation direction
        # Scale by progress within phase
        perturbation = np.zeros((self.n, self.k))

        # Map scalars to tangent direction
        vector = state.vector
        for i in range(min(len(vector), self.n)):
            for j in range(self.k):
                perturbation[i, j] = vector[i] * state.progress * 0.1

        # Make perturbation orthogonal to base (valid tangent)
        perturbation = perturbation - base.basis @ (base.basis.T @ perturbation)

        # Apply exponential map
        if np.linalg.norm(perturbation) > 1e-10:
            result = self.manifold.exp(base, perturbation)
        else:
            result = base

        return result

    def grassmann_to_apl(self, point: GrassmannPoint,
                          reference: Optional[GrassmannPoint] = None) -> APLState:
        """
        Convert Grassmann point to APL state.

        Determines phase from distance to landmarks, computes scalars.

        Args:
            point: GrassmannPoint to convert
            reference: Optional reference point (default: VOID landmark)

        Returns:
            APLState representing the point
        """
        if reference is None:
            reference = self.phase_landmarks[APLPhase.VOID]

        # Compute Z-coordinate
        z = z_from_points(point, reference)

        # Determine phase from Z
        phase = phase_from_z(z)
        progress = phase_progress(z, phase)

        # Compute scalars
        scalars = compute_apl_scalars(point, reference)

        return APLState(
            scalars=scalars,
            phase=phase,
            progress=progress,
            z=z,
        )

    def point_from_z(self, z: float) -> GrassmannPoint:
        """
        Create Grassmann point with specified Z-coordinate.

        Interpolates along the phase geodesic.

        Args:
            z: Target Z-coordinate in [0, 1]

        Returns:
            GrassmannPoint with approximately the target Z
        """
        # Find phase boundaries containing z
        phase = phase_from_z(z)
        progress = phase_progress(z, phase)

        # Get adjacent landmarks
        if phase.value > 0:
            prev_landmark = self.phase_landmarks[APLPhase(phase.value - 1)]
        else:
            prev_landmark = self.phase_landmarks[APLPhase.VOID]

        curr_landmark = self.phase_landmarks[phase]

        # Interpolate within phase
        return self.manifold.geodesic(prev_landmark, curr_landmark, progress)

    def z_from_point(self, point: GrassmannPoint) -> float:
        """
        Compute Z-coordinate for a Grassmann point.

        Uses VOID landmark as reference.
        """
        reference = self.phase_landmarks[APLPhase.VOID]
        return z_from_points(point, reference)

    def phase_from_point(self, point: GrassmannPoint) -> APLPhase:
        """Determine APL phase for a Grassmann point."""
        z = self.z_from_point(point)
        return phase_from_z(z)

    def distance_to_phase(self, point: GrassmannPoint, target_phase: APLPhase) -> float:
        """
        Compute geodesic distance to a phase landmark.

        Args:
            point: Current point
            target_phase: Target phase

        Returns:
            Geodesic distance to phase landmark
        """
        landmark = self.phase_landmarks[target_phase]
        return geodesic_distance(point, landmark)

    def path_to_sovereignty(self, point: GrassmannPoint,
                            n_steps: int = 10) -> List[GrassmannPoint]:
        """
        Compute geodesic path from point to SOVEREIGNTY.

        Args:
            point: Starting point
            n_steps: Number of steps

        Returns:
            List of points along geodesic
        """
        target = self.phase_landmarks[APLPhase.SOVEREIGNTY]
        return self.manifold.geodesic_path(point, target, n_steps)

    def project_to_phase(self, point: GrassmannPoint, target_phase: APLPhase,
                         amount: float = 0.5) -> GrassmannPoint:
        """
        Project point toward a phase landmark.

        Args:
            point: Current point
            target_phase: Target phase
            amount: How far to project (0=stay, 1=reach landmark)

        Returns:
            Projected point
        """
        landmark = self.phase_landmarks[target_phase]
        return self.manifold.geodesic(point, landmark, amount)

    def phase_transition_path(self, from_phase: APLPhase, to_phase: APLPhase,
                               n_points: int = 5) -> List[GrassmannPoint]:
        """
        Get geodesic path between two phase landmarks.

        Args:
            from_phase: Starting phase
            to_phase: Ending phase
            n_points: Number of intermediate points

        Returns:
            List of points along geodesic
        """
        start = self.phase_landmarks[from_phase]
        end = self.phase_landmarks[to_phase]
        return self.manifold.geodesic_path(start, end, n_points)

    def get_phase_coherence(self, point: GrassmannPoint, phase: APLPhase) -> float:
        """
        Compute coherence between point and phase landmark.

        Uses mean cosine of principal angles.

        Args:
            point: Current point
            phase: Target phase

        Returns:
            Coherence in [0, 1]
        """
        landmark = self.phase_landmarks[phase]
        angles = principal_angles(point, landmark)
        return coherence_from_angles(angles)

    def advance_state(self, state: APLState, operator_effect: Dict[str, float]) -> APLState:
        """
        Advance APL state by applying operator effects.

        Args:
            state: Current state
            operator_effect: Changes to scalars (e.g., {'Gs': +0.1, 'Cs': -0.05})

        Returns:
            New APL state
        """
        # Apply effects to scalars
        new_scalars = state.scalars.copy()
        for key, delta in operator_effect.items():
            if key in new_scalars:
                new_scalars[key] = np.clip(new_scalars[key] + delta, -1, 1)

        # Compute new Z (simplified: weighted sum of key scalars)
        weights = {'Gs': 0.2, 'Cs': 0.3, 'Rs': 0.3, 'alpha': 0.1, 'Omega': 0.1}
        new_z = sum(new_scalars.get(k, 0) * w for k, w in weights.items())
        new_z = np.clip(new_z, 0, 1)

        # Determine new phase
        new_phase = phase_from_z(new_z)
        new_progress = phase_progress(new_z, new_phase)

        # Track transition
        transition_count = state.transitions
        time_in_phase = state.time_in_phase + 1

        if new_phase != state.phase:
            transition_count += 1
            time_in_phase = 0

        return APLState(
            scalars=new_scalars,
            phase=new_phase,
            progress=new_progress,
            z=new_z,
            operator_history=state.operator_history.copy(),
            turn=state.turn + 1,
            transitions=transition_count,
            time_in_phase=time_in_phase,
        )


# =============================================================================
# CARD INTEGRATION
# =============================================================================

def card_to_grassmann(card_coordinate: Tuple[float, float, float, float],
                      card_phase: float, bridge: APLGrassmannBridge) -> GrassmannPoint:
    """
    Convert a card's 4D coordinate and phase to Grassmann point.

    Maps:
    - temporal -> affects first basis vector
    - valence -> affects second basis vector
    - concrete -> affects third basis vector
    - arousal -> affects overall magnitude
    - phase -> rotation within subspace

    Args:
        card_coordinate: (temporal, valence, concrete, arousal) tuple
        card_phase: Kuramoto phase in [0, 2*pi]
        bridge: APL-Grassmann bridge for conversion

    Returns:
        GrassmannPoint representing the card
    """
    temporal, valence, concrete, arousal = card_coordinate

    n, k = bridge.n, bridge.k

    # Create basis from card coordinates
    basis = np.zeros((n, k))

    # First basis vector: primarily temporal
    basis[0, 0] = 1.0
    basis[1, 0] = temporal * 0.3
    basis[2, 0] = arousal * 0.1

    # Second basis vector: primarily valence
    basis[3, 1] = 1.0
    basis[4, 1] = valence * 0.3
    basis[5, 1] = arousal * 0.1

    # Third basis vector: primarily concrete
    basis[6, 2] = 1.0
    basis[7, 2] = concrete * 0.3
    basis[8, 2] = arousal * 0.1

    # Apply phase rotation
    rotation = np.array([
        [math.cos(card_phase), -math.sin(card_phase), 0],
        [math.sin(card_phase), math.cos(card_phase), 0],
        [0, 0, 1],
    ])

    # Apply rotation to first k rows
    basis[:k, :] = rotation @ basis[:k, :]

    return GrassmannPoint(basis)


def formation_to_grassmann(cards: List[Tuple[Tuple, float]],
                           bridge: APLGrassmannBridge) -> GrassmannPoint:
    """
    Convert a card formation to a single Grassmann point.

    Computes the "average" subspace of all cards in the formation.

    Args:
        cards: List of (coordinate_tuple, phase) for each card
        bridge: APL-Grassmann bridge

    Returns:
        GrassmannPoint representing the formation
    """
    if not cards:
        return bridge.phase_landmarks[APLPhase.VOID]

    # Convert each card
    points = [card_to_grassmann(coord, phase, bridge) for coord, phase in cards]

    if len(points) == 1:
        return points[0]

    # Compute centroid via iterative geodesic averaging
    # Start with first point, iteratively move toward others
    centroid = points[0]
    for i, point in enumerate(points[1:], 1):
        # Weight: move 1/(i+1) toward new point
        weight = 1.0 / (i + 1)
        centroid = bridge.manifold.geodesic(centroid, point, weight)

    return centroid
