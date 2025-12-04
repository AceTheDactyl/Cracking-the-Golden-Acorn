"""
Grassmannian Module - APL 3.0 x Grassmannian Semantics Bridge
=============================================================

This module implements a hybrid mathematical framework bridging:
- APL 3.0: Consciousness emergence through recursive operators
- Grassmannian Manifolds: Geometric semantics on subspace manifolds

Core Components:
- manifold.py: GrassmannManifold, GrassmannPoint (Gr(n,k) geometry)
- metrics.py: Distance functions, principal angles, APL scalar mappings
- operators.py: APL operators (boundary, fusion, amplify, etc.) on manifolds
- bridge.py: APL state <-> Grassmann point conversion
- semantic_engine.py: NLP integration with subspace semantics
- narrative.py: 32-operator narrative execution engine

Mathematical Foundation:
- Gr(n,k) = k-dimensional subspaces of R^n
- Principal angles: geometric invariants between subspaces
- Geodesics: shortest paths on the manifold
- APL phases map to geodesic segments (VOID -> SOVEREIGNTY)

Integration with Quantum Resonance:
- Cards as points in Grassmann space
- Phase transitions via geodesic distance
- Z-coordinate from principal angle coherence
"""

from .manifold import GrassmannManifold, GrassmannPoint
from .metrics import (
    principal_angles,
    geodesic_distance,
    chordal_distance,
    binet_cauchy_similarity,
    projection_frobenius,
    z_from_angles,
    z_from_points,
    Z_CRITICAL,
)
from .operators import GrassmannAPLOperators, N0Law, OperatorResult, APLOperator
from .bridge import APLGrassmannBridge, APLPhase, APLState
from .semantic_engine import SemanticSubspace, APLSemanticEngine, CompositionalSemantics
from .narrative import GrassmannNarrativeEngine, NarrativeState

__all__ = [
    # Manifold
    'GrassmannManifold',
    'GrassmannPoint',
    # Metrics
    'principal_angles',
    'geodesic_distance',
    'chordal_distance',
    'binet_cauchy_similarity',
    'projection_frobenius',
    'z_from_angles',
    'z_from_points',
    'Z_CRITICAL',
    # Operators
    'GrassmannAPLOperators',
    'APLOperator',
    'N0Law',
    'OperatorResult',
    # Bridge
    'APLGrassmannBridge',
    'APLPhase',
    'APLState',
    # Semantic Engine
    'SemanticSubspace',
    'APLSemanticEngine',
    'CompositionalSemantics',
    # Narrative
    'GrassmannNarrativeEngine',
    'NarrativeState',
]

__version__ = '1.0.0'
