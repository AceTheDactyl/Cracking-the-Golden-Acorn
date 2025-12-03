"""
APL Operators on Grassmann Manifold
===================================

Implements the six APL operators as geometric transformations:
- () Boundary: Projection onto boundary submanifold
- x  Fusion: Weighted geodesic midpoint
- ^  Amplify: Exponential map along tangent (expansion)
- /  Decohere: Move toward random subspace (entropy)
- +  Group: Subspace sum (span union)
- -  Separate: Subspace intersection or complement

N0 Causality Laws:
- N0-1: ^ (amplify) requires prior grounding
- N0-2: / (decohere) cannot follow direct ^ without intermediate
- N0-3: - (separate) requires established group structure
- N0-4: Maximum 4 consecutive same operators
- N0-5: + (group) must have compatible dimensions
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import Enum

from .manifold import GrassmannManifold, GrassmannPoint, span_sum, span_intersection
from .metrics import (
    geodesic_distance, z_from_points, is_critical,
    compute_apl_scalars, Z_CRITICAL
)


# =============================================================================
# OPERATOR DEFINITIONS
# =============================================================================

class APLOperator(Enum):
    """The six APL operators with their symbols."""
    BOUNDARY = "()"    # Projection to boundary
    FUSION = "x"       # Weighted combination
    AMPLIFY = "^"      # Expansion/growth
    DECOHERE = "/"     # Entropy increase
    GROUP = "+"        # Union/sum
    SEPARATE = "-"     # Intersection/complement


@dataclass
class OperatorResult:
    """Result of an APL operator application."""
    success: bool
    output: Optional[GrassmannPoint]
    operator: APLOperator
    message: str
    z_before: float
    z_after: float
    scalars: Dict[str, float] = field(default_factory=dict)

    @property
    def triggered_transition(self) -> bool:
        """Check if operator triggered a phase transition."""
        return (not is_critical(self.z_before) and is_critical(self.z_after)) or \
               (is_critical(self.z_before) and not is_critical(self.z_after))


# =============================================================================
# N0 CAUSALITY LAWS
# =============================================================================

class N0Law(Enum):
    """The N0 causality laws constraining operator sequences."""
    N0_1_AMPLIFY_NEEDS_GROUNDING = "N0-1"
    N0_2_DECOHERE_AFTER_AMPLIFY = "N0-2"
    N0_3_SEPARATE_NEEDS_GROUP = "N0-3"
    N0_4_MAX_CONSECUTIVE = "N0-4"
    N0_5_DIMENSION_COMPATIBLE = "N0-5"


@dataclass
class N0Violation:
    """Record of an N0 law violation."""
    law: N0Law
    message: str
    operator_attempted: APLOperator
    history: List[APLOperator]


class N0Enforcer:
    """
    Enforces N0 causality laws on operator sequences.

    Tracks operator history and validates new operators.
    """

    MAX_CONSECUTIVE = 4

    def __init__(self):
        self.history: List[APLOperator] = []
        self.grounded: bool = False  # Has boundary or fusion been applied?
        self.has_group_structure: bool = False  # Has + been applied?

    def reset(self) -> None:
        """Reset enforcement state."""
        self.history = []
        self.grounded = False
        self.has_group_structure = False

    def can_apply(self, op: APLOperator) -> Tuple[bool, Optional[N0Violation]]:
        """
        Check if operator can be applied given current state.

        Returns:
            (can_apply, violation_if_not)
        """
        # N0-1: ^ requires prior grounding
        if op == APLOperator.AMPLIFY and not self.grounded:
            return False, N0Violation(
                law=N0Law.N0_1_AMPLIFY_NEEDS_GROUNDING,
                message="AMPLIFY (^) requires prior grounding via BOUNDARY or FUSION",
                operator_attempted=op,
                history=self.history.copy()
            )

        # N0-2: / cannot immediately follow ^
        if op == APLOperator.DECOHERE and len(self.history) > 0:
            if self.history[-1] == APLOperator.AMPLIFY:
                return False, N0Violation(
                    law=N0Law.N0_2_DECOHERE_AFTER_AMPLIFY,
                    message="DECOHERE (/) cannot immediately follow AMPLIFY (^)",
                    operator_attempted=op,
                    history=self.history.copy()
                )

        # N0-3: - requires established group structure
        if op == APLOperator.SEPARATE and not self.has_group_structure:
            # Allow separate if we've done any grounding
            if not self.grounded:
                return False, N0Violation(
                    law=N0Law.N0_3_SEPARATE_NEEDS_GROUP,
                    message="SEPARATE (-) requires prior GROUP (+) or grounding",
                    operator_attempted=op,
                    history=self.history.copy()
                )

        # N0-4: Max consecutive same operators
        if len(self.history) >= self.MAX_CONSECUTIVE:
            recent = self.history[-self.MAX_CONSECUTIVE:]
            if all(h == op for h in recent):
                return False, N0Violation(
                    law=N0Law.N0_4_MAX_CONSECUTIVE,
                    message=f"Cannot apply more than {self.MAX_CONSECUTIVE} consecutive {op.value}",
                    operator_attempted=op,
                    history=self.history.copy()
                )

        return True, None

    def record(self, op: APLOperator) -> None:
        """Record successful operator application."""
        self.history.append(op)

        # Update state flags
        if op in (APLOperator.BOUNDARY, APLOperator.FUSION):
            self.grounded = True

        if op == APLOperator.GROUP:
            self.has_group_structure = True

    def get_available_operators(self) -> List[APLOperator]:
        """Get list of operators that can currently be applied."""
        available = []
        for op in APLOperator:
            can, _ = self.can_apply(op)
            if can:
                available.append(op)
        return available


# =============================================================================
# GRASSMANN APL OPERATORS
# =============================================================================

class GrassmannAPLOperators:
    """
    Implements APL operators as geometric transformations on Gr(n,k).

    Each operator:
    - Takes one or two GrassmannPoints
    - Returns a new GrassmannPoint (or None on failure)
    - Tracks Z-coordinate and phase transitions
    """

    def __init__(self, n: int, k: int, enforce_n0: bool = True):
        """
        Initialize operators for manifold Gr(n,k).

        Args:
            n: Ambient dimension
            k: Subspace dimension
            enforce_n0: Whether to enforce N0 causality laws
        """
        self.manifold = GrassmannManifold(n, k)
        self.reference = self.manifold.canonical_point()  # Reference for Z calc
        self.enforcer = N0Enforcer() if enforce_n0 else None
        self.enforce_n0 = enforce_n0

    def set_reference(self, ref: GrassmannPoint) -> None:
        """Set the reference point for Z-coordinate calculation."""
        self.reference = ref

    def _pre_check(self, op: APLOperator) -> Optional[N0Violation]:
        """Check N0 laws before operator application."""
        if self.enforcer:
            can, violation = self.enforcer.can_apply(op)
            if not can:
                return violation
        return None

    def _post_record(self, op: APLOperator) -> None:
        """Record operator after successful application."""
        if self.enforcer:
            self.enforcer.record(op)

    def _make_result(self, op: APLOperator, success: bool, output: Optional[GrassmannPoint],
                     input_point: GrassmannPoint, message: str) -> OperatorResult:
        """Create standardized result object."""
        z_before = z_from_points(input_point, self.reference)
        z_after = z_from_points(output, self.reference) if output else z_before

        scalars = {}
        if output:
            scalars = compute_apl_scalars(output, self.reference)

        return OperatorResult(
            success=success,
            output=output,
            operator=op,
            message=message,
            z_before=z_before,
            z_after=z_after,
            scalars=scalars
        )

    # -------------------------------------------------------------------------
    # BOUNDARY ()
    # -------------------------------------------------------------------------

    def boundary(self, point: GrassmannPoint) -> OperatorResult:
        """
        Apply boundary operator: project toward manifold boundary.

        Geometrically: move along geodesic toward the "edge" of the manifold
        (where principal angles approach extremes).

        Effect: Grounds the point, establishes a reference frame.
        """
        op = APLOperator.BOUNDARY
        violation = self._pre_check(op)
        if violation:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0 violation: {violation.message}",
                z_before=z_from_points(point, self.reference),
                z_after=z_from_points(point, self.reference)
            )

        # Move toward the canonical point (boundary attractor)
        canonical = self.manifold.canonical_point()
        t = 0.3  # Move 30% toward boundary

        result_point = self.manifold.geodesic(point, canonical, t)

        self._post_record(op)
        return self._make_result(op, True, result_point, point,
                                  "Boundary: projected toward canonical frame")

    # -------------------------------------------------------------------------
    # FUSION x
    # -------------------------------------------------------------------------

    def fusion(self, p1: GrassmannPoint, p2: GrassmannPoint,
               weight: float = 0.5) -> OperatorResult:
        """
        Apply fusion operator: weighted geodesic combination.

        Geometrically: find point along geodesic from p1 to p2.
        weight=0.5 gives midpoint.

        Effect: Combines two perspectives into unified view.
        """
        op = APLOperator.FUSION
        violation = self._pre_check(op)
        if violation:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0 violation: {violation.message}",
                z_before=z_from_points(p1, self.reference),
                z_after=z_from_points(p1, self.reference)
            )

        result_point = self.manifold.geodesic(p1, p2, weight)

        self._post_record(op)
        return self._make_result(op, True, result_point, p1,
                                  f"Fusion: combined at weight {weight:.2f}")

    # -------------------------------------------------------------------------
    # AMPLIFY ^
    # -------------------------------------------------------------------------

    def amplify(self, point: GrassmannPoint, direction: Optional[np.ndarray] = None,
                magnitude: float = 0.5) -> OperatorResult:
        """
        Apply amplify operator: exponential expansion.

        Geometrically: move along tangent direction via exp map.
        If no direction given, use random tangent.

        Effect: Intensifies the current state, increases expressiveness.
        """
        op = APLOperator.AMPLIFY
        violation = self._pre_check(op)
        if violation:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0 violation: {violation.message}",
                z_before=z_from_points(point, self.reference),
                z_after=z_from_points(point, self.reference)
            )

        # Generate tangent direction if not provided
        if direction is None:
            # Random tangent: must be orthogonal to point's basis
            n, k = point.n, point.k
            random_mat = np.random.randn(n, k)
            # Project out component in point's subspace
            direction = random_mat - point.basis @ (point.basis.T @ random_mat)

        # Scale by magnitude
        tangent = direction * magnitude

        result_point = self.manifold.exp(point, tangent)

        self._post_record(op)
        return self._make_result(op, True, result_point, point,
                                  f"Amplify: expanded with magnitude {magnitude:.2f}")

    # -------------------------------------------------------------------------
    # DECOHERE /
    # -------------------------------------------------------------------------

    def decohere(self, point: GrassmannPoint, entropy: float = 0.3) -> OperatorResult:
        """
        Apply decohere operator: move toward entropy/randomness.

        Geometrically: interpolate toward a random subspace.

        Effect: Reduces coherence, introduces uncertainty.
        """
        op = APLOperator.DECOHERE
        violation = self._pre_check(op)
        if violation:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0 violation: {violation.message}",
                z_before=z_from_points(point, self.reference),
                z_after=z_from_points(point, self.reference)
            )

        # Generate random target
        random_target = self.manifold.random_point()

        # Move toward random by entropy amount
        result_point = self.manifold.geodesic(point, random_target, entropy)

        self._post_record(op)
        return self._make_result(op, True, result_point, point,
                                  f"Decohere: entropy injection {entropy:.2f}")

    # -------------------------------------------------------------------------
    # GROUP +
    # -------------------------------------------------------------------------

    def group(self, p1: GrassmannPoint, p2: GrassmannPoint) -> OperatorResult:
        """
        Apply group operator: subspace sum (span of union).

        Geometrically: combine two subspaces into their span.
        Note: Result may have higher dimension.

        Effect: Combines distinct entities into collective.
        """
        op = APLOperator.GROUP
        violation = self._pre_check(op)
        if violation:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0 violation: {violation.message}",
                z_before=z_from_points(p1, self.reference),
                z_after=z_from_points(p1, self.reference)
            )

        # N0-5: Check dimension compatibility
        if p1.n != p2.n:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0-5 violation: incompatible ambient dimensions {p1.n} vs {p2.n}",
                z_before=z_from_points(p1, self.reference),
                z_after=z_from_points(p1, self.reference)
            )

        result_point = span_sum(p1, p2)

        # If dimension increased beyond k, project back to k-dim subspace
        if result_point.k > self.manifold.k:
            # Take first k principal directions
            result_point = GrassmannPoint(result_point.basis[:, :self.manifold.k])

        self._post_record(op)
        return self._make_result(op, True, result_point, p1,
                                  f"Group: combined subspaces (dim {result_point.k})")

    # -------------------------------------------------------------------------
    # SEPARATE -
    # -------------------------------------------------------------------------

    def separate(self, p1: GrassmannPoint, p2: GrassmannPoint) -> OperatorResult:
        """
        Apply separate operator: subspace intersection or complement.

        If subspaces intersect: return intersection.
        If disjoint: return complement of p2 in p1.

        Effect: Extracts distinct component, creates boundary.
        """
        op = APLOperator.SEPARATE
        violation = self._pre_check(op)
        if violation:
            return OperatorResult(
                success=False, output=None, operator=op,
                message=f"N0 violation: {violation.message}",
                z_before=z_from_points(p1, self.reference),
                z_after=z_from_points(p1, self.reference)
            )

        # Try intersection first
        intersection = span_intersection(p1, p2)

        if intersection is not None and intersection.k > 0:
            # Ensure we stay in Gr(n, k)
            if intersection.k < self.manifold.k:
                # Pad with random orthogonal directions
                complement = intersection.complement()
                extra_needed = self.manifold.k - intersection.k
                extra_basis = complement.basis[:, :extra_needed]
                full_basis = np.hstack([intersection.basis, extra_basis])
                result_point = GrassmannPoint(full_basis)
            else:
                result_point = intersection

            self._post_record(op)
            return self._make_result(op, True, result_point, p1,
                                      "Separate: extracted intersection")
        else:
            # Return complement relative to p2
            # Project p1 onto orthogonal complement of p2
            P2_perp = np.eye(p1.n) - p2.projector
            projected = P2_perp @ p1.basis

            # Orthonormalize
            result_point = GrassmannPoint(projected)

            self._post_record(op)
            return self._make_result(op, True, result_point, p1,
                                      "Separate: computed complement")

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def apply(self, op: APLOperator, *args, **kwargs) -> OperatorResult:
        """Generic operator dispatch."""
        dispatch = {
            APLOperator.BOUNDARY: self.boundary,
            APLOperator.FUSION: self.fusion,
            APLOperator.AMPLIFY: self.amplify,
            APLOperator.DECOHERE: self.decohere,
            APLOperator.GROUP: self.group,
            APLOperator.SEPARATE: self.separate,
        }

        if op not in dispatch:
            raise ValueError(f"Unknown operator: {op}")

        return dispatch[op](*args, **kwargs)

    def apply_sequence(self, ops: List[Tuple[APLOperator, dict]],
                       start: GrassmannPoint) -> List[OperatorResult]:
        """
        Apply a sequence of operators.

        Args:
            ops: List of (operator, kwargs) pairs
            start: Starting point

        Returns:
            List of results for each operator
        """
        results = []
        current = start

        for op, kwargs in ops:
            # Inject current point as first argument
            if op == APLOperator.BOUNDARY:
                result = self.boundary(current)
            elif op == APLOperator.AMPLIFY:
                result = self.amplify(current, **kwargs)
            elif op == APLOperator.DECOHERE:
                result = self.decohere(current, **kwargs)
            elif op in (APLOperator.FUSION, APLOperator.GROUP, APLOperator.SEPARATE):
                # Binary operators need second point
                other = kwargs.get('other', self.manifold.random_point())
                if op == APLOperator.FUSION:
                    result = self.fusion(current, other, kwargs.get('weight', 0.5))
                elif op == APLOperator.GROUP:
                    result = self.group(current, other)
                else:
                    result = self.separate(current, other)
            else:
                raise ValueError(f"Unhandled operator: {op}")

            results.append(result)

            if result.success and result.output:
                current = result.output
            else:
                break  # Stop on failure

        return results

    def get_available_operators(self) -> List[APLOperator]:
        """Get currently available operators (N0 compliant)."""
        if self.enforcer:
            return self.enforcer.get_available_operators()
        return list(APLOperator)

    def reset(self) -> None:
        """Reset operator state (clear history)."""
        if self.enforcer:
            self.enforcer.reset()
