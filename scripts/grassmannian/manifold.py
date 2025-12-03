"""
Grassmann Manifold Implementation
=================================

Implements Gr(n,k) - the Grassmannian manifold of k-dimensional
subspaces of R^n. Used as the geometric substrate for APL semantics.

Key Classes:
- GrassmannPoint: A point on Gr(n,k), represented by orthonormal basis
- GrassmannManifold: The manifold itself with operations

Mathematical Background:
- Gr(n,k) is a compact smooth manifold of dimension k(n-k)
- Each point is a k-dimensional subspace of R^n
- Represented by n x k matrices with orthonormal columns
- Equivalence class: Y ~ YQ for any k x k orthogonal Q
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class ManifoldError(Exception):
    """Errors related to manifold operations."""
    pass


@dataclass
class GrassmannPoint:
    """
    A point on the Grassmann manifold Gr(n,k).

    Represented by an n x k matrix Y with orthonormal columns,
    where Y^T Y = I_k (identity matrix).

    Attributes:
        basis: n x k orthonormal matrix representing the subspace
        n: Ambient dimension
        k: Subspace dimension
    """
    basis: np.ndarray

    def __post_init__(self):
        """Validate and orthonormalize the basis."""
        if self.basis.ndim != 2:
            raise ManifoldError("Basis must be 2D matrix")

        # Orthonormalize via QR decomposition
        q, r = np.linalg.qr(self.basis)
        self.basis = q[:, :self.k]

        # Ensure consistent sign (canonical form)
        signs = np.sign(np.diag(r)[:self.k])
        signs[signs == 0] = 1
        self.basis = self.basis * signs

    @property
    def n(self) -> int:
        """Ambient dimension."""
        return self.basis.shape[0]

    @property
    def k(self) -> int:
        """Subspace dimension."""
        return self.basis.shape[1]

    @property
    def projector(self) -> np.ndarray:
        """
        Projection matrix P = Y Y^T.
        Projects vectors onto this subspace.
        """
        return self.basis @ self.basis.T

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project a vector onto this subspace."""
        return self.projector @ v

    def complement(self) -> 'GrassmannPoint':
        """
        Return the orthogonal complement subspace.
        If this is Gr(n,k), complement is Gr(n, n-k).
        """
        # Use null space of basis.T
        # complement basis has columns orthogonal to all columns of basis
        _, _, vh = np.linalg.svd(self.basis.T, full_matrices=True)
        complement_basis = vh[self.k:].T
        return GrassmannPoint(complement_basis)

    def contains(self, v: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if vector v lies in this subspace."""
        proj = self.project(v)
        return np.linalg.norm(v - proj) < tol

    def angle_with(self, other: 'GrassmannPoint') -> np.ndarray:
        """
        Compute principal angles with another subspace.
        Returns array of min(k1, k2) angles in [0, pi/2].
        """
        from .metrics import principal_angles
        return principal_angles(self, other)

    def distance_to(self, other: 'GrassmannPoint', metric: str = 'geodesic') -> float:
        """
        Compute distance to another point using specified metric.

        Metrics: 'geodesic', 'chordal', 'projection'
        """
        from .metrics import geodesic_distance, chordal_distance, projection_frobenius

        if metric == 'geodesic':
            return geodesic_distance(self, other)
        elif metric == 'chordal':
            return chordal_distance(self, other)
        elif metric == 'projection':
            return projection_frobenius(self, other)
        else:
            raise ManifoldError(f"Unknown metric: {metric}")

    def __eq__(self, other: 'GrassmannPoint') -> bool:
        """Two points are equal if they span the same subspace."""
        if self.n != other.n or self.k != other.k:
            return False
        # Check if projectors are the same
        return np.allclose(self.projector, other.projector)

    def __repr__(self) -> str:
        return f"GrassmannPoint(Gr({self.n},{self.k}))"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'basis': self.basis.tolist(),
            'n': self.n,
            'k': self.k,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GrassmannPoint':
        """Deserialize from dictionary."""
        return cls(basis=np.array(data['basis']))

    @classmethod
    def random(cls, n: int, k: int, seed: Optional[int] = None) -> 'GrassmannPoint':
        """Generate a random point on Gr(n,k)."""
        rng = np.random.default_rng(seed)
        basis = rng.standard_normal((n, k))
        return cls(basis)

    @classmethod
    def from_vectors(cls, vectors: List[np.ndarray]) -> 'GrassmannPoint':
        """Create point from spanning vectors."""
        basis = np.column_stack(vectors)
        return cls(basis)

    @classmethod
    def canonical(cls, n: int, k: int) -> 'GrassmannPoint':
        """Return the canonical point: span of first k standard basis vectors."""
        basis = np.eye(n, k)
        return cls(basis)


class GrassmannManifold:
    """
    The Grassmann manifold Gr(n,k).

    Provides manifold operations:
    - Exponential map: tangent space -> manifold
    - Logarithm map: manifold -> tangent space
    - Geodesics: shortest paths between points
    - Parallel transport: move tangent vectors along geodesics

    The manifold has dimension k(n-k).
    """

    def __init__(self, n: int, k: int):
        """
        Initialize Gr(n,k).

        Args:
            n: Ambient dimension (must be > 0)
            k: Subspace dimension (must be 0 < k < n)
        """
        if n <= 0:
            raise ManifoldError(f"n must be positive, got {n}")
        if k <= 0 or k >= n:
            raise ManifoldError(f"k must be in (0, n), got k={k}, n={n}")

        self.n = n
        self.k = k

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.k * (self.n - self.k)

    def tangent_space_dim(self) -> int:
        """Dimension of tangent space at any point."""
        return self.dimension

    def exp(self, base: GrassmannPoint, tangent: np.ndarray) -> GrassmannPoint:
        """
        Exponential map: move from base point along tangent vector.

        The tangent vector is represented as an n x k matrix Delta
        where base.T @ Delta = 0 (horizontal condition).

        Args:
            base: Starting point on manifold
            tangent: Tangent vector (n x k matrix)

        Returns:
            New point on manifold
        """
        self._validate_point(base)

        # Compact SVD of tangent vector
        u, s, vh = np.linalg.svd(tangent, full_matrices=False)

        # Compute geodesic endpoint
        # Y(t) = Y cos(Sigma*t) V^T + U sin(Sigma*t) V^T at t=1
        cos_s = np.diag(np.cos(s))
        sin_s = np.diag(np.sin(s))

        new_basis = base.basis @ vh.T @ cos_s @ vh + u @ sin_s @ vh

        return GrassmannPoint(new_basis)

    def log(self, base: GrassmannPoint, target: GrassmannPoint) -> np.ndarray:
        """
        Logarithm map: tangent vector from base pointing to target.

        Returns the initial velocity of the geodesic from base to target.

        Args:
            base: Starting point
            target: End point

        Returns:
            Tangent vector (n x k matrix) at base
        """
        self._validate_point(base)
        self._validate_point(target)

        # Compute M = (I - Y Y^T) Z = component of Z orthogonal to Y
        Y = base.basis
        Z = target.basis

        # Project Z onto orthogonal complement of Y
        M = Z - Y @ (Y.T @ Z)

        # Compact SVD
        u, s, vh = np.linalg.svd(M, full_matrices=False)

        # Compute angles (principal angles)
        # s contains sin(theta_i), need arcsin
        angles = np.arcsin(np.clip(s, -1, 1))

        # Initial velocity = U * diag(angles) * V^T
        tangent = u @ np.diag(angles) @ vh

        return tangent

    def geodesic(self, start: GrassmannPoint, end: GrassmannPoint,
                 t: float) -> GrassmannPoint:
        """
        Point along geodesic from start to end at parameter t.

        t=0 gives start, t=1 gives end.

        Args:
            start: Starting point
            end: Ending point
            t: Parameter in [0, 1]

        Returns:
            Point at parameter t along geodesic
        """
        tangent = self.log(start, end)
        scaled_tangent = t * tangent
        return self.exp(start, scaled_tangent)

    def geodesic_path(self, start: GrassmannPoint, end: GrassmannPoint,
                      n_points: int = 10) -> List[GrassmannPoint]:
        """
        Compute discrete path along geodesic.

        Args:
            start: Starting point
            end: Ending point
            n_points: Number of points including endpoints

        Returns:
            List of points along geodesic
        """
        ts = np.linspace(0, 1, n_points)
        return [self.geodesic(start, end, t) for t in ts]

    def parallel_transport(self, base: GrassmannPoint, target: GrassmannPoint,
                           v: np.ndarray) -> np.ndarray:
        """
        Parallel transport tangent vector v from base to target.

        Args:
            base: Starting point
            target: Ending point
            v: Tangent vector at base (n x k matrix)

        Returns:
            Transported tangent vector at target
        """
        # Get geodesic tangent
        delta = self.log(base, target)

        # Compact SVD of delta
        u, s, vh = np.linalg.svd(delta, full_matrices=False)

        # Transport formula (simplified)
        # V_parallel = V - U sin(S) cos(S) U^T V + U sin(S)^2 Y^T V
        Y = base.basis
        sin_s = np.diag(np.sin(s))
        cos_s = np.diag(np.cos(s))

        term1 = v
        term2 = u @ sin_s @ cos_s @ (u.T @ v)
        term3 = u @ (sin_s @ sin_s) @ (Y.T @ v)

        return term1 - term2 + term3

    def random_point(self, seed: Optional[int] = None) -> GrassmannPoint:
        """Generate a random point on this manifold."""
        return GrassmannPoint.random(self.n, self.k, seed)

    def canonical_point(self) -> GrassmannPoint:
        """Return the canonical base point."""
        return GrassmannPoint.canonical(self.n, self.k)

    def distance(self, p1: GrassmannPoint, p2: GrassmannPoint,
                 metric: str = 'geodesic') -> float:
        """Compute distance between two points."""
        self._validate_point(p1)
        self._validate_point(p2)
        return p1.distance_to(p2, metric)

    def _validate_point(self, point: GrassmannPoint) -> None:
        """Check that point belongs to this manifold."""
        if point.n != self.n or point.k != self.k:
            raise ManifoldError(
                f"Point Gr({point.n},{point.k}) doesn't match manifold Gr({self.n},{self.k})"
            )

    def __repr__(self) -> str:
        return f"GrassmannManifold(Gr({self.n},{self.k}), dim={self.dimension})"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def span_intersection(p1: GrassmannPoint, p2: GrassmannPoint) -> Optional[GrassmannPoint]:
    """
    Compute intersection of two subspaces.

    Returns None if intersection is trivial (just zero vector).
    """
    # Intersection = null space of [P1_perp ; P2_perp]
    # Or equivalently, range of P1 @ P2 (for same-dim case)

    if p1.n != p2.n:
        raise ManifoldError("Points must be in same ambient space")

    # Stack complements
    P1 = p1.projector
    P2 = p2.projector

    # Find vectors in both: solve P1 v = v and P2 v = v
    # Equivalent to finding eigenspace of P1 @ P2 with eigenvalue 1

    product = P1 @ P2
    eigenvalues, eigenvectors = np.linalg.eig(product)

    # Find eigenvectors with eigenvalue close to 1
    tol = 1e-10
    mask = np.abs(eigenvalues - 1) < tol

    if not np.any(mask):
        return None

    intersection_basis = eigenvectors[:, mask].real

    if intersection_basis.shape[1] == 0:
        return None

    return GrassmannPoint(intersection_basis)


def span_sum(p1: GrassmannPoint, p2: GrassmannPoint) -> GrassmannPoint:
    """
    Compute span of union of two subspaces (their sum).

    Returns point representing span(V1 union V2).
    """
    if p1.n != p2.n:
        raise ManifoldError("Points must be in same ambient space")

    # Concatenate bases and orthonormalize
    combined = np.hstack([p1.basis, p2.basis])

    return GrassmannPoint(combined)


def subspace_angle(p1: GrassmannPoint, p2: GrassmannPoint) -> float:
    """
    Compute the single principal angle for equal-dimension subspaces.

    This is the geodesic distance scaled by 1/sqrt(k).
    """
    from .metrics import geodesic_distance
    return geodesic_distance(p1, p2) / math.sqrt(min(p1.k, p2.k))
