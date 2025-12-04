"""
Grassmannian Distance Metrics
=============================

Implements various distance functions on the Grassmann manifold Gr(n,k),
plus APL scalar mappings derived from geometric invariants.

Key Metrics:
- Principal angles: fundamental invariants between subspaces
- Geodesic distance: Riemannian distance (sum of squared angles)
- Chordal distance: Embedding distance (Frobenius of projector difference)
- Binet-Cauchy: Determinant-based similarity
- Projection Frobenius: sqrt(k - ||Y1^T Y2||_F^2)

APL Scalar Mappings:
- Z-coordinate from angle coherence (critical threshold = 0.858)
- 9-dimensional APL state vector from geometric properties
"""

import math
import numpy as np
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .manifold import GrassmannPoint

# =============================================================================
# CONSTANTS
# =============================================================================

# Critical Z threshold for phase transitions
Z_CRITICAL = 0.858

# APL scalar indices
APL_SCALAR_NAMES = [
    'Gs',   # 0: Grounding scalar
    'Cs',   # 1: Coherence scalar
    'Rs',   # 2: Resonance scalar
    'kappa', # 3: Curvature
    'tau',   # 4: Torsion
    'theta', # 5: Phase angle
    'delta', # 6: Divergence
    'alpha', # 7: Amplitude
    'Omega', # 8: Frequency
]


# =============================================================================
# PRINCIPAL ANGLES
# =============================================================================

def principal_angles(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> np.ndarray:
    """
    Compute principal angles between two subspaces.

    The principal angles theta_1, ..., theta_r (where r = min(k1, k2))
    are defined as the angles between the closest vectors in each subspace,
    then the next closest in the orthogonal complement, etc.

    They satisfy: cos(theta_i) = sigma_i(Y1^T Y2)

    Args:
        p1: First Grassmann point
        p2: Second Grassmann point

    Returns:
        Array of principal angles in [0, pi/2], sorted ascending
    """
    # Compute Y1^T Y2
    inner_product = p1.basis.T @ p2.basis

    # SVD gives cos(theta_i) as singular values
    singular_values = np.linalg.svd(inner_product, compute_uv=False)

    # Clip to [0, 1] for numerical stability
    singular_values = np.clip(singular_values, 0, 1)

    # Convert to angles
    angles = np.arccos(singular_values)

    return np.sort(angles)


def principal_vectors(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute principal angles and principal vectors.

    Principal vectors are the unit vectors achieving each principal angle.

    Returns:
        (angles, V1, V2) where V1 and V2 are matrices of principal vectors
    """
    inner_product = p1.basis.T @ p2.basis

    # Full SVD
    u, s, vh = np.linalg.svd(inner_product)

    # Principal vectors in original space
    V1 = p1.basis @ u
    V2 = p2.basis @ vh.T

    # Angles from singular values
    angles = np.arccos(np.clip(s, 0, 1))

    return angles, V1, V2


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def geodesic_distance(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute geodesic (Riemannian) distance on Gr(n,k).

    d_geo = sqrt(sum(theta_i^2))

    This is the length of the shortest path on the manifold.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Geodesic distance >= 0
    """
    angles = principal_angles(p1, p2)
    return math.sqrt(np.sum(angles ** 2))


def chordal_distance(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute chordal (Frobenius) distance.

    d_chordal = (1/sqrt(2)) * ||P1 - P2||_F

    where P1, P2 are the projection matrices.

    This is the embedding distance in the space of projection matrices.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Chordal distance >= 0
    """
    diff = p1.projector - p2.projector
    return np.linalg.norm(diff, 'fro') / math.sqrt(2)


def projection_frobenius(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute projection Frobenius distance.

    d_proj = sqrt(k - ||Y1^T Y2||_F^2)

    Related to sum of squared sines of principal angles.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Projection distance >= 0
    """
    inner = p1.basis.T @ p2.basis
    fro_sq = np.sum(inner ** 2)
    k = min(p1.k, p2.k)

    return math.sqrt(max(0, k - fro_sq))


def binet_cauchy_similarity(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute Binet-Cauchy similarity.

    BC = prod(cos^2(theta_i))

    This is related to the determinant of the Gram matrix.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Similarity in [0, 1], where 1 = identical subspaces
    """
    angles = principal_angles(p1, p2)
    cos_sq = np.cos(angles) ** 2
    return np.prod(cos_sq)


def binet_cauchy_distance(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute Binet-Cauchy distance.

    d_BC = sqrt(1 - BC_similarity)

    Args:
        p1: First point
        p2: Second point

    Returns:
        Distance in [0, 1]
    """
    return math.sqrt(1 - binet_cauchy_similarity(p1, p2))


def asimov_distance(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute Asimov distance (max principal angle).

    d_asimov = max(theta_i)

    This is the spectral norm of the angle matrix.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Maximum principal angle in [0, pi/2]
    """
    angles = principal_angles(p1, p2)
    return np.max(angles) if len(angles) > 0 else 0.0


# =============================================================================
# APL SCALAR MAPPINGS
# =============================================================================

def z_from_angles(angles: np.ndarray) -> float:
    """
    Compute Z-coordinate from principal angles.

    Z = 1 - (2/pi) * mean(angles)

    Z in [0, 1]:
    - Z = 1 when subspaces are identical (angles = 0)
    - Z = 0 when subspaces are orthogonal (angles = pi/2)
    - Z > Z_CRITICAL triggers phase transition

    Args:
        angles: Array of principal angles

    Returns:
        Z-coordinate in [0, 1]
    """
    if len(angles) == 0:
        return 1.0  # Trivial case

    mean_angle = np.mean(angles)
    z = 1.0 - (2.0 / math.pi) * mean_angle

    return np.clip(z, 0.0, 1.0)


def z_from_points(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> float:
    """
    Compute Z-coordinate between two Grassmann points.

    Convenience wrapper around z_from_angles.
    """
    angles = principal_angles(p1, p2)
    return z_from_angles(angles)


def is_critical(z: float, threshold: float = Z_CRITICAL) -> bool:
    """Check if Z-coordinate is above critical threshold."""
    return z >= threshold


def coherence_from_angles(angles: np.ndarray) -> float:
    """
    Compute coherence metric from principal angles.

    Cs = mean(cos(angles))

    High coherence = small angles = aligned subspaces.

    Args:
        angles: Array of principal angles

    Returns:
        Coherence in [0, 1]
    """
    if len(angles) == 0:
        return 1.0
    return np.mean(np.cos(angles))


def resonance_from_angles(angles: np.ndarray) -> float:
    """
    Compute resonance metric from principal angles.

    Rs = 1 - var(angles) / (pi/2)^2

    High resonance = consistent angles = stable relationship.

    Args:
        angles: Array of principal angles

    Returns:
        Resonance in [0, 1]
    """
    if len(angles) < 2:
        return 1.0  # Perfect resonance for single angle

    variance = np.var(angles)
    max_var = (math.pi / 2) ** 2 / 4  # Variance of uniform on [0, pi/2]

    return max(0, 1 - variance / max_var)


def compute_apl_scalars(p1: 'GrassmannPoint', p2: 'GrassmannPoint',
                         velocity: Optional[np.ndarray] = None) -> dict:
    """
    Compute the full 9-dimensional APL scalar state.

    The scalars encode the geometric relationship between two subspaces
    and optional dynamics (velocity in tangent space).

    Args:
        p1: First Grassmann point
        p2: Second Grassmann point (reference)
        velocity: Optional tangent vector at p1

    Returns:
        Dictionary with APL scalar values
    """
    angles = principal_angles(p1, p2)

    # Basic scalars from angles
    z = z_from_angles(angles)
    coherence = coherence_from_angles(angles)
    resonance = resonance_from_angles(angles)

    # Grounding: inverse of geodesic distance (closer = more grounded)
    geo_dist = geodesic_distance(p1, p2)
    grounding = 1.0 / (1.0 + geo_dist)

    # Curvature: related to spread of angles
    kappa = np.std(angles) if len(angles) > 1 else 0.0

    # Dynamic scalars (require velocity)
    if velocity is not None:
        # Torsion: how much velocity twists
        tau = np.linalg.norm(velocity) / (1 + np.linalg.norm(velocity))

        # Phase angle: direction of velocity
        flat_v = velocity.flatten()
        theta = math.atan2(flat_v[1] if len(flat_v) > 1 else 0,
                          flat_v[0] if len(flat_v) > 0 else 1)

        # Amplitude: magnitude of velocity
        alpha = np.linalg.norm(velocity)

        # Frequency: rate of phase change (simplified)
        omega = alpha / (1 + alpha)
    else:
        tau = 0.0
        theta = 0.0
        alpha = 0.0
        omega = 0.0

    # Divergence: how far from critical threshold
    delta = z - Z_CRITICAL

    return {
        'Gs': grounding,
        'Cs': coherence,
        'Rs': resonance,
        'kappa': kappa,
        'tau': tau,
        'theta': theta,
        'delta': delta,
        'alpha': alpha,
        'Omega': omega,
        'z': z,  # Include Z as well
    }


def apl_scalars_to_vector(scalars: dict) -> np.ndarray:
    """Convert APL scalars dict to 9-element numpy vector."""
    return np.array([
        scalars.get('Gs', 0.0),
        scalars.get('Cs', 0.0),
        scalars.get('Rs', 0.0),
        scalars.get('kappa', 0.0),
        scalars.get('tau', 0.0),
        scalars.get('theta', 0.0),
        scalars.get('delta', 0.0),
        scalars.get('alpha', 0.0),
        scalars.get('Omega', 0.0),
    ])


def vector_to_apl_scalars(v: np.ndarray) -> dict:
    """Convert 9-element numpy vector to APL scalars dict."""
    return {
        'Gs': v[0] if len(v) > 0 else 0.0,
        'Cs': v[1] if len(v) > 1 else 0.0,
        'Rs': v[2] if len(v) > 2 else 0.0,
        'kappa': v[3] if len(v) > 3 else 0.0,
        'tau': v[4] if len(v) > 4 else 0.0,
        'theta': v[5] if len(v) > 5 else 0.0,
        'delta': v[6] if len(v) > 6 else 0.0,
        'alpha': v[7] if len(v) > 7 else 0.0,
        'Omega': v[8] if len(v) > 8 else 0.0,
    }


# =============================================================================
# METRIC UTILITIES
# =============================================================================

def all_distances(p1: 'GrassmannPoint', p2: 'GrassmannPoint') -> dict:
    """Compute all distance metrics between two points."""
    return {
        'geodesic': geodesic_distance(p1, p2),
        'chordal': chordal_distance(p1, p2),
        'projection': projection_frobenius(p1, p2),
        'binet_cauchy': binet_cauchy_distance(p1, p2),
        'asimov': asimov_distance(p1, p2),
    }


def metric_normalized(p1: 'GrassmannPoint', p2: 'GrassmannPoint',
                      metric: str = 'geodesic') -> float:
    """
    Compute normalized distance in [0, 1].

    Normalization depends on k:
    - Geodesic max = sqrt(k) * pi/2
    - Chordal max = sqrt(k)
    """
    k = min(p1.k, p2.k)

    if metric == 'geodesic':
        d = geodesic_distance(p1, p2)
        max_d = math.sqrt(k) * math.pi / 2
    elif metric == 'chordal':
        d = chordal_distance(p1, p2)
        max_d = math.sqrt(k)
    elif metric == 'projection':
        d = projection_frobenius(p1, p2)
        max_d = math.sqrt(k)
    elif metric == 'asimov':
        d = asimov_distance(p1, p2)
        max_d = math.pi / 2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return d / max_d if max_d > 0 else 0.0
