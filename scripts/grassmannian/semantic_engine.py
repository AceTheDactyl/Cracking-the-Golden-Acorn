"""
Grassmannian Semantic Engine
============================

Implements semantic processing using Grassmann manifold geometry.
Meanings are represented as subspaces, allowing:
- Compositional semantics via subspace operations
- Similarity via principal angles
- Context as ambient subspace

Key Components:
- SemanticSubspace: Word/phrase meaning as k-dimensional subspace
- APLSemanticEngine: Applies APL operators to semantic states
- CompositionalSemantics: Combines meanings via geometric operations

Theory:
Words and phrases span semantic subspaces. Composition (combining words)
corresponds to subspace operations. The APL operators provide structured
transformations reflecting cognitive/linguistic processes.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum

from .manifold import GrassmannManifold, GrassmannPoint, span_sum, span_intersection
from .metrics import (
    principal_angles, geodesic_distance, chordal_distance,
    z_from_points, coherence_from_angles, compute_apl_scalars
)
from .operators import GrassmannAPLOperators, APLOperator, OperatorResult
from .bridge import APLGrassmannBridge, APLPhase, APLState, phase_from_z


# =============================================================================
# SEMANTIC ROLES
# =============================================================================

class SemanticRole(Enum):
    """Semantic roles for compositional structure."""
    AGENT = "agent"           # Who performs action
    PATIENT = "patient"       # Who receives action
    INSTRUMENT = "instrument" # Means of action
    LOCATION = "location"     # Where action occurs
    TEMPORAL = "temporal"     # When action occurs
    MANNER = "manner"         # How action is performed
    CAUSE = "cause"           # Why action occurs
    GOAL = "goal"             # Purpose of action
    SOURCE = "source"         # Origin
    THEME = "theme"           # What is affected


class SemanticCategory(Enum):
    """High-level semantic categories."""
    ENTITY = "entity"       # Nouns, noun phrases
    ACTION = "action"       # Verbs, verbal phrases
    PROPERTY = "property"   # Adjectives, adverbs
    RELATION = "relation"   # Prepositions, connectives
    MODIFIER = "modifier"   # Quantifiers, determiners
    ABSTRACT = "abstract"   # Concepts, ideas


# =============================================================================
# SEMANTIC SUBSPACE
# =============================================================================

@dataclass
class SemanticSubspace:
    """
    Represents the meaning of a word or phrase as a Grassmann point.

    The subspace captures the semantic "directions" that the meaning
    spans. Multiple senses of a word span different directions.

    Attributes:
        point: The Grassmann point representing the meaning
        label: Human-readable label
        category: Semantic category
        roles: Set of semantic roles this can fill
        context_sensitive: Whether meaning varies with context
    """
    point: GrassmannPoint
    label: str
    category: SemanticCategory
    roles: Set[SemanticRole] = field(default_factory=set)
    context_sensitive: bool = False

    # Metadata
    frequency: float = 1.0  # Usage frequency (affects composition)
    specificity: float = 0.5  # How specific vs general (0=abstract, 1=concrete)
    valence: float = 0.0  # Emotional valence (-1=negative, +1=positive)

    @property
    def dimension(self) -> int:
        """Semantic dimension (subspace dimension)."""
        return self.point.k

    def similarity(self, other: 'SemanticSubspace') -> float:
        """
        Compute semantic similarity with another subspace.

        Uses mean cosine of principal angles.
        """
        angles = principal_angles(self.point, other.point)
        return coherence_from_angles(angles)

    def distance(self, other: 'SemanticSubspace', metric: str = 'geodesic') -> float:
        """Compute semantic distance."""
        return self.point.distance_to(other.point, metric)

    def z_relative_to(self, other: 'SemanticSubspace') -> float:
        """Compute Z-coordinate relative to another meaning."""
        return z_from_points(self.point, other.point)

    def contextualize(self, context: GrassmannPoint, strength: float = 0.3) -> 'SemanticSubspace':
        """
        Create context-adjusted meaning.

        Moves the semantic point toward the context.
        """
        manifold = GrassmannManifold(self.point.n, self.point.k)
        new_point = manifold.geodesic(self.point, context, strength)

        return SemanticSubspace(
            point=new_point,
            label=f"{self.label}[contextualized]",
            category=self.category,
            roles=self.roles.copy(),
            context_sensitive=True,
            frequency=self.frequency,
            specificity=self.specificity,
            valence=self.valence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'point': self.point.to_dict(),
            'label': self.label,
            'category': self.category.value,
            'roles': [r.value for r in self.roles],
            'context_sensitive': self.context_sensitive,
            'frequency': self.frequency,
            'specificity': self.specificity,
            'valence': self.valence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticSubspace':
        """Deserialize from dictionary."""
        return cls(
            point=GrassmannPoint.from_dict(data['point']),
            label=data['label'],
            category=SemanticCategory(data['category']),
            roles={SemanticRole(r) for r in data.get('roles', [])},
            context_sensitive=data.get('context_sensitive', False),
            frequency=data.get('frequency', 1.0),
            specificity=data.get('specificity', 0.5),
            valence=data.get('valence', 0.0),
        )

    @classmethod
    def random(cls, n: int, k: int, label: str,
               category: SemanticCategory = SemanticCategory.ENTITY) -> 'SemanticSubspace':
        """Create random semantic subspace."""
        point = GrassmannPoint.random(n, k)
        return cls(point=point, label=label, category=category)


# =============================================================================
# SEMANTIC LEXICON
# =============================================================================

class SemanticLexicon:
    """
    Dictionary of semantic subspaces for words.

    Provides lookup and creation of word meanings.
    """

    def __init__(self, n: int = 9, k: int = 3):
        """
        Initialize lexicon.

        Args:
            n: Ambient dimension for semantic space
            k: Subspace dimension for each meaning
        """
        self.n = n
        self.k = k
        self.manifold = GrassmannManifold(n, k)
        self.entries: Dict[str, SemanticSubspace] = {}

        # Category-specific base points
        self._category_bases: Dict[SemanticCategory, GrassmannPoint] = {}
        self._initialize_category_bases()

    def _initialize_category_bases(self) -> None:
        """Create base points for each semantic category."""
        # Distribute categories across the manifold
        num_cats = len(SemanticCategory)
        for i, cat in enumerate(SemanticCategory):
            # Create basis with category-specific structure
            basis = np.zeros((self.n, self.k))
            offset = (i * self.k) % self.n

            for j in range(self.k):
                idx = (offset + j) % self.n
                basis[idx, j] = 1.0
                # Add some variation
                for l in range(self.n):
                    if l != idx:
                        basis[l, j] = 0.1 * np.sin(2 * np.pi * (i + j + l) / self.n)

            self._category_bases[cat] = GrassmannPoint(basis)

    def add(self, word: str, subspace: SemanticSubspace) -> None:
        """Add word to lexicon."""
        self.entries[word] = subspace

    def get(self, word: str) -> Optional[SemanticSubspace]:
        """Look up word meaning."""
        return self.entries.get(word)

    def create(self, word: str, category: SemanticCategory,
               specificity: float = 0.5, valence: float = 0.0) -> SemanticSubspace:
        """
        Create and add new word meaning.

        Uses category base with random perturbation.
        """
        base = self._category_bases[category]

        # Perturb from category base
        perturbation = np.random.randn(self.n, self.k) * 0.3
        perturbation = perturbation - base.basis @ (base.basis.T @ perturbation)

        new_point = self.manifold.exp(base, perturbation)

        subspace = SemanticSubspace(
            point=new_point,
            label=word,
            category=category,
            specificity=specificity,
            valence=valence,
        )

        self.add(word, subspace)
        return subspace

    def nearest(self, query: SemanticSubspace, n: int = 5) -> List[Tuple[str, float]]:
        """
        Find nearest words to a query meaning.

        Returns list of (word, similarity) pairs.
        """
        similarities = []
        for word, subspace in self.entries.items():
            sim = query.similarity(subspace)
            similarities.append((word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def cluster_by_category(self) -> Dict[SemanticCategory, List[str]]:
        """Group words by their semantic category."""
        clusters: Dict[SemanticCategory, List[str]] = {cat: [] for cat in SemanticCategory}
        for word, subspace in self.entries.items():
            clusters[subspace.category].append(word)
        return clusters


# =============================================================================
# APL SEMANTIC ENGINE
# =============================================================================

class APLSemanticEngine:
    """
    Applies APL operators to semantic states.

    Maps cognitive/linguistic operations to APL operators:
    - BOUNDARY: Focus/attention (restricts meaning)
    - FUSION: Blending (combines meanings)
    - AMPLIFY: Emphasis (strengthens meaning)
    - DECOHERE: Ambiguity (introduces uncertainty)
    - GROUP: Collection (gathers related meanings)
    - SEPARATE: Distinction (differentiates meanings)
    """

    def __init__(self, lexicon: SemanticLexicon, enforce_n0: bool = True):
        """
        Initialize semantic engine.

        Args:
            lexicon: Semantic lexicon for word lookups
            enforce_n0: Whether to enforce N0 causality laws
        """
        self.lexicon = lexicon
        self.operators = GrassmannAPLOperators(lexicon.n, lexicon.k, enforce_n0)
        self.bridge = APLGrassmannBridge(lexicon.n, lexicon.k)

        # Processing state
        self.current_context: Optional[GrassmannPoint] = None
        self.processing_history: List[OperatorResult] = []

    def set_context(self, context: SemanticSubspace) -> None:
        """Set the current semantic context."""
        self.current_context = context.point
        self.operators.set_reference(context.point)

    def focus(self, meaning: SemanticSubspace) -> SemanticSubspace:
        """
        Apply BOUNDARY: focus attention on meaning.

        Projects toward the canonical frame, sharpening the meaning.
        """
        result = self.operators.boundary(meaning.point)
        self.processing_history.append(result)

        if result.success and result.output:
            return SemanticSubspace(
                point=result.output,
                label=f"focus({meaning.label})",
                category=meaning.category,
                roles=meaning.roles,
                specificity=min(1.0, meaning.specificity + 0.2),
                valence=meaning.valence,
            )
        return meaning

    def blend(self, m1: SemanticSubspace, m2: SemanticSubspace,
              weight: float = 0.5) -> SemanticSubspace:
        """
        Apply FUSION: blend two meanings.

        Creates a meaning that combines aspects of both inputs.
        """
        result = self.operators.fusion(m1.point, m2.point, weight)
        self.processing_history.append(result)

        if result.success and result.output:
            # Combine properties
            new_specificity = weight * m1.specificity + (1 - weight) * m2.specificity
            new_valence = weight * m1.valence + (1 - weight) * m2.valence

            return SemanticSubspace(
                point=result.output,
                label=f"blend({m1.label},{m2.label})",
                category=m1.category,  # Inherit from first
                roles=m1.roles | m2.roles,
                specificity=new_specificity,
                valence=new_valence,
            )
        return m1

    def emphasize(self, meaning: SemanticSubspace,
                  magnitude: float = 0.5) -> SemanticSubspace:
        """
        Apply AMPLIFY: emphasize/intensify meaning.

        Moves meaning away from neutral, increasing its distinctiveness.
        """
        result = self.operators.amplify(meaning.point, magnitude=magnitude)
        self.processing_history.append(result)

        if result.success and result.output:
            return SemanticSubspace(
                point=result.output,
                label=f"emphasize({meaning.label})",
                category=meaning.category,
                roles=meaning.roles,
                specificity=meaning.specificity,
                valence=meaning.valence * (1 + magnitude),  # Intensify valence
            )
        return meaning

    def ambiguate(self, meaning: SemanticSubspace,
                  entropy: float = 0.3) -> SemanticSubspace:
        """
        Apply DECOHERE: introduce ambiguity.

        Moves meaning toward more general/uncertain state.
        """
        result = self.operators.decohere(meaning.point, entropy)
        self.processing_history.append(result)

        if result.success and result.output:
            return SemanticSubspace(
                point=result.output,
                label=f"ambiguous({meaning.label})",
                category=meaning.category,
                roles=meaning.roles,
                specificity=max(0.0, meaning.specificity - entropy),
                valence=meaning.valence * (1 - entropy),  # Reduce valence
            )
        return meaning

    def collect(self, m1: SemanticSubspace, m2: SemanticSubspace) -> SemanticSubspace:
        """
        Apply GROUP: collect related meanings.

        Creates a meaning spanning both input meanings.
        """
        result = self.operators.group(m1.point, m2.point)
        self.processing_history.append(result)

        if result.success and result.output:
            return SemanticSubspace(
                point=result.output,
                label=f"collect({m1.label},{m2.label})",
                category=SemanticCategory.ABSTRACT,  # Collection is abstract
                roles=m1.roles | m2.roles,
                specificity=min(m1.specificity, m2.specificity),
                valence=(m1.valence + m2.valence) / 2,
            )
        return m1

    def distinguish(self, m1: SemanticSubspace, m2: SemanticSubspace) -> SemanticSubspace:
        """
        Apply SEPARATE: distinguish/differentiate.

        Creates meaning capturing what's unique to m1 relative to m2.
        """
        result = self.operators.separate(m1.point, m2.point)
        self.processing_history.append(result)

        if result.success and result.output:
            return SemanticSubspace(
                point=result.output,
                label=f"distinct({m1.label},{m2.label})",
                category=m1.category,
                roles=m1.roles - m2.roles,  # Roles unique to m1
                specificity=m1.specificity,
                valence=m1.valence,
            )
        return m1

    def get_apl_state(self, meaning: SemanticSubspace) -> APLState:
        """Get APL state for a semantic subspace."""
        return self.bridge.grassmann_to_apl(meaning.point, self.current_context)

    def reset(self) -> None:
        """Reset processing state."""
        self.current_context = None
        self.processing_history = []
        self.operators.reset()


# =============================================================================
# COMPOSITIONAL SEMANTICS
# =============================================================================

class CompositionalSemantics:
    """
    Implements compositional semantics via Grassmann operations.

    Provides rules for combining word meanings into phrase meanings.
    """

    def __init__(self, engine: APLSemanticEngine):
        """
        Initialize compositional semantics.

        Args:
            engine: APL semantic engine for operations
        """
        self.engine = engine
        self.lexicon = engine.lexicon

    def compose_modifier_noun(self, modifier: SemanticSubspace,
                               noun: SemanticSubspace) -> SemanticSubspace:
        """
        Compose modifier + noun (e.g., "red ball").

        Blends modifier into noun, with noun dominant.
        """
        # Modifier restricts noun meaning
        return self.engine.blend(noun, modifier, weight=0.7)

    def compose_verb_object(self, verb: SemanticSubspace,
                            obj: SemanticSubspace) -> SemanticSubspace:
        """
        Compose verb + object (e.g., "eat apple").

        Groups verb and object, then focuses.
        """
        combined = self.engine.collect(verb, obj)
        return self.engine.focus(combined)

    def compose_subject_predicate(self, subject: SemanticSubspace,
                                   predicate: SemanticSubspace) -> SemanticSubspace:
        """
        Compose subject + predicate (e.g., "dog runs").

        Fuses subject into predicate frame.
        """
        return self.engine.blend(predicate, subject, weight=0.6)

    def negate(self, meaning: SemanticSubspace) -> SemanticSubspace:
        """
        Negate a meaning (e.g., "not happy").

        Moves toward complement, inverts valence.
        """
        # Get complement direction
        complement = meaning.point.complement()
        negated_point = GrassmannManifold(
            meaning.point.n, meaning.point.k
        ).geodesic(meaning.point, complement, 0.5)

        return SemanticSubspace(
            point=negated_point,
            label=f"not({meaning.label})",
            category=meaning.category,
            roles=meaning.roles,
            specificity=meaning.specificity,
            valence=-meaning.valence,
        )

    def quantify(self, meaning: SemanticSubspace,
                 quantifier: str = "some") -> SemanticSubspace:
        """
        Apply quantifier (e.g., "all", "some", "no").

        Adjusts specificity based on quantifier strength.
        """
        quantifier_strength = {
            "all": 1.0,
            "most": 0.8,
            "many": 0.6,
            "some": 0.4,
            "few": 0.2,
            "no": 0.0,
        }

        strength = quantifier_strength.get(quantifier.lower(), 0.5)

        if strength > 0.5:
            # Strengthen meaning
            result = self.engine.emphasize(meaning, magnitude=strength - 0.5)
        else:
            # Weaken meaning
            result = self.engine.ambiguate(meaning, entropy=0.5 - strength)

        result.label = f"{quantifier}({meaning.label})"
        return result

    def parse_and_compose(self, tokens: List[str],
                          structure: Optional[List] = None) -> SemanticSubspace:
        """
        Parse tokens and compose their meanings.

        Simple left-to-right composition if no structure given.

        Args:
            tokens: List of word tokens
            structure: Optional parse structure (nested lists)

        Returns:
            Composed semantic subspace
        """
        if not tokens:
            return SemanticSubspace.random(
                self.lexicon.n, self.lexicon.k,
                "empty", SemanticCategory.ABSTRACT
            )

        # Look up or create meanings
        meanings = []
        for token in tokens:
            meaning = self.lexicon.get(token)
            if meaning is None:
                # Create on the fly
                meaning = self.lexicon.create(token, SemanticCategory.ENTITY)
            meanings.append(meaning)

        if len(meanings) == 1:
            return meanings[0]

        # Simple left-to-right composition
        result = meanings[0]
        for meaning in meanings[1:]:
            result = self.engine.blend(result, meaning, weight=0.5)

        result.label = " ".join(tokens)
        return result

    def similarity_matrix(self, subspaces: List[SemanticSubspace]) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            subspaces: List of semantic subspaces

        Returns:
            n x n similarity matrix
        """
        n = len(subspaces)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i, j] = subspaces[i].similarity(subspaces[j])

        return matrix
