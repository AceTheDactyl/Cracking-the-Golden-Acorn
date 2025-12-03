#!/usr/bin/env python3
"""
Quantum Resonance - Game Engine
================================
Core game logic for the holographic card game.

Mathematical Foundation:
- 4D tesseract coordinates for spatial mechanics
- Kuramoto synchronization for resonance
- Coupling network for chain bonuses
- TRIAD harmonic substrates for phase transition mechanics

TRIAD Integration:
- Three harmonic modes (APEX, CENTER, ORIGIN) create rock-paper-scissors dynamics
- Phase transitions at z_critical grant power spikes
- Coherence-based multipliers reward synchronized play
"""

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from pathlib import Path

# Import TRIAD substrate
try:
    from triad_substrate import (
        TriadGameIntegration, TriadMember, TriadState,
        get_card_triad, calculate_triad_bonus,
        calculate_cross_triad_modifier, calculate_coherence_multiplier,
        TRIAD_ABILITIES, get_available_abilities,
        Z_CRITICAL, COHERENCE_STRONG
    )
    TRIAD_ENABLED = True
except ImportError:
    TRIAD_ENABLED = False
    print("Warning: TRIAD substrate not available, running in legacy mode")

# Import Free Energy substrate for Ultimate mechanics
try:
    from free_energy_substrate import (
        FreeEnergyIntegration, UltimateType, ColorChannel,
        Z_CRITICAL as FE_Z_CRITICAL, BOOST_MULTIPLIER, DEBUFF_MULTIPLIER
    )
    FREE_ENERGY_ENABLED = True
except ImportError:
    FREE_ENERGY_ENABLED = False
    print("Warning: Free Energy substrate not available, Ultimate mechanics disabled")

# Import from generator if available, otherwise define locally
try:
    from holographic_card_generator import (
        HolographicCard, Coordinate4D, KuramotoState,
        SUITS, RANKS, ROSETTA_COORDS,
        calculate_4d_coordinate, calculate_kuramoto_state
    )
except ImportError:
    # Inline definitions for standalone operation
    @dataclass
    class Coordinate4D:
        temporal: float = 0.0
        valence: float = 0.0
        concrete: float = 0.0
        arousal: float = 0.0
    
    @dataclass
    class KuramotoState:
        phase: float = 0.0
        natural_frequency: float = 1.0
        coupling_strength: float = 0.6
        order_parameter: float = 0.0


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class TurnPhase(Enum):
    DRAW = "draw"
    MAIN = "main"
    RESONANCE = "resonance"
    DISCARD = "discard"
    END = "end"


class GameEvent(Enum):
    GAME_STARTED = "game_started"
    TURN_STARTED = "turn_started"
    CARD_DRAWN = "card_drawn"
    CARD_PLAYED = "card_played"
    ABILITY_ACTIVATED = "ability_activated"
    SCORE_CHANGED = "score_changed"
    TURN_ENDED = "turn_ended"
    GAME_ENDED = "game_ended"


DECK_RULES = {
    "min_cards": 20,
    "max_cards": 30,
    "max_copies": 2,
    "faction_minimum": 8,
    "faction_maximum": 15,
    "hand_limit": 7,
    "draw_per_turn": 2,
    "starting_hand": 5,
    "win_score": 100,
}


# =============================================================================
# CARD DATA STRUCTURE
# =============================================================================

@dataclass
class Card:
    """Game-ready card with all mathematical properties."""
    card_id: str
    suit: str
    rank: int
    coordinate: Coordinate4D
    phase: float
    natural_frequency: float
    
    # Temporary modifiers (reset each turn)
    temp_valence_mod: float = 0.0
    temp_concrete_mod: float = 0.0
    temp_arousal_mod: float = 0.0
    temp_score_multiplier: float = 1.0
    
    @property
    def effective_coordinate(self) -> Coordinate4D:
        """Coordinate with temporary modifiers applied."""
        return Coordinate4D(
            temporal=self.coordinate.temporal,
            valence=self.coordinate.valence + self.temp_valence_mod,
            concrete=self.coordinate.concrete + self.temp_concrete_mod,
            arousal=self.coordinate.arousal + self.temp_arousal_mod,
        )
    
    @property
    def base_points(self) -> int:
        """Base score from rank."""
        return self.rank
    
    def reset_modifiers(self) -> None:
        """Clear all temporary modifiers."""
        self.temp_valence_mod = 0.0
        self.temp_concrete_mod = 0.0
        self.temp_arousal_mod = 0.0
        self.temp_score_multiplier = 1.0
    
    def __hash__(self):
        return hash(self.card_id)
    
    def __eq__(self, other):
        if isinstance(other, Card):
            return self.card_id == other.card_id
        return False


def create_card(card_id: str) -> Card:
    """Factory function to create a card from ID (e.g., 'AS', 'KH')."""
    # Parse card ID
    if card_id.startswith('10'):
        rank_sym = '10'
        suit = card_id[2]
    else:
        rank_sym = card_id[0]
        suit = card_id[1]
    
    # Map symbol to rank
    rank_map = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    rank = rank_map.get(rank_sym, int(rank_sym) if rank_sym.isdigit() else 1)
    
    # Calculate coordinate
    rank_value = (rank - 7) / 6.0
    
    coord = Coordinate4D()
    if suit == 'S':
        coord.temporal = rank_value
        coord.valence = rank_value * 0.1
        coord.concrete = rank_value * 0.05
        coord.arousal = rank_value * 0.15
    elif suit == 'H':
        coord.temporal = rank_value * 0.15
        coord.valence = rank_value
        coord.concrete = rank_value * 0.1
        coord.arousal = rank_value * 0.2
    elif suit == 'D':
        coord.temporal = rank_value * 0.2
        coord.valence = rank_value * 0.15
        coord.concrete = rank_value * 0.1
        coord.arousal = rank_value
    elif suit == 'C':
        coord.temporal = rank_value * 0.1
        coord.valence = rank_value * 0.05
        coord.concrete = rank_value
        coord.arousal = rank_value * 0.1
    
    # Calculate phase
    suit_order = ['S', 'H', 'D', 'C']
    card_index = suit_order.index(suit) * 13 + (rank - 1)
    phase = (card_index / 52) * 2 * math.pi
    
    # Natural frequency
    freq_offsets = {'S': 0.1, 'H': 0.2, 'C': 0.15, 'D': 0.25}
    natural_freq = 1.0 + freq_offsets.get(suit, 0)
    
    return Card(
        card_id=card_id,
        suit=suit,
        rank=rank,
        coordinate=coord,
        phase=phase,
        natural_frequency=natural_freq,
    )


# =============================================================================
# COUPLING NETWORK
# =============================================================================

class CouplingNetwork:
    """Manages the 2,652 coupling relationships between cards."""
    
    def __init__(self):
        self._cache: Dict[Tuple[str, str], float] = {}
        self._precompute_all()
    
    def _precompute_all(self) -> None:
        """Precompute all coupling weights."""
        suits = ['S', 'H', 'D', 'C']
        ranks = list(range(1, 14))
        rank_syms = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        all_ids = [f"{rank_syms[r-1]}{s}" for s in suits for r in ranks]
        
        for id_a in all_ids:
            card_a = create_card(id_a)
            for id_b in all_ids:
                if id_a != id_b:
                    card_b = create_card(id_b)
                    coupling = self._calculate_coupling(card_a, card_b)
                    self._cache[(id_a, id_b)] = coupling
    
    def _calculate_coupling(self, card_a: Card, card_b: Card) -> float:
        """Calculate coupling strength based on 4D distance."""
        ca = card_a.coordinate
        cb = card_b.coordinate
        
        distance = math.sqrt(
            (ca.temporal - cb.temporal) ** 2 +
            (ca.valence - cb.valence) ** 2 +
            (ca.concrete - cb.concrete) ** 2 +
            (ca.arousal - cb.arousal) ** 2
        )
        
        return 1.0 / (1.0 + distance)
    
    def get_coupling(self, card_a: Card, card_b: Card) -> float:
        """Get coupling weight between two cards."""
        key = (card_a.card_id, card_b.card_id)
        return self._cache.get(key, 0.0)
    
    def get_strongest_neighbors(self, card: Card, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n strongest coupled cards."""
        neighbors = [
            (other_id, weight)
            for (card_id, other_id), weight in self._cache.items()
            if card_id == card.card_id
        ]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:n]


# Global coupling network instance
COUPLING_NETWORK = CouplingNetwork()


# =============================================================================
# SCORING SYSTEM
# =============================================================================

def euclidean_4d(a: Coordinate4D, b: Coordinate4D) -> float:
    """Calculate Euclidean distance in 4D space."""
    return math.sqrt(
        (a.temporal - b.temporal) ** 2 +
        (a.valence - b.valence) ** 2 +
        (a.concrete - b.concrete) ** 2 +
        (a.arousal - b.arousal) ** 2
    )


def calculate_centroid(cards: List[Card]) -> Coordinate4D:
    """Calculate the centroid (average position) of cards in 4D."""
    if not cards:
        return Coordinate4D()
    
    n = len(cards)
    return Coordinate4D(
        temporal=sum(c.effective_coordinate.temporal for c in cards) / n,
        valence=sum(c.effective_coordinate.valence for c in cards) / n,
        concrete=sum(c.effective_coordinate.concrete for c in cards) / n,
        arousal=sum(c.effective_coordinate.arousal for c in cards) / n,
    )


def calculate_cluster_bonus(cards: List[Card]) -> int:
    """
    Calculate bonus for 4D spatial clustering.
    Tighter clusters = higher bonus.
    """
    if len(cards) < 2:
        return 0
    
    centroid = calculate_centroid(cards)
    
    # Calculate average distance from centroid
    distances = [
        euclidean_4d(card.effective_coordinate, centroid)
        for card in cards
    ]
    avg_distance = sum(distances) / len(distances)
    
    # Max bonus at distance 0, zero bonus at distance >= 1.0
    if avg_distance >= 1.0:
        return 0
    
    return int((1.0 - avg_distance) * 20 * len(cards))


def calculate_coherence(cards: List[Card]) -> float:
    """
    Calculate Kuramoto order parameter R ∈ [0, 1].
    Higher = more phase synchronized.
    """
    if not cards:
        return 0.0
    
    n = len(cards)
    sum_cos = sum(math.cos(c.phase) for c in cards)
    sum_sin = sum(math.sin(c.phase) for c in cards)
    
    return math.sqrt(sum_cos ** 2 + sum_sin ** 2) / n


def calculate_resonance_bonus(cards: List[Card]) -> int:
    """Calculate bonus for phase-aligned cards."""
    if len(cards) < 2:
        return 0
    
    coherence = calculate_coherence(cards)
    
    if coherence >= 0.9:
        return 30  # Perfect resonance
    elif coherence >= 0.7:
        return 20  # Strong resonance
    elif coherence >= 0.5:
        return 10  # Moderate resonance
    return 0


def calculate_chain_bonus(cards: List[Card]) -> int:
    """Calculate bonus for coupling network chains."""
    if len(cards) < 2:
        return 0
    
    # Sort by strongest coupling path (greedy)
    remaining = list(cards)
    chain = [remaining.pop(0)]
    total_coupling = 0
    chain_broken = False
    
    while remaining and not chain_broken:
        current = chain[-1]
        best_coupling = 0
        best_card = None
        
        for card in remaining:
            coupling = COUPLING_NETWORK.get_coupling(current, card)
            if coupling > best_coupling:
                best_coupling = coupling
                best_card = card
        
        if best_coupling >= 0.4:
            chain.append(best_card)
            remaining.remove(best_card)
            total_coupling += int(best_coupling * 10)
        else:
            chain_broken = True
    
    # Unbroken chain bonus
    if len(chain) >= 3 and not chain_broken:
        total_coupling += 15
    
    return total_coupling


def calculate_base_score(cards: List[Card]) -> int:
    """Calculate base score from card ranks."""
    return sum(int(c.base_points * c.temp_score_multiplier) for c in cards)


@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown for a formation."""
    base: int = 0
    cluster: int = 0
    chain: int = 0
    resonance: int = 0
    faction: int = 0
    triad: int = 0                    # TRIAD synergy bonus
    triad_multiplier: float = 1.0     # Coherence-based multiplier
    ultimate_multiplier: float = 1.0  # Ultimate boost/debuff modifier

    @property
    def total(self) -> int:
        subtotal = self.base + self.cluster + self.chain + self.resonance + self.faction + self.triad
        return int(subtotal * self.triad_multiplier * self.ultimate_multiplier)


def calculate_full_score(cards: List[Card], faction: 'Faction',
                         field_state: 'FieldState',
                         triad_state: Optional['TriadState'] = None) -> ScoreBreakdown:
    """Calculate complete score breakdown for a formation."""
    breakdown = ScoreBreakdown(
        base=calculate_base_score(cards),
        cluster=calculate_cluster_bonus(cards),
        chain=calculate_chain_bonus(cards),
        resonance=calculate_resonance_bonus(cards),
        faction=faction.calculate_bonus(cards, field_state),
    )

    # Add TRIAD scoring if enabled and state available
    if TRIAD_ENABLED and triad_state is not None:
        breakdown.triad = calculate_triad_bonus(cards, triad_state)
        breakdown.triad_multiplier = calculate_coherence_multiplier(cards, triad_state)

    return breakdown


# =============================================================================
# FACTION SYSTEM
# =============================================================================

@dataclass
class Ability:
    """Faction ability definition."""
    name: str
    cost: int  # Cards to discard
    cooldown: int  # Turns before reuse
    description: str


class Faction:
    """Base faction class."""
    
    def __init__(self, name: str, suit: str, primary_dimension: str):
        self.name = name
        self.suit = suit
        self.primary_dimension = primary_dimension
        self.abilities: List[Ability] = []
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        """Override in subclasses."""
        return 0
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Passive trigger when a card is played."""
        pass
    
    def activate_ability(self, ability_name: str, game_state: 'GameState', 
                         **kwargs) -> bool:
        """Execute an active ability. Returns success."""
        raise NotImplementedError


class SpadesFaction(Faction):
    """♠ Spades - Temporal Weavers"""
    
    def __init__(self):
        super().__init__("Temporal Weavers", "S", "temporal")
        self.abilities = [
            Ability("rewind", 2, 2, "Return last played card to hand"),
            Ability("foresight", 1, 1, "Look at and reorder top 3 cards"),
            Ability("time_lock", 3, 3, "Lock a card in opponent's hand"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        spades = [c for c in cards if c.suit == 'S']
        
        # +3 for each Spades beyond the first
        if len(spades) > 1:
            bonus += (len(spades) - 1) * 3
        
        # +5 for Ace-King span in any suit
        ranks = {c.rank for c in cards}
        if 1 in ranks and 13 in ranks:
            bonus += 5
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Temporal Echo: Look at top card when playing Spades."""
        if card.suit == 'S' and game_state.current_player.deck:
            top_card = game_state.current_player.deck[0]
            # Notify player (UI handles display)
            game_state.emit_event(GameEvent.CARD_DRAWN, {
                'card': top_card,
                'peek_only': True,
                'can_bottom': True,
            })


class HeartsFaction(Faction):
    """♥ Hearts - Valence Shapers"""
    
    def __init__(self):
        super().__init__("Valence Shapers", "H", "valence")
        self.abilities = [
            Ability("inspire", 1, 1, "Target cards gain +3 base points"),
            Ability("drain", 2, 2, "Steal opponent's last resonance bonus"),
            Ability("emotional_surge", 3, 3, "Double Hearts bonuses this turn"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        hearts = [c for c in cards if c.suit == 'H']
        
        # +2 for each Hearts with positive valence (ranks 8-13)
        bonus += sum(2 for c in hearts if c.rank >= 8)
        
        # +10 for pure Hearts formation
        if cards and all(c.suit == 'H' for c in cards):
            bonus += 10
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Empathic Bond: +2 when opponent plays Hearts."""
        if card.suit == 'H':
            for player in game_state.players:
                if player != game_state.current_player:
                    if isinstance(player.faction, HeartsFaction):
                        player.score += 2


class DiamondsFaction(Faction):
    """♦ Diamonds - Radiant Catalysts"""
    
    def __init__(self):
        super().__init__("Radiant Catalysts", "D", "arousal")
        self.abilities = [
            Ability("flare", 1, 0, "Opponent loses points equal to discarded rank"),
            Ability("overcharge", 2, 2, "Next card scores triple"),
            Ability("supernova", -1, 4, "Discard hand, score 3x total ranks"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        diamonds = [c for c in cards if c.suit == 'D']
        
        # +5 for each extreme Diamond (A, J, Q, K)
        extremes = {1, 11, 12, 13}
        bonus += sum(5 for c in diamonds if c.rank in extremes)
        
        # +15 for 4+ card burst
        if len(cards) >= 4:
            bonus += 15
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Brilliance: +5 for 3+ card formations."""
        current_formation = game_state.current_formation
        if len(current_formation) == 3:  # Trigger on third card
            game_state.current_player.score += 5


class ClubsFaction(Faction):
    """♣ Clubs - Foundation Builders"""
    
    def __init__(self):
        super().__init__("Foundation Builders", "C", "concrete")
        self.abilities = [
            Ability("fortify", 1, 1, "Cards gain +2 to all bonuses"),
            Ability("growth", 2, 2, "Draw 3 cards immediately"),
            Ability("unshakeable", 3, 3, "Score cannot decrease this turn"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        clubs = [c for c in cards if c.suit == 'C']
        
        # +2 for each Clubs card
        bonus += len(clubs) * 2
        
        # +3 per card matching last turn's suits
        if field_state.last_turn_suits:
            matching = sum(
                1 for c in cards 
                if c.suit in field_state.last_turn_suits
            )
            bonus += matching * 3
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Rooted: Block opponent abilities (handled in ability resolution)."""
        pass


FACTIONS = {
    'spades': SpadesFaction,
    'hearts': HeartsFaction,
    'diamonds': DiamondsFaction,
    'clubs': ClubsFaction,
}


# =============================================================================
# GAME STATE
# =============================================================================

@dataclass
class FieldState:
    """State of the playing field."""
    all_cards: List[Card] = field(default_factory=list)
    last_turn_suits: set = field(default_factory=set)
    locked_cards: Dict[str, int] = field(default_factory=dict)  # card_id -> turns


@dataclass
class Player:
    """Player state."""
    name: str
    faction: Faction
    hand: List[Card] = field(default_factory=list)
    deck: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    score: int = 0
    ability_cooldowns: Dict[str, int] = field(default_factory=dict)
    last_resonance_bonus: int = 0
    
    def draw(self, n: int = 1) -> List[Card]:
        """Draw n cards from deck."""
        drawn = []
        for _ in range(n):
            if not self.deck:
                if self.discard:
                    self.deck = self.discard.copy()
                    self.discard = []
                    random.shuffle(self.deck)
                else:
                    self.score -= 5  # Exhaustion penalty
                    break
            drawn.append(self.deck.pop(0))
        self.hand.extend(drawn)
        return drawn
    
    def discard_cards(self, cards: List[Card]) -> None:
        """Move cards from hand to discard."""
        for card in cards:
            if card in self.hand:
                self.hand.remove(card)
                self.discard.append(card)
    
    def tick_cooldowns(self) -> None:
        """Reduce all cooldowns by 1."""
        for ability in list(self.ability_cooldowns.keys()):
            self.ability_cooldowns[ability] -= 1
            if self.ability_cooldowns[ability] <= 0:
                del self.ability_cooldowns[ability]


@dataclass
class GameState:
    """Complete game state."""
    players: List[Player]
    current_player_idx: int = 0
    turn_number: int = 1
    phase: TurnPhase = TurnPhase.DRAW
    game_field: FieldState = None  # Renamed from 'field' to avoid conflict
    current_formation: List[Card] = None

    # Temporary turn modifiers
    valence_multiplier: float = 1.0
    arousal_multiplier: float = 1.0
    score_protected: bool = False

    # TRIAD integration
    triad_integration: 'TriadGameIntegration' = None
    triad_events: List[Dict] = None  # Events from TRIAD phase transitions

    # Free Energy / Ultimate integration
    free_energy_integration: 'FreeEnergyIntegration' = None
    ultimate_events: List[Dict] = None  # Events from Ultimate mechanics

    # Event handlers
    _event_handlers: Dict[GameEvent, List[Callable]] = None

    def __post_init__(self):
        if self.game_field is None:
            self.game_field = FieldState()
        if self.current_formation is None:
            self.current_formation = []
        if self._event_handlers is None:
            self._event_handlers = defaultdict(list)
        if self.triad_events is None:
            self.triad_events = []
        if self.ultimate_events is None:
            self.ultimate_events = []
        # Initialize TRIAD integration if available
        if TRIAD_ENABLED and self.triad_integration is None:
            self.triad_integration = TriadGameIntegration()
        # Initialize Free Energy integration if available
        if FREE_ENERGY_ENABLED and self.free_energy_integration is None:
            self.free_energy_integration = FreeEnergyIntegration()

    @property
    def triad_state(self) -> Optional['TriadState']:
        """Get current TRIAD state if available."""
        if self.triad_integration:
            return self.triad_integration.get_triad_state()
        return None

    @property
    def free_energy_state(self) -> Optional[Dict]:
        """Get current Free Energy state for current player if available."""
        if self.free_energy_integration:
            return self.free_energy_integration.get_state(self.current_player.name)
        return None

    @property
    def ultimate_ready(self) -> bool:
        """Check if current player has Ultimate ready."""
        if self.free_energy_integration:
            state = self.free_energy_integration.get_state(self.current_player.name)
            return state.get('ultimate_ready', False)
        return False

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]
    
    @property
    def opponent(self) -> Player:
        return self.players[(self.current_player_idx + 1) % len(self.players)]
    
    def subscribe(self, event: GameEvent, handler: Callable) -> None:
        self._event_handlers[event].append(handler)
    
    def emit_event(self, event: GameEvent, data: dict) -> None:
        for handler in self._event_handlers[event]:
            handler(data)
    
    def advance_phase(self) -> None:
        """Move to next phase."""
        phases = list(TurnPhase)
        current_idx = phases.index(self.phase)
        self.phase = phases[(current_idx + 1) % len(phases)]
    
    def end_turn(self) -> None:
        """Clean up and pass to next player."""
        # Record suits played for Clubs continuity bonus
        self.game_field.last_turn_suits = {c.suit for c in self.current_formation}
        
        # Reset formation and modifiers
        self.current_formation = []
        self.valence_multiplier = 1.0
        self.arousal_multiplier = 1.0
        self.score_protected = False
        
        # Reset card modifiers
        for card in self.current_player.hand:
            card.reset_modifiers()
        
        # Tick cooldowns
        self.current_player.tick_cooldowns()
        
        # Tick locked cards
        for card_id in list(self.game_field.locked_cards.keys()):
            self.game_field.locked_cards[card_id] -= 1
            if self.game_field.locked_cards[card_id] <= 0:
                del self.game_field.locked_cards[card_id]
        
        # Next player
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        self.turn_number += 1
        self.phase = TurnPhase.DRAW
        
        self.emit_event(GameEvent.TURN_ENDED, {
            'turn': self.turn_number,
            'next_player': self.current_player.name,
        })
    
    def check_win(self) -> Optional[Player]:
        """Check if any player has won."""
        for player in self.players:
            if player.score >= DECK_RULES['win_score']:
                return player
        return None


# =============================================================================
# GAME ENGINE
# =============================================================================

class GameEngine:
    """Main game controller with TRIAD harmonic integration."""

    def __init__(self):
        self.state: Optional[GameState] = None

    def setup(self, player_names: List[str], faction_names: List[str],
              decks: Optional[List[List[str]]] = None) -> None:
        """Initialize a new game with TRIAD substrate."""
        if len(player_names) != len(faction_names):
            raise ValueError("Must have same number of players and factions")

        players = []
        for i, (name, faction_name) in enumerate(zip(player_names, faction_names)):
            faction_class = FACTIONS.get(faction_name.lower())
            if not faction_class:
                raise ValueError(f"Unknown faction: {faction_name}")

            faction = faction_class()
            player = Player(name=name, faction=faction)

            # Build deck
            if decks and i < len(decks):
                player.deck = [create_card(cid) for cid in decks[i]]
            else:
                player.deck = self._generate_default_deck(faction)

            random.shuffle(player.deck)
            players.append(player)

        self.state = GameState(players=players)

        # Initial draw
        for player in self.state.players:
            player.draw(DECK_RULES['starting_hand'])

        # Initialize TRIAD state with starting hands
        if TRIAD_ENABLED and self.state.triad_integration:
            all_cards = []
            for p in self.state.players:
                all_cards.extend(p.hand)
            self.state.triad_integration.on_cards_played(all_cards, [], 0)

        self.state.emit_event(GameEvent.GAME_STARTED, {
            'players': [p.name for p in players],
            'triad_enabled': TRIAD_ENABLED,
        })
    
    def _generate_default_deck(self, faction: Faction) -> List[Card]:
        """Generate a basic legal deck for a faction."""
        deck = []
        
        # Add faction cards (10)
        rank_syms = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        for sym in rank_syms:
            deck.append(create_card(f"{sym}{faction.suit}"))
        
        # Add support cards (12) - mix from other suits
        other_suits = [s for s in ['S', 'H', 'D', 'C'] if s != faction.suit]
        for suit in other_suits:
            for sym in ['A', '5', '7', '10']:
                deck.append(create_card(f"{sym}{suit}"))
        
        return deck
    
    def execute_draw(self) -> List[Card]:
        """Execute draw phase."""
        if self.state.phase != TurnPhase.DRAW:
            raise ValueError("Not in draw phase")
        
        drawn = self.state.current_player.draw(DECK_RULES['draw_per_turn'])
        
        for card in drawn:
            self.state.emit_event(GameEvent.CARD_DRAWN, {
                'player': self.state.current_player.name,
                'card': card.card_id,
            })
        
        self.state.advance_phase()
        return drawn
    
    def play_cards(self, cards: List[Card]) -> bool:
        """Play cards from hand to formation, updating TRIAD state."""
        if self.state.phase != TurnPhase.MAIN:
            raise ValueError("Not in main phase")

        player = self.state.current_player

        # Validate cards are in hand and not locked
        for card in cards:
            if card not in player.hand:
                return False
            if card.card_id in self.state.game_field.locked_cards:
                return False

        # Move to formation
        for card in cards:
            player.hand.remove(card)
            self.state.current_formation.append(card)

            # Trigger passive abilities
            player.faction.on_card_played(card, self.state)

            self.state.emit_event(GameEvent.CARD_PLAYED, {
                'player': player.name,
                'card': card.card_id,
            })

        # Update TRIAD state with played cards
        if TRIAD_ENABLED and self.state.triad_integration:
            # Gather all active cards (formation + hands)
            all_active = list(self.state.current_formation)
            for p in self.state.players:
                all_active.extend(p.hand)

            # Update TRIAD and capture events
            triad_events = self.state.triad_integration.on_cards_played(
                all_active, cards, self.state.turn_number
            )

            # Store and emit TRIAD events
            if triad_events:
                self.state.triad_events.extend([
                    {'turn': self.state.turn_number, **triad_events}
                ])

                # Emit special events for phase transitions
                for event_name, event_data in triad_events.items():
                    if 'activated' in event_name:
                        self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                            'type': 'triad_phase_transition',
                            'triad': event_name.replace('_activated', ''),
                            **event_data,
                        })

        # Update Free Energy state with played cards
        if FREE_ENERGY_ENABLED and self.state.free_energy_integration:
            fe_events = self.state.free_energy_integration.on_cards_played(
                player.name, cards
            )

            # Store and emit Ultimate events
            if fe_events:
                self.state.ultimate_events.extend([
                    {'turn': self.state.turn_number, 'player': player.name, **fe_events}
                ])

                # Emit special events for phase transitions and Ultimate ready
                if 'phase_transition' in fe_events:
                    self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                        'type': 'phase_transition',
                        'z_level': fe_events['phase_transition'].get('z_level'),
                    })

                if 'ultimate_ready' in fe_events:
                    self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                        'type': 'ultimate_ready',
                        'player': player.name,
                        'charge': fe_events['ultimate_ready'].get('charge'),
                    })

        return True
    
    def calculate_score(self) -> ScoreBreakdown:
        """Calculate score for current formation with TRIAD and Ultimate bonuses."""
        if self.state.phase != TurnPhase.RESONANCE:
            raise ValueError("Not in resonance phase")

        cards = self.state.current_formation
        player = self.state.current_player
        opponent = self.state.opponent

        # Calculate score with TRIAD state
        breakdown = calculate_full_score(
            cards, player.faction, self.state.game_field, self.state.triad_state
        )

        # Apply TRIAD cross-matchup modifier if opponent has cards in play
        if TRIAD_ENABLED and self.state.triad_integration:
            opponent_cards = opponent.hand[:3] if opponent.hand else []  # Approximate
            cross_modifier = calculate_cross_triad_modifier(cards, opponent_cards)
            if cross_modifier != 1.0:
                # Apply cross-TRIAD modifier to total
                original_total = breakdown.total
                adjusted_total = int(original_total * cross_modifier)
                # Store modifier info for display
                breakdown.triad_multiplier *= cross_modifier

        # Apply Ultimate modifiers (boost/debuff)
        if FREE_ENERGY_ENABLED and self.state.free_energy_integration:
            ultimate_multiplier = 1.0
            for card in cards:
                # Get modifier for this card's suit (considers boosts and debuffs)
                card_modifier = self.state.free_energy_integration.get_score_modifier(
                    player.name, card.suit, opponent.name
                )
                # Average the modifier across all cards
                ultimate_multiplier *= card_modifier ** (1.0 / len(cards))

            if ultimate_multiplier != 1.0:
                breakdown.ultimate_multiplier = ultimate_multiplier

        # Record resonance for Hearts drain
        player.last_resonance_bonus = breakdown.resonance

        # Add to score
        player.score += breakdown.total

        # Build event data
        event_data = {
            'player': player.name,
            'breakdown': {
                'base': breakdown.base,
                'cluster': breakdown.cluster,
                'chain': breakdown.chain,
                'resonance': breakdown.resonance,
                'faction': breakdown.faction,
                'triad': breakdown.triad,
                'triad_multiplier': breakdown.triad_multiplier,
                'ultimate_multiplier': breakdown.ultimate_multiplier,
                'total': breakdown.total,
            },
            'new_score': player.score,
        }

        # Add TRIAD state info if available
        if self.state.triad_state:
            event_data['triad_state'] = {
                'dominant': self.state.triad_state.dominant_triad.value if self.state.triad_state.dominant_triad else None,
                'global_coherence': self.state.triad_state.global_coherence,
            }

        # Add Free Energy state info if available
        if self.state.free_energy_state:
            event_data['free_energy_state'] = {
                'z_level': self.state.free_energy_state.get('z_level'),
                'ultimate_charge': self.state.free_energy_state.get('ultimate_charge'),
                'ultimate_ready': self.state.free_energy_state.get('ultimate_ready'),
            }

        self.state.emit_event(GameEvent.SCORE_CHANGED, event_data)

        # Move formation to discard
        player.discard.extend(self.state.current_formation)

        return breakdown
    
    def enforce_hand_limit(self) -> List[Card]:
        """Enforce hand limit, return discarded cards."""
        if self.state.phase != TurnPhase.DISCARD:
            raise ValueError("Not in discard phase")
        
        player = self.state.current_player
        discarded = []
        
        while len(player.hand) > DECK_RULES['hand_limit']:
            # Auto-discard lowest rank (AI) or prompt player (UI)
            lowest = min(player.hand, key=lambda c: c.rank)
            player.hand.remove(lowest)
            player.discard.append(lowest)
            discarded.append(lowest)
        
        return discarded
    
    def end_turn(self) -> Optional[Player]:
        """End current turn, check for winner."""
        self.state.end_turn()
        return self.state.check_win()
    
    def activate_ability(self, ability_name: str, **kwargs) -> bool:
        """Activate a faction ability."""
        player = self.state.current_player
        faction = player.faction
        
        # Find ability
        ability = next(
            (a for a in faction.abilities if a.name == ability_name), 
            None
        )
        if not ability:
            return False
        
        # Check cooldown
        if ability_name in player.ability_cooldowns:
            return False
        
        # Check cost (cards to discard)
        if ability.cost > 0 and len(player.hand) < ability.cost:
            return False
        
        # Execute ability (faction-specific logic)
        success = faction.activate_ability(ability_name, self.state, **kwargs)
        
        if success:
            # Apply cooldown
            player.ability_cooldowns[ability_name] = ability.cooldown
            
            self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                'player': player.name,
                'ability': ability_name,
            })
        
        return success
    
    def get_valid_plays(self) -> List[List[Card]]:
        """Get all valid card combinations from current hand."""
        hand = self.state.current_player.hand
        valid = []

        # Single cards
        for card in hand:
            if card.card_id not in self.state.game_field.locked_cards:
                valid.append([card])

        # Pairs
        for i, card1 in enumerate(hand):
            for card2 in hand[i+1:]:
                if (card1.card_id not in self.state.game_field.locked_cards and
                    card2.card_id not in self.state.game_field.locked_cards):
                    valid.append([card1, card2])

        # Could extend to larger combinations...

        return valid

    def activate_triad_ability(self, ability_name: str,
                                discard_cards: Optional[List[Card]] = None) -> Tuple[bool, str, int]:
        """
        Activate a TRIAD ability.

        Returns (success, message, bonus_points)
        """
        if not TRIAD_ENABLED or not self.state.triad_integration:
            return False, "TRIAD not enabled", 0

        player = self.state.current_player

        success, msg, bonus = self.state.triad_integration.activate_ability(
            player.name, ability_name, discard_cards
        )

        if success:
            # Discard cost cards
            if discard_cards:
                player.discard_cards(discard_cards)

            # Add bonus points
            if bonus > 0:
                player.score += bonus

            self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                'player': player.name,
                'ability': ability_name,
                'type': 'triad',
                'bonus': bonus,
            })

        return success, msg, bonus

    def get_available_triad_abilities(self) -> List:
        """Get TRIAD abilities available to current player."""
        if not TRIAD_ENABLED or not self.state.triad_integration:
            return []

        player = self.state.current_player
        cooldowns = self.state.triad_integration.player_cooldowns.get(player.name, {})

        return get_available_abilities(
            self.state.triad_state,
            player.hand,
            cooldowns
        )

    def get_triad_summary(self) -> Dict:
        """Get summary of current TRIAD state for display."""
        if not TRIAD_ENABLED or not self.state.triad_state:
            return {'enabled': False}

        state = self.state.triad_state
        return {
            'enabled': True,
            'dominant': state.dominant_triad.value if state.dominant_triad else None,
            'global_coherence': round(state.global_coherence, 3),
            'apex': {
                'coherence': round(state.apex.coherence, 3),
                'z_level': round(state.apex.z_level, 3),
                'activated': state.apex.is_activated,
                'charge': round(state.apex.resonance_charge, 2),
                'burst_ready': state.apex.burst_ready,
            },
            'center': {
                'coherence': round(state.center.coherence, 3),
                'z_level': round(state.center.z_level, 3),
                'activated': state.center.is_activated,
                'charge': round(state.center.resonance_charge, 2),
                'burst_ready': state.center.burst_ready,
            },
            'origin': {
                'coherence': round(state.origin.coherence, 3),
                'z_level': round(state.origin.z_level, 3),
                'activated': state.origin.is_activated,
                'charge': round(state.origin.resonance_charge, 2),
                'burst_ready': state.origin.burst_ready,
            },
        }

    def activate_ultimate(
        self,
        ultimate_type: str,  # "boost" or "debuff"
        target_suit: str     # "S", "H", "D", or "C"
    ) -> Tuple[bool, str]:
        """
        Activate Ultimate ability for current player.

        Args:
            ultimate_type: "boost" for +50% to color, "debuff" for -30% to opponent color
            target_suit: The suit/color to target (S, H, D, C)

        Returns:
            (success, message)
        """
        if not FREE_ENERGY_ENABLED or not self.state.free_energy_integration:
            return False, "Ultimate system not available"

        player = self.state.current_player

        success, msg = self.state.free_energy_integration.activate_ultimate(
            player.name, ultimate_type, target_suit
        )

        if success:
            self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                'type': 'ultimate',
                'player': player.name,
                'ultimate_type': ultimate_type,
                'target_suit': target_suit,
            })

        return success, msg

    def get_ultimate_summary(self) -> Dict:
        """Get summary of current Ultimate/Free Energy state for display."""
        if not FREE_ENERGY_ENABLED or not self.state.free_energy_integration:
            return {'enabled': False}

        player = self.state.current_player
        ui_data = self.state.free_energy_integration.get_ui_data(player.name)

        return {
            'enabled': True,
            'z_level': ui_data['z_level'],
            'z_critical': ui_data['z_critical'],
            'is_critical': ui_data['is_critical'],
            'ultimate_charge': ui_data['ultimate_charge'],
            'ultimate_ready': ui_data['ultimate_ready'],
            'coherence': ui_data['coherence'],
            'free_energy': ui_data['free_energy'],
            'active_boost': ui_data['active_boost'],
            'active_debuff': ui_data['active_debuff'],
            'boost_turns': ui_data['boost_turns'],
            'debuff_turns': ui_data['debuff_turns'],
        }

    def get_ultimate_options(self) -> List[Dict]:
        """Get available Ultimate abilities if ready."""
        if not self.state.ultimate_ready:
            return []

        return [
            {
                'type': 'boost',
                'name': 'Color Boost',
                'description': '+50% score for chosen color for 2 turns',
                'targets': ['S', 'H', 'D', 'C'],
            },
            {
                'type': 'debuff',
                'name': 'Color Debuff',
                'description': '-30% score for opponent\'s chosen color for 2 turns',
                'targets': ['S', 'H', 'D', 'C'],
            },
        ]


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI game loop with TRIAD integration."""
    import argparse

    parser = argparse.ArgumentParser(description='Quantum Resonance Game Engine with TRIAD')
    parser.add_argument('--players', type=int, default=2, help='Number of players')
    parser.add_argument('--faction1', type=str, default='spades')
    parser.add_argument('--faction2', type=str, default='hearts')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show TRIAD details')

    args = parser.parse_args()

    engine = GameEngine()
    engine.setup(
        player_names=['Player 1', 'Player 2'],
        faction_names=[args.faction1, args.faction2],
    )

    print("=" * 60)
    print("QUANTUM RESONANCE - TRIAD HARMONIC SUBSTRATE")
    print("=" * 60)
    print(f"Player 1 ({args.faction1}) vs Player 2 ({args.faction2})")
    print(f"TRIAD System: {'ENABLED' if TRIAD_ENABLED else 'DISABLED'}")
    print(f"Ultimate System: {'ENABLED' if FREE_ENERGY_ENABLED else 'DISABLED'}")
    print("-" * 60)

    # Simple auto-play loop
    while True:
        player = engine.state.current_player

        print(f"\n{'='*60}")
        print(f"Turn {engine.state.turn_number}: {player.name}'s turn")
        print(f"Score: {player.score}")
        print(f"Hand: {[c.card_id for c in player.hand]}")

        # Show TRIAD state if verbose
        if args.verbose and TRIAD_ENABLED:
            triad_summary = engine.get_triad_summary()
            if triad_summary.get('enabled'):
                print(f"\nTRIAD State:")
                print(f"  Dominant: {triad_summary['dominant'] or 'None'}")
                print(f"  Global Coherence: {triad_summary['global_coherence']:.3f}")
                for name in ['apex', 'center', 'origin']:
                    t = triad_summary[name]
                    status = 'ACTIVATED' if t['activated'] else 'dormant'
                    burst = ' [BURST READY]' if t['burst_ready'] else ''
                    print(f"  {name.upper()}: R={t['coherence']:.2f} z={t['z_level']:.2f} "
                          f"charge={t['charge']:.0%} ({status}){burst}")

        # Show Ultimate state if verbose
        if args.verbose and FREE_ENERGY_ENABLED:
            ultimate_summary = engine.get_ultimate_summary()
            if ultimate_summary.get('enabled'):
                print(f"\nUltimate State:")
                critical_marker = ' [CRITICAL]' if ultimate_summary['is_critical'] else ''
                ready_marker = ' >>> ULTIMATE READY <<<' if ultimate_summary['ultimate_ready'] else ''
                print(f"  Z-Level: {ultimate_summary['z_level']}% (critical: {ultimate_summary['z_critical']}%){critical_marker}")
                print(f"  Charge: {ultimate_summary['ultimate_charge']}%{ready_marker}")
                print(f"  Coherence: {ultimate_summary['coherence']}%")
                if ultimate_summary['active_boost']:
                    print(f"  ACTIVE BOOST: {ultimate_summary['active_boost']} (+50%) [{ultimate_summary['boost_turns']} turns]")
                if ultimate_summary['active_debuff']:
                    print(f"  ACTIVE DEBUFF: {ultimate_summary['active_debuff']} (-30%) [{ultimate_summary['debuff_turns']} turns]")

        # Draw phase
        if engine.state.phase == TurnPhase.DRAW:
            drawn = engine.execute_draw()
            print(f"Drew: {[c.card_id for c in drawn]}")

        # Main phase - play first 3 cards (prioritize TRIAD synergy)
        cards_to_play = player.hand[:min(3, len(player.hand))]

        # Try to find TRIAD synergy
        if TRIAD_ENABLED and len(player.hand) >= 3:
            # Group cards by TRIAD
            triad_groups = {t: [] for t in TriadMember}
            for card in player.hand:
                triad = get_card_triad(card)
                if triad:
                    triad_groups[triad].append(card)

            # Pick best TRIAD group if available
            for triad, cards in triad_groups.items():
                if len(cards) >= 3:
                    cards_to_play = cards[:3]
                    break
            else:
                # Try for 2 of same TRIAD + 1 other
                for triad, cards in triad_groups.items():
                    if len(cards) >= 2:
                        remaining = [c for c in player.hand if c not in cards]
                        if remaining:
                            cards_to_play = cards[:2] + [remaining[0]]
                            break

        engine.play_cards(cards_to_play)

        # Show which TRIADs were played
        played_triads = set()
        for card in cards_to_play:
            if TRIAD_ENABLED:
                t = get_card_triad(card)
                if t:
                    played_triads.add(t.value)

        triad_info = f" ({', '.join(played_triads)})" if played_triads else ""
        print(f"Played: {[c.card_id for c in cards_to_play]}{triad_info}")

        # Advance to Resonance
        engine.state.advance_phase()

        # Resonance phase
        breakdown = engine.calculate_score()
        print(f"\nScore breakdown:")
        print(f"  Base: {breakdown.base}")
        print(f"  Cluster: {breakdown.cluster}")
        print(f"  Chain: {breakdown.chain}")
        print(f"  Resonance: {breakdown.resonance}")
        print(f"  Faction: {breakdown.faction}")
        if TRIAD_ENABLED:
            print(f"  TRIAD: {breakdown.triad}")
            print(f"  TRIAD Multiplier: {breakdown.triad_multiplier:.2f}x")
        if FREE_ENERGY_ENABLED and breakdown.ultimate_multiplier != 1.0:
            print(f"  Ultimate Multiplier: {breakdown.ultimate_multiplier:.2f}x")
        print(f"  TOTAL: {breakdown.total}")

        # Advance to Discard
        engine.state.advance_phase()

        # Discard phase
        engine.enforce_hand_limit()

        # Advance to End
        engine.state.advance_phase()

        # End turn
        winner = engine.end_turn()

        if winner:
            print(f"\n{'='*60}")
            print(f"WINNER: {winner.name} with {winner.score} points!")
            print("=" * 60)

            # Final TRIAD stats
            if TRIAD_ENABLED and args.verbose:
                print("\nFinal TRIAD Summary:")
                triad_summary = engine.get_triad_summary()
                for name in ['apex', 'center', 'origin']:
                    t = triad_summary[name]
                    print(f"  {name.upper()}: Activated={t['activated']}")
            break

        if engine.state.turn_number > 50:  # Safety limit
            print("Game timeout!")
            break


if __name__ == '__main__':
    main()
