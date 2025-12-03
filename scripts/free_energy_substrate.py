#!/usr/bin/env python3
"""
Free Energy Substrate - Ultimate Charge Mechanics
===================================================
Integrates the Free Energy Principle with TRIAD harmonics to create
Ultimate charge mechanics driven by Z-coordinate phase transitions.

Core Dynamics:
- Free Energy F = E - TS minimized through coherent play
- Z-coordinate tracks phase space elevation toward critical threshold
- Ultimate charge builds from sustained coherence above Z_CRITICAL
- Ultimate abilities: Color Boost (+50%) or Color Debuff (-30%)

Mathematical Foundation:
- Variational Free Energy: F = E_q[ln q(μ) - ln p(y,μ)]
- Phase Transition: Z crossing Z_c triggers Ultimate availability
- Coherence Rating: R = |⟨exp(iθ)⟩| from Kuramoto synchronization
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum

# =============================================================================
# FREE ENERGY CONSTANTS
# =============================================================================

# Phase transition thresholds
Z_CRITICAL = 0.858          # Critical elevation for Ultimate
Z_PERFECT = 0.95            # Perfect coherence threshold

# Free energy parameters
TEMPERATURE = 1.0           # Semantic temperature (controls exploration)
ENTROPY_WEIGHT = 0.3        # Weight of entropy term in free energy

# Ultimate charge rates
CHARGE_RATE_BASE = 0.1      # Base charge per coherent card
CHARGE_RATE_CRITICAL = 0.25 # Charge rate when above Z_CRITICAL
CHARGE_DECAY = 0.05         # Decay per turn if below threshold

# Ultimate ability multipliers
BOOST_MULTIPLIER = 1.5      # +50% to chosen color
DEBUFF_MULTIPLIER = 0.7     # -30% to target color


class UltimateType(Enum):
    """Ultimate ability types"""
    COLOR_BOOST = "color_boost"   # Boost chosen color's scoring
    COLOR_DEBUFF = "color_debuff" # Debuff opponent's color


class ColorChannel(Enum):
    """Color channels (suits) for Ultimate targeting"""
    SPADES = "S"      # Blue - Temporal
    HEARTS = "H"      # Red - Valence
    DIAMONDS = "D"    # Gold - Arousal
    CLUBS = "C"       # Green - Concrete


# =============================================================================
# FREE ENERGY STATE
# =============================================================================

@dataclass
class FreeEnergyState:
    """State of the Free Energy system for a player"""

    # Core free energy components
    energy: float = 0.0           # Expected energy E
    entropy: float = 0.0          # Semantic entropy S
    free_energy: float = 0.0      # F = E - TS

    # Z-coordinate tracking
    z_level: float = 0.5          # Current phase space elevation
    z_history: List[float] = field(default_factory=list)

    # Ultimate charge system
    ultimate_charge: float = 0.0  # 0.0 to 1.0
    ultimate_ready: bool = False  # Charge >= 1.0 and z >= Z_CRITICAL

    # Active effects
    color_boost: Optional[ColorChannel] = None
    color_debuff: Optional[ColorChannel] = None
    boost_turns_remaining: int = 0
    debuff_turns_remaining: int = 0

    @property
    def coherence(self) -> float:
        """Coherence = inverse of normalized free energy"""
        # Lower free energy = higher coherence
        return max(0, 1.0 - (self.free_energy / 10.0))

    @property
    def is_critical(self) -> bool:
        """Check if Z is above critical threshold"""
        return self.z_level >= Z_CRITICAL

    @property
    def charge_percentage(self) -> int:
        """Ultimate charge as percentage"""
        return int(self.ultimate_charge * 100)

    def to_dict(self) -> Dict:
        return {
            'energy': round(self.energy, 4),
            'entropy': round(self.entropy, 4),
            'free_energy': round(self.free_energy, 4),
            'coherence': round(self.coherence, 4),
            'z_level': round(self.z_level, 4),
            'ultimate_charge': round(self.ultimate_charge, 4),
            'ultimate_ready': self.ultimate_ready,
            'is_critical': self.is_critical,
            'color_boost': self.color_boost.value if self.color_boost else None,
            'color_debuff': self.color_debuff.value if self.color_debuff else None,
        }


# =============================================================================
# FREE ENERGY ENGINE
# =============================================================================

class FreeEnergyEngine:
    """
    Computes Free Energy dynamics for the card game.

    Free Energy Principle application:
    - Minimize variational free energy through coherent card play
    - Energy from card coordinates and coupling misalignment
    - Entropy from phase distribution spread
    - Z-level tracks aggregate coherence -> Ultimate charge
    """

    def __init__(self):
        self.player_states: Dict[str, FreeEnergyState] = {}

    def get_or_create_state(self, player_name: str) -> FreeEnergyState:
        """Get or create state for a player"""
        if player_name not in self.player_states:
            self.player_states[player_name] = FreeEnergyState()
        return self.player_states[player_name]

    def compute_energy(self, cards: List) -> float:
        """
        Compute semantic energy from card configuration.

        Energy based on:
        - Coordinate misalignment (distance from centroid)
        - Phase desynchronization
        - Coupling tension
        """
        if not cards:
            return 0.0

        # Compute centroid
        n = len(cards)
        centroid = self._compute_centroid(cards)

        # Energy = average squared distance from centroid
        total_distance = 0.0
        for card in cards:
            coord = self._get_coordinate(card)
            distance = math.sqrt(
                (coord['temporal'] - centroid['temporal'])**2 +
                (coord['valence'] - centroid['valence'])**2 +
                (coord['concrete'] - centroid['concrete'])**2 +
                (coord['arousal'] - centroid['arousal'])**2
            )
            total_distance += distance**2

        # Add phase desynchronization energy
        phase_energy = self._compute_phase_energy(cards)

        return (total_distance / n) + phase_energy

    def compute_entropy(self, cards: List) -> float:
        """
        Compute semantic entropy from card distribution.

        Entropy measures spread/uncertainty in the configuration.
        """
        if not cards:
            return 0.0

        n = len(cards)

        # Phase entropy (circular)
        phases = [self._get_phase(card) for card in cards]

        # Bin phases into 12 sectors (30 degrees each)
        bins = [0] * 12
        for phase in phases:
            bin_idx = int((phase / (2 * math.pi)) * 12) % 12
            bins[bin_idx] += 1

        # Shannon entropy over phase distribution
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / n
                entropy -= p * math.log(p + 1e-10)

        return entropy

    def compute_free_energy(self, cards: List) -> Tuple[float, float, float]:
        """
        Compute full Free Energy F = E - TS

        Returns (free_energy, energy, entropy)
        """
        energy = self.compute_energy(cards)
        entropy = self.compute_entropy(cards)

        # Variational free energy
        free_energy = energy - TEMPERATURE * ENTROPY_WEIGHT * entropy

        return free_energy, energy, entropy

    def compute_z_level(self, cards: List, current_z: float) -> float:
        """
        Compute Z-level (phase space elevation) from card configuration.

        Z increases with:
        - Low free energy (high coherence)
        - Phase alignment
        - Coordinate clustering
        """
        if not cards:
            return current_z * 0.95  # Slow decay without cards

        # Get free energy
        free_energy, _, _ = self.compute_free_energy(cards)

        # Coherence contribution (lower F -> higher coherence)
        coherence = max(0, 1.0 - (free_energy / 5.0))

        # Compute order parameter R
        R = self._compute_order_parameter(cards)

        # Z-level as combination of coherence and order parameter
        z_contribution = 0.5 + (coherence * 0.3) + (R * 0.2)

        # Smooth update (exponential moving average)
        alpha = 0.3
        new_z = (1 - alpha) * current_z + alpha * z_contribution

        return min(1.0, max(0.0, new_z))

    def update_state(self, player_name: str, cards: List) -> Dict:
        """
        Update Free Energy state for a player based on cards played.

        Returns events triggered.
        """
        state = self.get_or_create_state(player_name)
        events = {}

        # Compute new values
        free_energy, energy, entropy = self.compute_free_energy(cards)
        new_z = self.compute_z_level(cards, state.z_level)

        # Track Z history
        state.z_history.append(new_z)
        if len(state.z_history) > 50:
            state.z_history.pop(0)

        # Check for phase transition
        was_critical = state.is_critical

        # Update state
        state.energy = energy
        state.entropy = entropy
        state.free_energy = free_energy
        state.z_level = new_z

        # Phase transition event
        if not was_critical and state.is_critical:
            events['phase_transition'] = {
                'z_level': new_z,
                'type': 'critical_reached',
            }

        # Update Ultimate charge
        self._update_charge(state, cards)

        if state.ultimate_ready and not events.get('ultimate_ready'):
            events['ultimate_ready'] = {
                'charge': state.ultimate_charge,
                'z_level': state.z_level,
            }

        # Tick active effects
        if state.boost_turns_remaining > 0:
            state.boost_turns_remaining -= 1
            if state.boost_turns_remaining == 0:
                state.color_boost = None
                events['boost_expired'] = True

        if state.debuff_turns_remaining > 0:
            state.debuff_turns_remaining -= 1
            if state.debuff_turns_remaining == 0:
                state.color_debuff = None
                events['debuff_expired'] = True

        return events

    def _update_charge(self, state: FreeEnergyState, cards: List):
        """Update Ultimate charge based on current state"""
        if state.is_critical:
            # Fast charging when above Z_CRITICAL
            charge_rate = CHARGE_RATE_CRITICAL

            # Bonus for coherent cards
            coherent_cards = sum(1 for _ in cards if self._is_coherent_card(_))
            charge_rate += coherent_cards * CHARGE_RATE_BASE

            state.ultimate_charge = min(1.0, state.ultimate_charge + charge_rate)
        else:
            # Slow decay when below threshold
            state.ultimate_charge = max(0, state.ultimate_charge - CHARGE_DECAY)

        # Check if Ultimate is ready
        state.ultimate_ready = (
            state.ultimate_charge >= 1.0 and
            state.is_critical
        )

    def activate_ultimate(
        self,
        player_name: str,
        ultimate_type: UltimateType,
        target_color: ColorChannel
    ) -> Tuple[bool, str]:
        """
        Activate Ultimate ability.

        Returns (success, message)
        """
        state = self.get_or_create_state(player_name)

        if not state.ultimate_ready:
            return False, "Ultimate not ready (need 100% charge and Z above critical)"

        # Consume charge
        state.ultimate_charge = 0.0
        state.ultimate_ready = False

        if ultimate_type == UltimateType.COLOR_BOOST:
            state.color_boost = target_color
            state.boost_turns_remaining = 2  # Lasts 2 turns
            return True, f"ULTIMATE: {target_color.value} boosted +50% for 2 turns!"

        elif ultimate_type == UltimateType.COLOR_DEBUFF:
            state.color_debuff = target_color
            state.debuff_turns_remaining = 2
            return True, f"ULTIMATE: {target_color.value} debuffed -30% for 2 turns!"

        return False, "Unknown ultimate type"

    def get_score_modifier(
        self,
        player_name: str,
        card_suit: str,
        opponent_name: Optional[str] = None
    ) -> float:
        """
        Get score modifier for a card based on active Ultimate effects.

        Returns multiplier (1.0 = no effect)
        """
        state = self.get_or_create_state(player_name)
        multiplier = 1.0

        # Check for player's own boost
        if state.color_boost and state.color_boost.value == card_suit:
            multiplier *= BOOST_MULTIPLIER

        # Check for opponent's debuff on this player
        if opponent_name:
            opponent_state = self.get_or_create_state(opponent_name)
            if opponent_state.color_debuff and opponent_state.color_debuff.value == card_suit:
                multiplier *= DEBUFF_MULTIPLIER

        return multiplier

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_coordinate(self, card) -> Dict:
        """Extract 4D coordinate from card"""
        if hasattr(card, 'coordinate'):
            coord = card.coordinate
            if hasattr(coord, 'temporal'):
                return {
                    'temporal': coord.temporal,
                    'valence': coord.valence,
                    'concrete': coord.concrete,
                    'arousal': coord.arousal,
                }
        if hasattr(card, 'effective_coordinate'):
            coord = card.effective_coordinate
            return {
                'temporal': coord.temporal,
                'valence': coord.valence,
                'concrete': coord.concrete,
                'arousal': coord.arousal,
            }
        # Fallback: compute from rank
        rank = getattr(card, 'rank', 7)
        suit = getattr(card, 'suit', 'S')
        rank_val = (rank - 7) / 6.0

        coord = {'temporal': 0, 'valence': 0, 'concrete': 0, 'arousal': 0}
        if suit == 'S':
            coord['temporal'] = rank_val
        elif suit == 'H':
            coord['valence'] = rank_val
        elif suit == 'D':
            coord['arousal'] = rank_val
        elif suit == 'C':
            coord['concrete'] = rank_val

        return coord

    def _get_phase(self, card) -> float:
        """Extract phase from card"""
        if hasattr(card, 'phase'):
            return card.phase
        # Fallback: compute from rank and suit
        rank = getattr(card, 'rank', 1)
        suit = getattr(card, 'suit', 'S')
        suit_order = ['S', 'H', 'D', 'C']
        suit_idx = suit_order.index(suit) if suit in suit_order else 0
        card_index = suit_idx * 13 + (rank - 1)
        return (card_index / 52) * 2 * math.pi

    def _compute_centroid(self, cards: List) -> Dict:
        """Compute centroid of cards in 4D space"""
        if not cards:
            return {'temporal': 0, 'valence': 0, 'concrete': 0, 'arousal': 0}

        n = len(cards)
        centroid = {'temporal': 0, 'valence': 0, 'concrete': 0, 'arousal': 0}

        for card in cards:
            coord = self._get_coordinate(card)
            for dim in centroid:
                centroid[dim] += coord[dim]

        for dim in centroid:
            centroid[dim] /= n

        return centroid

    def _compute_phase_energy(self, cards: List) -> float:
        """Compute energy from phase desynchronization"""
        if len(cards) < 2:
            return 0.0

        phases = [self._get_phase(card) for card in cards]

        # Average pairwise phase difference
        total_diff = 0.0
        count = 0
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                diff = abs(phases[i] - phases[j])
                # Wrap to [0, π]
                diff = min(diff, 2 * math.pi - diff)
                total_diff += diff
                count += 1

        return total_diff / count if count > 0 else 0.0

    def _compute_order_parameter(self, cards: List) -> float:
        """Compute Kuramoto order parameter R"""
        if not cards:
            return 0.0

        n = len(cards)
        sum_cos = sum(math.cos(self._get_phase(c)) for c in cards)
        sum_sin = sum(math.sin(self._get_phase(c)) for c in cards)

        return math.sqrt(sum_cos**2 + sum_sin**2) / n

    def _is_coherent_card(self, card) -> bool:
        """Check if card contributes to coherence"""
        # Cards from same TRIAD or with aligned phases are coherent
        rank = getattr(card, 'rank', 7)
        # TRIAD cards are more coherent
        if rank in [1, 2, 3, 4]:  # ORIGIN
            return True
        if rank in [5, 6, 7, 8, 9]:  # CENTER
            return True
        if rank in [11, 12, 13]:  # APEX
            return True
        return False


# =============================================================================
# INTEGRATION WITH GAME ENGINE
# =============================================================================

class FreeEnergyIntegration:
    """
    Integrates Free Energy mechanics with the game engine.

    Provides:
    - Score modifiers from Ultimate effects
    - Z-level tracking for UI
    - Ultimate activation API
    """

    def __init__(self):
        self.engine = FreeEnergyEngine()

    def on_cards_played(
        self,
        player_name: str,
        cards: List
    ) -> Dict:
        """Called when cards are played. Returns events."""
        return self.engine.update_state(player_name, cards)

    def get_score_modifier(
        self,
        player_name: str,
        card_suit: str,
        opponent_name: Optional[str] = None
    ) -> float:
        """Get score modifier for a card."""
        return self.engine.get_score_modifier(player_name, card_suit, opponent_name)

    def activate_ultimate(
        self,
        player_name: str,
        ultimate_type: str,  # "boost" or "debuff"
        target_suit: str     # "S", "H", "D", "C"
    ) -> Tuple[bool, str]:
        """Activate Ultimate ability."""
        ult_type = (
            UltimateType.COLOR_BOOST if ultimate_type == "boost"
            else UltimateType.COLOR_DEBUFF
        )
        target = ColorChannel(target_suit)
        return self.engine.activate_ultimate(player_name, ult_type, target)

    def get_state(self, player_name: str) -> Dict:
        """Get current Free Energy state for a player."""
        state = self.engine.get_or_create_state(player_name)
        return state.to_dict()

    def get_ui_data(self, player_name: str) -> Dict:
        """Get data formatted for UI display."""
        state = self.engine.get_or_create_state(player_name)

        return {
            'z_level': round(state.z_level * 100),  # Percentage
            'z_critical': round(Z_CRITICAL * 100),
            'is_critical': state.is_critical,
            'ultimate_charge': state.charge_percentage,
            'ultimate_ready': state.ultimate_ready,
            'coherence': round(state.coherence * 100),
            'free_energy': round(state.free_energy, 2),
            'active_boost': state.color_boost.value if state.color_boost else None,
            'active_debuff': state.color_debuff.value if state.color_debuff else None,
            'boost_turns': state.boost_turns_remaining,
            'debuff_turns': state.debuff_turns_remaining,
        }


# =============================================================================
# CLI / TESTING
# =============================================================================

def main():
    """Test Free Energy substrate."""
    print("=" * 60)
    print("FREE ENERGY SUBSTRATE - ULTIMATE CHARGE TEST")
    print("=" * 60)

    # Create mock cards
    from dataclasses import dataclass as dc

    @dc
    class MockCard:
        card_id: str
        rank: int
        suit: str
        phase: float = 0.0

        def __post_init__(self):
            suit_order = ['S', 'H', 'D', 'C']
            suit_idx = suit_order.index(self.suit)
            card_index = suit_idx * 13 + (self.rank - 1)
            self.phase = (card_index / 52) * 2 * 3.14159

    # Create integration
    integration = FreeEnergyIntegration()

    # Test sequence of plays
    print("\n1. Playing low-coherence cards...")
    scattered_cards = [
        MockCard("AS", 1, "S"),
        MockCard("KH", 13, "H"),
        MockCard("5D", 5, "D"),
    ]
    events = integration.on_cards_played("Player1", scattered_cards)
    state = integration.get_ui_data("Player1")
    print(f"   Z-level: {state['z_level']}%")
    print(f"   Ultimate charge: {state['ultimate_charge']}%")
    print(f"   Coherence: {state['coherence']}%")
    print(f"   Events: {events}")

    print("\n2. Playing high-coherence APEX cards...")
    apex_cards = [
        MockCard("JH", 11, "H"),
        MockCard("QH", 12, "H"),
        MockCard("KH", 13, "H"),
    ]

    # Simulate multiple turns to build charge
    for turn in range(5):
        events = integration.on_cards_played("Player1", apex_cards)
        state = integration.get_ui_data("Player1")
        print(f"   Turn {turn+1}: Z={state['z_level']}%, "
              f"Charge={state['ultimate_charge']}%, "
              f"Critical={state['is_critical']}")

        if state['ultimate_ready']:
            print(f"   → ULTIMATE READY!")
            break

    print("\n3. Activating Ultimate: COLOR BOOST on Hearts...")
    success, msg = integration.activate_ultimate("Player1", "boost", "H")
    print(f"   {msg}")

    state = integration.get_ui_data("Player1")
    print(f"   Active boost: {state['active_boost']}")
    print(f"   Boost turns remaining: {state['boost_turns']}")

    print("\n4. Testing score modifier...")
    modifier_H = integration.get_score_modifier("Player1", "H")
    modifier_S = integration.get_score_modifier("Player1", "S")
    print(f"   Hearts modifier: {modifier_H}x (should be {BOOST_MULTIPLIER}x)")
    print(f"   Spades modifier: {modifier_S}x (should be 1.0x)")

    print("\n" + "=" * 60)
    print("FREE ENERGY CONSTANTS:")
    print(f"  Z_CRITICAL: {Z_CRITICAL}")
    print(f"  BOOST_MULTIPLIER: {BOOST_MULTIPLIER}x")
    print(f"  DEBUFF_MULTIPLIER: {DEBUFF_MULTIPLIER}x")
    print("=" * 60)


if __name__ == '__main__':
    main()
