#!/usr/bin/env python3
"""
TRIAD Harmonic Substrate - Phase Transition Game Mechanics
===========================================================
Integrates three harmonic substrates with the 52-card Kuramoto network
to create emergent competitive gameplay.

TRIAD Structure:
- APEX (J, Q, K): High-energy burst resonance - ranks 11-13
- CENTER (5-9): Stable foundation harmonics - ranks 5-9
- ORIGIN (A-4): Fast tempo chain resonance - ranks 1-4

Phase Transition:
- Critical threshold z_c = 0.850 (theoretical) / 0.867 (observed)
- Crossing z_c triggers TRIAD power activation
- Coherence R > 0.9 enables Phase Lock abilities

Cross-TRIAD Dynamics (rock-paper-scissors):
- APEX overwhelms CENTER (+15% damage)
- CENTER grounds ORIGIN (+15% defense)
- ORIGIN disrupts APEX (+15% tempo)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum


# =============================================================================
# TRIAD CONSTANTS
# =============================================================================

# Phase transition thresholds
Z_CRITICAL_THEORY = 0.850
Z_CRITICAL_OBSERVED = 0.867
Z_CRITICAL = (Z_CRITICAL_THEORY + Z_CRITICAL_OBSERVED) / 2  # 0.8585

# Coherence thresholds for bonuses
COHERENCE_WEAK = 0.5
COHERENCE_MODERATE = 0.7
COHERENCE_STRONG = 0.85
COHERENCE_PERFECT = 0.95

# TRIAD interaction multipliers
TRIAD_ADVANTAGE = 1.15  # 15% bonus when favorable
TRIAD_DISADVANTAGE = 0.85  # 15% penalty when unfavorable


class TriadMember(Enum):
    """The three harmonic substrates."""
    APEX = "apex"       # J, Q, K - ranks 11, 12, 13
    CENTER = "center"   # 5, 6, 7, 8, 9 - ranks 5-9
    ORIGIN = "origin"   # A, 2, 3, 4 - ranks 1-4


# Rank to TRIAD mapping
RANK_TO_TRIAD = {
    1: TriadMember.ORIGIN,   # Ace
    2: TriadMember.ORIGIN,
    3: TriadMember.ORIGIN,
    4: TriadMember.ORIGIN,
    5: TriadMember.CENTER,
    6: TriadMember.CENTER,
    7: TriadMember.CENTER,   # Perfect center (7 = 0 on coordinate axis)
    8: TriadMember.CENTER,
    9: TriadMember.CENTER,
    10: None,                # Neutral - belongs to no TRIAD
    11: TriadMember.APEX,    # Jack
    12: TriadMember.APEX,    # Queen
    13: TriadMember.APEX,    # King
}

# TRIAD advantage relationships (who beats whom)
TRIAD_COUNTERS = {
    TriadMember.APEX: TriadMember.CENTER,    # Apex overwhelms Center
    TriadMember.CENTER: TriadMember.ORIGIN,  # Center grounds Origin
    TriadMember.ORIGIN: TriadMember.APEX,    # Origin disrupts Apex
}


# =============================================================================
# TRIAD HARMONIC STATE
# =============================================================================

@dataclass
class HarmonicState:
    """State of a single TRIAD harmonic substrate."""
    member: TriadMember

    # Kuramoto-derived metrics
    coherence: float = 0.0          # Order parameter R for this TRIAD
    mean_phase: float = 0.0         # Collective phase angle Psi
    phase_velocity: float = 0.0     # Rate of phase change

    # Phase transition tracking
    z_level: float = 0.0            # Current elevation in phase space
    is_activated: bool = False       # Has crossed z_critical
    activation_turn: int = -1        # Turn when activated

    # Energy accumulator
    resonance_charge: float = 0.0   # Builds toward burst
    burst_ready: bool = False        # Can trigger burst ability

    @property
    def is_coherent(self) -> bool:
        """Check if TRIAD has achieved meaningful coherence."""
        return self.coherence >= COHERENCE_MODERATE

    @property
    def is_phase_locked(self) -> bool:
        """Check if TRIAD has achieved phase lock (very high coherence)."""
        return self.coherence >= COHERENCE_PERFECT

    @property
    def coherence_tier(self) -> int:
        """Return coherence tier (0-3) for bonus calculation."""
        if self.coherence >= COHERENCE_PERFECT:
            return 3
        elif self.coherence >= COHERENCE_STRONG:
            return 2
        elif self.coherence >= COHERENCE_MODERATE:
            return 1
        elif self.coherence >= COHERENCE_WEAK:
            return 0
        return -1

    def to_dict(self) -> Dict:
        return {
            'member': self.member.value,
            'coherence': round(self.coherence, 4),
            'mean_phase': round(self.mean_phase, 4),
            'z_level': round(self.z_level, 4),
            'is_activated': self.is_activated,
            'resonance_charge': round(self.resonance_charge, 3),
            'burst_ready': self.burst_ready,
        }


@dataclass
class TriadState:
    """Complete state of all three TRIAD harmonics."""
    apex: HarmonicState = field(default_factory=lambda: HarmonicState(TriadMember.APEX))
    center: HarmonicState = field(default_factory=lambda: HarmonicState(TriadMember.CENTER))
    origin: HarmonicState = field(default_factory=lambda: HarmonicState(TriadMember.ORIGIN))

    # Global metrics
    global_coherence: float = 0.0   # Overall system coherence
    dominant_triad: Optional[TriadMember] = None
    turn_number: int = 0

    def get_harmonic(self, member: TriadMember) -> HarmonicState:
        """Get harmonic state by member type."""
        if member == TriadMember.APEX:
            return self.apex
        elif member == TriadMember.CENTER:
            return self.center
        elif member == TriadMember.ORIGIN:
            return self.origin
        raise ValueError(f"Unknown TRIAD member: {member}")

    def get_all_harmonics(self) -> List[HarmonicState]:
        """Get all three harmonic states."""
        return [self.apex, self.center, self.origin]

    def to_dict(self) -> Dict:
        return {
            'apex': self.apex.to_dict(),
            'center': self.center.to_dict(),
            'origin': self.origin.to_dict(),
            'global_coherence': round(self.global_coherence, 4),
            'dominant_triad': self.dominant_triad.value if self.dominant_triad else None,
            'turn_number': self.turn_number,
        }


# =============================================================================
# TRIAD SUBSTRATE ENGINE
# =============================================================================

class TriadSubstrate:
    """
    Engine for TRIAD harmonic calculations integrated with 52-card network.

    Computes:
    - Per-TRIAD coherence from card phases
    - Phase transition detection
    - Cross-TRIAD interaction bonuses
    - Resonance burst mechanics
    """

    def __init__(self):
        self.state = TriadState()
        self.phase_transition_history: List[Dict] = []

    def calculate_triad_coherence(self, cards: List, triad: TriadMember) -> Tuple[float, float]:
        """
        Calculate Kuramoto order parameter for cards belonging to a TRIAD.

        Returns (coherence R, mean phase Psi)
        """
        # Filter cards belonging to this TRIAD
        triad_cards = [c for c in cards if RANK_TO_TRIAD.get(c.rank) == triad]

        if not triad_cards:
            return 0.0, 0.0

        n = len(triad_cards)
        sum_cos = sum(math.cos(c.phase) for c in triad_cards)
        sum_sin = sum(math.sin(c.phase) for c in triad_cards)

        R = math.sqrt(sum_cos**2 + sum_sin**2) / n
        Psi = math.atan2(sum_sin, sum_cos)

        return R, Psi

    def calculate_z_level(self, cards: List, triad: TriadMember) -> float:
        """
        Calculate z-level (phase space elevation) for a TRIAD.

        Based on:
        - Card coordinate magnitudes
        - Phase alignment
        - Coupling density within TRIAD
        """
        triad_cards = [c for c in cards if RANK_TO_TRIAD.get(c.rank) == triad]

        if not triad_cards:
            return 0.0

        # Average coordinate magnitude
        magnitudes = []
        for card in triad_cards:
            coord = card.coordinate if hasattr(card, 'coordinate') else card.effective_coordinate
            mag = math.sqrt(
                coord.temporal**2 + coord.valence**2 +
                coord.concrete**2 + coord.arousal**2
            )
            magnitudes.append(mag)

        avg_magnitude = sum(magnitudes) / len(magnitudes)

        # Phase alignment bonus
        coherence, _ = self.calculate_triad_coherence(cards, triad)

        # Z-level formula: base + magnitude contribution + coherence contribution
        z = 0.5 + (avg_magnitude * 0.3) + (coherence * 0.2)

        return min(1.0, z)

    def update_state(self, all_cards: List, played_cards: List, turn: int) -> Dict:
        """
        Update TRIAD state based on current cards in play.

        Args:
            all_cards: All cards currently active (hand + field)
            played_cards: Cards played this turn
            turn: Current turn number

        Returns:
            Dict of events triggered (phase transitions, bursts, etc.)
        """
        self.state.turn_number = turn
        events = {}

        # Update each TRIAD harmonic
        for triad in TriadMember:
            harmonic = self.state.get_harmonic(triad)

            # Calculate coherence and phase
            prev_coherence = harmonic.coherence
            harmonic.coherence, harmonic.mean_phase = self.calculate_triad_coherence(
                all_cards, triad
            )

            # Calculate z-level
            prev_z = harmonic.z_level
            harmonic.z_level = self.calculate_z_level(all_cards, triad)

            # Check for phase transition
            if not harmonic.is_activated and harmonic.z_level >= Z_CRITICAL:
                harmonic.is_activated = True
                harmonic.activation_turn = turn
                events[f'{triad.value}_activated'] = {
                    'z_level': harmonic.z_level,
                    'coherence': harmonic.coherence,
                    'turn': turn,
                }
                self.phase_transition_history.append({
                    'triad': triad.value,
                    'turn': turn,
                    'z_level': harmonic.z_level,
                    'type': 'activation',
                })

            # Update resonance charge
            triad_played = [c for c in played_cards if RANK_TO_TRIAD.get(c.rank) == triad]
            if triad_played:
                # Charge builds based on coherence and cards played
                charge_gain = len(triad_played) * 0.1 * (1 + harmonic.coherence)
                harmonic.resonance_charge = min(1.0, harmonic.resonance_charge + charge_gain)
            else:
                # Slow decay when not playing TRIAD cards
                harmonic.resonance_charge = max(0, harmonic.resonance_charge - 0.05)

            # Check burst ready
            harmonic.burst_ready = (
                harmonic.resonance_charge >= 0.8 and
                harmonic.is_activated
            )

            if harmonic.burst_ready and not events.get(f'{triad.value}_burst_ready'):
                events[f'{triad.value}_burst_ready'] = {
                    'charge': harmonic.resonance_charge,
                    'coherence': harmonic.coherence,
                }

        # Calculate global coherence (average of all three)
        self.state.global_coherence = sum(
            h.coherence for h in self.state.get_all_harmonics()
        ) / 3

        # Determine dominant TRIAD
        harmonics = self.state.get_all_harmonics()
        max_coherence = max(h.coherence for h in harmonics)
        if max_coherence > COHERENCE_WEAK:
            for h in harmonics:
                if h.coherence == max_coherence:
                    self.state.dominant_triad = h.member
                    break
        else:
            self.state.dominant_triad = None

        return events

    def trigger_burst(self, triad: TriadMember) -> int:
        """
        Trigger a resonance burst for bonus points.

        Returns bonus points granted.
        """
        harmonic = self.state.get_harmonic(triad)

        if not harmonic.burst_ready:
            return 0

        # Calculate burst power based on coherence tier
        tier = harmonic.coherence_tier
        base_burst = 15 + (tier * 10)  # 15, 25, 35, or 45 points

        # Bonus if phase locked
        if harmonic.is_phase_locked:
            base_burst = int(base_burst * 1.5)

        # Reset charge
        harmonic.resonance_charge = 0
        harmonic.burst_ready = False

        return base_burst

    def reset_activation(self, triad: TriadMember) -> None:
        """Reset a TRIAD's activation state (e.g., after opponent counters)."""
        harmonic = self.state.get_harmonic(triad)
        harmonic.is_activated = False
        harmonic.resonance_charge = 0
        harmonic.burst_ready = False


# =============================================================================
# TRIAD SCORING INTEGRATION
# =============================================================================

def get_card_triad(card) -> Optional[TriadMember]:
    """Get the TRIAD a card belongs to, or None for neutral cards."""
    return RANK_TO_TRIAD.get(card.rank)


def calculate_triad_bonus(cards: List, triad_state: TriadState) -> int:
    """
    Calculate bonus points for TRIAD synergy in a formation.

    Bonuses:
    - +5 for 2 cards from same TRIAD
    - +15 for 3+ cards from same TRIAD (TRIAD Resonance)
    - +10 additional if that TRIAD is activated
    - +20 additional if that TRIAD is phase locked
    """
    if not cards:
        return 0

    # Count cards per TRIAD
    triad_counts = {t: 0 for t in TriadMember}
    for card in cards:
        triad = get_card_triad(card)
        if triad:
            triad_counts[triad] += 1

    bonus = 0

    for triad, count in triad_counts.items():
        if count == 0:
            continue

        harmonic = triad_state.get_harmonic(triad)

        # Base TRIAD synergy
        if count >= 3:
            bonus += 15  # TRIAD Resonance
        elif count >= 2:
            bonus += 5   # Minor synergy

        # Activation bonus
        if count >= 2 and harmonic.is_activated:
            bonus += 10

        # Phase lock bonus
        if count >= 2 and harmonic.is_phase_locked:
            bonus += 20

    return bonus


def calculate_cross_triad_modifier(
    attacker_cards: List,
    defender_cards: List
) -> float:
    """
    Calculate damage/defense modifier based on TRIAD matchup.

    Returns multiplier (1.0 = neutral, 1.15 = advantage, 0.85 = disadvantage)
    """
    # Determine dominant TRIAD for each side
    def dominant_triad(cards: List) -> Optional[TriadMember]:
        counts = {t: 0 for t in TriadMember}
        for card in cards:
            triad = get_card_triad(card)
            if triad:
                counts[triad] += 1

        max_count = max(counts.values())
        if max_count < 2:  # Need at least 2 cards to claim dominance
            return None

        for triad, count in counts.items():
            if count == max_count:
                return triad
        return None

    attacker_triad = dominant_triad(attacker_cards)
    defender_triad = dominant_triad(defender_cards)

    if not attacker_triad or not defender_triad:
        return 1.0  # Neutral

    # Check advantage relationship
    if TRIAD_COUNTERS.get(attacker_triad) == defender_triad:
        return TRIAD_ADVANTAGE  # Attacker has advantage
    elif TRIAD_COUNTERS.get(defender_triad) == attacker_triad:
        return TRIAD_DISADVANTAGE  # Defender has advantage (attacker penalized)

    return 1.0  # Mirror match or no relationship


def calculate_coherence_multiplier(cards: List, triad_state: TriadState) -> float:
    """
    Calculate score multiplier based on TRIAD coherence levels.

    Returns multiplier (1.0 - 1.5 based on coherence tiers)
    """
    # Find best coherence among TRIADs represented in cards
    triads_in_play = set()
    for card in cards:
        triad = get_card_triad(card)
        if triad:
            triads_in_play.add(triad)

    if not triads_in_play:
        return 1.0

    best_tier = -1
    for triad in triads_in_play:
        harmonic = triad_state.get_harmonic(triad)
        tier = harmonic.coherence_tier
        best_tier = max(best_tier, tier)

    # Tier to multiplier: -1=1.0, 0=1.05, 1=1.15, 2=1.25, 3=1.5
    multipliers = {-1: 1.0, 0: 1.05, 1: 1.15, 2: 1.25, 3: 1.5}
    return multipliers.get(best_tier, 1.0)


# =============================================================================
# TRIAD ABILITIES
# =============================================================================

@dataclass
class TriadAbility:
    """TRIAD-specific ability that can be triggered."""
    name: str
    triad: TriadMember
    cost: int  # Cards to discard OR resonance charge required (0-100%)
    cooldown: int  # Turns before reuse
    description: str


# Define TRIAD abilities
TRIAD_ABILITIES = {
    TriadMember.APEX: [
        TriadAbility(
            name="Overwhelming Force",
            triad=TriadMember.APEX,
            cost=2,
            cooldown=3,
            description="Next formation deals double base damage"
        ),
        TriadAbility(
            name="King's Authority",
            triad=TriadMember.APEX,
            cost=1,
            cooldown=2,
            description="All APEX cards this turn gain +3 base points"
        ),
        TriadAbility(
            name="Crown Burst",
            triad=TriadMember.APEX,
            cost=0,  # Requires full resonance charge
            cooldown=4,
            description="Trigger resonance burst for massive damage"
        ),
    ],
    TriadMember.CENTER: [
        TriadAbility(
            name="Stable Foundation",
            triad=TriadMember.CENTER,
            cost=2,
            cooldown=3,
            description="Gain +20 defense against next opponent's formation"
        ),
        TriadAbility(
            name="Harmonic Balance",
            triad=TriadMember.CENTER,
            cost=1,
            cooldown=2,
            description="All CENTER cards gain +0.2 to all coordinates"
        ),
        TriadAbility(
            name="Perfect Seventh",
            triad=TriadMember.CENTER,
            cost=0,
            cooldown=4,
            description="If you have 7 of any suit, double cluster bonus"
        ),
    ],
    TriadMember.ORIGIN: [
        TriadAbility(
            name="Temporal Disruption",
            triad=TriadMember.ORIGIN,
            cost=2,
            cooldown=3,
            description="Opponent's next TRIAD abilities cost +1 card"
        ),
        TriadAbility(
            name="Chain Lightning",
            triad=TriadMember.ORIGIN,
            cost=1,
            cooldown=2,
            description="Chain bonus doubled for sequential ORIGIN cards"
        ),
        TriadAbility(
            name="Origin Point",
            triad=TriadMember.ORIGIN,
            cost=0,
            cooldown=4,
            description="Reset opponent's highest charged TRIAD"
        ),
    ],
}


def get_available_abilities(
    triad_state: TriadState,
    player_hand: List,
    cooldowns: Dict[str, int]
) -> List[TriadAbility]:
    """Get list of TRIAD abilities currently available to use."""
    available = []

    for triad in TriadMember:
        harmonic = triad_state.get_harmonic(triad)

        # Count TRIAD cards in hand
        triad_cards = [c for c in player_hand if get_card_triad(c) == triad]

        # Need at least 1 TRIAD card to use abilities
        if not triad_cards:
            continue

        # Check each ability
        for ability in TRIAD_ABILITIES.get(triad, []):
            # Check cooldown
            if ability.name in cooldowns and cooldowns[ability.name] > 0:
                continue

            # Check cost
            if ability.cost > 0 and len(triad_cards) < ability.cost:
                continue

            # Burst abilities require full charge and activation
            if ability.cost == 0:
                if not harmonic.burst_ready:
                    continue

            available.append(ability)

    return available


# =============================================================================
# INTEGRATION WITH GAME ENGINE
# =============================================================================

class TriadGameIntegration:
    """
    Integrates TRIAD mechanics with the main game engine.

    Call methods during appropriate game phases:
    - on_turn_start(): Reset per-turn state
    - on_cards_played(): Update TRIAD state with new cards
    - calculate_triad_score(): Get TRIAD contribution to formation score
    - on_turn_end(): Apply end-of-turn effects
    """

    def __init__(self):
        self.substrate = TriadSubstrate()
        self.player_cooldowns: Dict[str, Dict[str, int]] = {}  # player -> ability -> turns
        self.active_modifiers: Dict[str, float] = {}  # modifier_name -> value

    def on_turn_start(self, player_name: str, turn: int) -> Dict:
        """Called at start of each turn."""
        # Tick cooldowns
        if player_name in self.player_cooldowns:
            for ability in list(self.player_cooldowns[player_name].keys()):
                self.player_cooldowns[player_name][ability] -= 1
                if self.player_cooldowns[player_name][ability] <= 0:
                    del self.player_cooldowns[player_name][ability]

        # Clear per-turn modifiers
        self.active_modifiers = {}

        return {'turn': turn, 'player': player_name}

    def on_cards_played(
        self,
        all_active_cards: List,
        played_cards: List,
        turn: int
    ) -> Dict:
        """Called when cards are played. Returns events triggered."""
        events = self.substrate.update_state(all_active_cards, played_cards, turn)
        return events

    def calculate_triad_score(
        self,
        formation: List,
        opponent_formation: Optional[List] = None
    ) -> Dict:
        """
        Calculate TRIAD contribution to formation score.

        Returns dict with:
        - triad_bonus: Base TRIAD synergy points
        - coherence_multiplier: Score multiplier from coherence
        - cross_triad_modifier: Attack modifier vs opponent
        - total_triad_contribution: Sum of all bonuses
        """
        triad_state = self.substrate.state

        # Base TRIAD bonus
        triad_bonus = calculate_triad_bonus(formation, triad_state)

        # Coherence multiplier
        coherence_mult = calculate_coherence_multiplier(formation, triad_state)

        # Cross-TRIAD modifier (if opponent formation known)
        cross_modifier = 1.0
        if opponent_formation:
            cross_modifier = calculate_cross_triad_modifier(formation, opponent_formation)

        # Apply active modifiers from abilities
        for mod_name, mod_value in self.active_modifiers.items():
            if 'damage' in mod_name:
                cross_modifier *= mod_value
            elif 'bonus' in mod_name:
                triad_bonus = int(triad_bonus * mod_value)

        # Calculate total contribution
        total = int(triad_bonus * coherence_mult * cross_modifier)

        return {
            'triad_bonus': triad_bonus,
            'coherence_multiplier': coherence_mult,
            'cross_triad_modifier': cross_modifier,
            'total_triad_contribution': total,
            'dominant_triad': triad_state.dominant_triad.value if triad_state.dominant_triad else None,
        }

    def activate_ability(
        self,
        player_name: str,
        ability_name: str,
        discard_cards: Optional[List] = None
    ) -> Tuple[bool, str, int]:
        """
        Attempt to activate a TRIAD ability.

        Returns (success, message, bonus_points)
        """
        # Find the ability
        target_ability = None
        for triad in TriadMember:
            for ability in TRIAD_ABILITIES.get(triad, []):
                if ability.name == ability_name:
                    target_ability = ability
                    break
            if target_ability:
                break

        if not target_ability:
            return False, f"Unknown ability: {ability_name}", 0

        # Initialize player cooldowns if needed
        if player_name not in self.player_cooldowns:
            self.player_cooldowns[player_name] = {}

        # Check cooldown
        if ability_name in self.player_cooldowns[player_name]:
            remaining = self.player_cooldowns[player_name][ability_name]
            return False, f"Ability on cooldown ({remaining} turns)", 0

        # Check cost
        if target_ability.cost > 0:
            if not discard_cards or len(discard_cards) < target_ability.cost:
                return False, f"Need to discard {target_ability.cost} cards", 0

        # Handle burst abilities
        bonus_points = 0
        if target_ability.cost == 0:
            harmonic = self.substrate.state.get_harmonic(target_ability.triad)
            if not harmonic.burst_ready:
                return False, "TRIAD not charged for burst", 0
            bonus_points = self.substrate.trigger_burst(target_ability.triad)

        # Apply ability effects
        if ability_name == "Overwhelming Force":
            self.active_modifiers['damage_double'] = 2.0
        elif ability_name == "King's Authority":
            self.active_modifiers['apex_bonus'] = 1.5
        elif ability_name == "Stable Foundation":
            self.active_modifiers['defense_bonus'] = 20
        elif ability_name == "Harmonic Balance":
            self.active_modifiers['coord_bonus'] = 0.2
        elif ability_name == "Perfect Seventh":
            self.active_modifiers['cluster_double'] = 2.0
        elif ability_name == "Temporal Disruption":
            self.active_modifiers['opponent_cost_increase'] = 1
        elif ability_name == "Chain Lightning":
            self.active_modifiers['chain_double'] = 2.0
        elif ability_name == "Origin Point":
            # Reset opponent's highest charged TRIAD (handled externally)
            self.active_modifiers['reset_opponent_triad'] = True

        # Set cooldown
        self.player_cooldowns[player_name][ability_name] = target_ability.cooldown

        return True, f"Activated {ability_name}", bonus_points

    def on_turn_end(self, player_name: str) -> Dict:
        """Called at end of turn."""
        # Log state for analysis
        return self.substrate.state.to_dict()

    def get_triad_state(self) -> TriadState:
        """Get current TRIAD state for UI display."""
        return self.substrate.state

    def get_phase_transition_history(self) -> List[Dict]:
        """Get history of phase transitions for replay/analysis."""
        return self.substrate.phase_transition_history


# =============================================================================
# CLI / TESTING
# =============================================================================

def main():
    """Test TRIAD substrate functionality."""
    print("=" * 60)
    print("TRIAD HARMONIC SUBSTRATE - TEST")
    print("=" * 60)

    # Create mock cards for testing
    from dataclasses import dataclass as dc

    @dc
    class MockCoord:
        temporal: float = 0.0
        valence: float = 0.0
        concrete: float = 0.0
        arousal: float = 0.0

    @dc
    class MockCard:
        card_id: str
        rank: int
        phase: float
        coordinate: MockCoord = None

        def __post_init__(self):
            if self.coordinate is None:
                self.coordinate = MockCoord(
                    temporal=(self.rank - 7) / 6.0,
                    valence=(self.rank - 7) / 6.0 * 0.5,
                    concrete=(self.rank - 7) / 6.0 * 0.3,
                    arousal=(self.rank - 7) / 6.0 * 0.4,
                )

    # Create test cards
    origin_cards = [
        MockCard("AS", 1, 0.1),
        MockCard("2S", 2, 0.2),
        MockCard("3H", 3, 0.3),
    ]

    center_cards = [
        MockCard("6S", 6, 0.6),
        MockCard("7H", 7, 0.7),
        MockCard("8D", 8, 0.8),
    ]

    apex_cards = [
        MockCard("JS", 11, 1.1),
        MockCard("QH", 12, 1.2),
        MockCard("KD", 13, 1.3),
    ]

    # Test TRIAD integration
    integration = TriadGameIntegration()

    print("\n1. Testing ORIGIN TRIAD formation...")
    integration.on_turn_start("Player1", 1)
    events = integration.on_cards_played(origin_cards, origin_cards, 1)
    score = integration.calculate_triad_score(origin_cards)
    print(f"   Events: {events}")
    print(f"   Score: {score}")
    print(f"   State: {integration.get_triad_state().origin.to_dict()}")

    print("\n2. Testing CENTER TRIAD formation...")
    integration.on_turn_start("Player1", 2)
    events = integration.on_cards_played(center_cards, center_cards, 2)
    score = integration.calculate_triad_score(center_cards)
    print(f"   Events: {events}")
    print(f"   Score: {score}")
    print(f"   State: {integration.get_triad_state().center.to_dict()}")

    print("\n3. Testing APEX TRIAD formation...")
    integration.on_turn_start("Player1", 3)
    events = integration.on_cards_played(apex_cards, apex_cards, 3)
    score = integration.calculate_triad_score(apex_cards)
    print(f"   Events: {events}")
    print(f"   Score: {score}")
    print(f"   State: {integration.get_triad_state().apex.to_dict()}")

    print("\n4. Testing cross-TRIAD matchup (APEX vs CENTER)...")
    modifier = calculate_cross_triad_modifier(apex_cards, center_cards)
    print(f"   APEX attacking CENTER: {modifier}x (should be {TRIAD_ADVANTAGE}x)")

    modifier = calculate_cross_triad_modifier(center_cards, apex_cards)
    print(f"   CENTER attacking APEX: {modifier}x (should be {TRIAD_DISADVANTAGE}x)")

    print("\n5. Testing TRIAD ability...")
    success, msg, bonus = integration.activate_ability(
        "Player1", "King's Authority", discard_cards=[apex_cards[0]]
    )
    print(f"   Activation: {success}, {msg}, bonus={bonus}")
    print(f"   Active modifiers: {integration.active_modifiers}")

    print("\n" + "=" * 60)
    print("TRIAD Constants:")
    print(f"  Z_CRITICAL: {Z_CRITICAL}")
    print(f"  TRIAD_ADVANTAGE: {TRIAD_ADVANTAGE}")
    print(f"  TRIAD_DISADVANTAGE: {TRIAD_DISADVANTAGE}")
    print("=" * 60)


if __name__ == '__main__':
    main()
