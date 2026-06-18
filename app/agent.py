"""
Q-Learning Agent for Sate Keliling Route Optimization.
Implements the QLearningAgent, QLearningEnvironment, and QTable from the Class Diagram.

State: (lokasi_saat_ini, kondisi_cuaca, hari_kuliah)
Action: Composite — (lokasi_tujuan, durasi_mangkal_discrete)
Reward: Penjualan per lokasi (step-based, bukan total harian)
Update Rule: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
"""

import random
import json
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Lokasi, Penjualan, Kunjungan


# =========================================================
# DURASI DISCRETIZATION
# =========================================================

DURASI_VALUES = [30, 60, 120, 240]

DURASI_LABELS = {
    30: "≤ 45 menit",
    60: "46–90 menit",
    120: "91–150 menit",
    240: "> 150 menit",
}

# Batas minimum dan maksimum durasi mangkal (menit)
DURASI_MIN = 30
DURASI_MAX = 240


def discretize_durasi(durasi_menit: float) -> int:
    """
    Discretize durasi mangkal ke bin terdekat.

    Rules:
        ≤ 45  menit → 30
        46–90 menit → 60
        91–150 menit → 120
        > 150 menit → 240

    Durasi di-clamp ke [DURASI_MIN, DURASI_MAX] sebelum discretization.

    Args:
        durasi_menit: Durasi aktual dalam menit (float)

    Returns:
        Discretized duration value (30, 60, 120, atau 240)
    """
    # Clamp ke batas min/max
    durasi_menit = max(DURASI_MIN, min(DURASI_MAX, durasi_menit))

    if durasi_menit <= 45:
        return 30
    elif durasi_menit <= 90:
        return 60
    elif durasi_menit <= 150:
        return 120
    else:
        return 240


# =========================================================
# STATE
# =========================================================

@dataclass(frozen=True)
class State:
    """
    State representation for Q-Learning.

    Attributes:
        lokasi_saat_ini: Current location ID
        kondisi_cuaca: Weather condition ("cerah" | "mendung" | "hujan")
        hari_kuliah: 1 = hari kuliah aktif, 0 = akhir pekan / libur
    """
    lokasi_saat_ini: int
    kondisi_cuaca: str
    hari_kuliah: int

    def to_key(self) -> str:
        """Convert state to a unique string key for Q-table lookup."""
        return json.dumps({
            "lokasi": self.lokasi_saat_ini,
            "cuaca": self.kondisi_cuaca,
            "hari": self.hari_kuliah
        }, sort_keys=True)

    @classmethod
    def from_key(cls, key: str) -> "State":
        """Create state from a string key."""
        data = json.loads(key)
        return cls(
            lokasi_saat_ini=data["lokasi"],
            kondisi_cuaca=data["cuaca"],
            hari_kuliah=data["hari"]
        )


# =========================================================
# COMPOSITE ACTION
# =========================================================

@dataclass(frozen=True)
class CompositeAction:
    """
    Composite action: memilih lokasi tujuan + durasi mangkal.

    Attributes:
        lokasi_tujuan: ID lokasi yang dipilih sebagai tujuan berikutnya
        durasi_mangkal: Discretized duration (10, 20, 30, atau 60 menit)
    """
    lokasi_tujuan: int
    durasi_mangkal: int  # 10, 20, 30, atau 60


# =========================================================
# Q-TABLE (Dict-based, supports DB persistence)
# =========================================================

class QTable:
    """
    Q-Table for storing Q-values.
    Supports both in-memory and database-backed storage.

    Key format: state.to_key() → str
    Action key: action_to_index(action) → int
    """

    def __init__(self, db: Session = None, pedagang_id: int = None, db_manager=None):
        """Initialize Q-Table."""
        self.q_values: Dict[str, Dict[int, float]] = {}
        self.db = db
        self.pedagang_id = pedagang_id
        self.db_manager = db_manager

        # Load existing Q-values from database if available
        if db and pedagang_id and db_manager:
            self._load_from_db()

    def _load_from_db(self):
        """Load Q-values from database."""
        entries = self.db_manager.get_all_q_values(self.db, self.pedagang_id)
        for entry in entries:
            if entry.state_key not in self.q_values:
                self.q_values[entry.state_key] = {}
            self.q_values[entry.state_key][entry.action] = entry.q_value

    def get_q_value(self, state: State, action_index: int) -> float:
        """
        Get Q-value for a state-action pair.
        Returns 0.0 if not found (optimistic initialization).
        """
        state_key = state.to_key()
        if state_key in self.q_values:
            return self.q_values[state_key].get(action_index, 0.0)
        return 0.0

    def update_q_value(self, state: State, action_index: int, value: float):
        """Update Q-value for a state-action pair."""
        state_key = state.to_key()
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        self.q_values[state_key][action_index] = value

        # Persist to database if available
        if self.db and self.pedagang_id and self.db_manager:
            self.db_manager.update_q_value(
                self.db, self.pedagang_id, state_key, action_index, value
            )

    def get_max_q_value(self, state: State, num_actions: int) -> float:
        """Get the maximum Q-value for any action in a given state."""
        state_key = state.to_key()
        if state_key not in self.q_values:
            return 0.0

        values = self.q_values[state_key]
        if not values:
            return 0.0

        return max(values.values())


# =========================================================
# ACTION SPACE HELPER
# =========================================================

class ActionSpace:
    """
    Manages the composite action space: (lokasi_tujuan, durasi_mangkal).
    Maps between CompositeAction and integer indices for Q-table.
    """

    def __init__(self, lokasi_ids: List[int]):
        """
        Build action space from available location IDs.

        Total actions = len(lokasi_ids) × len(DURASI_VALUES)
        """
        self.lokasi_ids = lokasi_ids
        self.actions: List[CompositeAction] = []
        self._index_map: Dict[Tuple[int, int], int] = {}

        for lok_id in lokasi_ids:
            for dur in DURASI_VALUES:
                idx = len(self.actions)
                action = CompositeAction(lokasi_tujuan=lok_id, durasi_mangkal=dur)
                self.actions.append(action)
                self._index_map[(lok_id, dur)] = idx

    @property
    def size(self) -> int:
        return len(self.actions)

    def action_to_index(self, action: CompositeAction) -> int:
        """Convert CompositeAction to integer index."""
        return self._index_map.get(
            (action.lokasi_tujuan, action.durasi_mangkal), 0
        )

    def index_to_action(self, index: int) -> CompositeAction:
        """Convert integer index back to CompositeAction."""
        if 0 <= index < len(self.actions):
            return self.actions[index]
        return self.actions[0]

    def get_actions_for_lokasi(self, lokasi_id: int) -> List[int]:
        """Get all action indices for a specific location."""
        return [
            self._index_map[(lokasi_id, dur)]
            for dur in DURASI_VALUES
            if (lokasi_id, dur) in self._index_map
        ]


# =========================================================
# Q-LEARNING ENVIRONMENT
# =========================================================

class QLearningEnvironment:
    """
    Environment for Q-Learning training.
    Implements: getInitialState(), executeAction(), calculateReward(), isTerminalState()

    Reward dihitung dari data penjualan historis PER LOKASI (bukan total harian).
    """

    def __init__(
        self,
        lokasi_list: List[Lokasi],
        kunjungan_data: List[Kunjungan],
        action_space: ActionSpace,
        waktu_operasional: int = None,
        kondisi_cuaca: str = "cerah",
        hari_kuliah: int = 1,
        start_lokasi_id: Optional[int] = None
    ):
        """
        Initialize the environment.

        Args:
            lokasi_list: List of locations available
            kunjungan_data: Historical visit data for reward calculation
            action_space: ActionSpace instance
            waktu_operasional: Total operational time in minutes
            kondisi_cuaca: Weather condition (cerah, mendung, hujan)
            hari_kuliah: 1 if weekday/school day, 0 if weekend/holiday
        """
        self.lokasi_list = lokasi_list
        self.lokasi_ids = [l.id for l in lokasi_list]
        self.kunjungan_data = kunjungan_data
        self.action_space = action_space
        self.waktu_operasional = waktu_operasional or settings.WAKTU_OPERASIONAL
        self.kondisi_cuaca = kondisi_cuaca
        self.hari_kuliah = hari_kuliah
        self.start_lokasi_id = start_lokasi_id

        # Pre-compute location statistics for reward calculation
        self.lokasi_stats = self._compute_lokasi_stats()

        # Pre-compute reward lookup: (lokasi_id, durasi_bin, cuaca, hari) → avg reward
        self.reward_lookup = self._compute_reward_lookup()

        # Pre-compute distances between locations
        self.distances = self._compute_distances()

        # Track visited locations and remaining time during episode
        self._visited: List[int] = []
        self._waktu_tersisa: int = self.waktu_operasional

    def _compute_lokasi_stats(self) -> Dict[int, Dict[str, float]]:
        """Compute average sales statistics for each location."""
        stats = {}
        for lokasi in self.lokasi_list:
            lokasi_kunjungan = [
                k for k in self.kunjungan_data if k.lokasi_id == lokasi.id
            ]
            if lokasi_kunjungan:
                # Calculate total sales for each visit
                terjual_per_kunjungan = [
                    sum(t.jumlah_terjual for t in k.transaksi)
                    for k in lokasi_kunjungan
                ]

                # Average duration per visit in minutes
                # k.durasi_mangkal is already in minutes
                durasi_per_kunjungan = [
                    (k.durasi_mangkal or 0.0)
                    for k in lokasi_kunjungan
                ]

                avg_terjual = sum(terjual_per_kunjungan) / len(lokasi_kunjungan)
                avg_durasi = sum(durasi_per_kunjungan) / len(lokasi_kunjungan)
                total_terjual = sum(terjual_per_kunjungan)
            else:
                # Default values for unexplored locations (exploration bonus)
                avg_terjual = 5.0  # Optimistic default
                avg_durasi = 30.0
                total_terjual = 0

            stats[lokasi.id] = {
                "avg_terjual": avg_terjual,
                "avg_durasi": avg_durasi,
                "total_terjual": total_terjual,
                "effectiveness": float(avg_terjual) / max(float(avg_durasi), 1.0)  # Sales per minute
            }
        return stats

    def _compute_reward_lookup(self) -> Dict[Tuple[int, int, str, int], float]:
        """
        Build reward lookup table from historical kunjungan data.
        Key: (lokasi_id, durasi_bin, kondisi_cuaca, hari_kuliah)
        Value: average jumlah_terjual (reward per lokasi)

        Jika tidak ada data untuk kombinasi tertentu, fallback ke rata-rata lokasi.
        """
        # Aggregate: (lokasi_id, durasi_bin, cuaca, hari) → [list of rewards]
        aggregates: Dict[Tuple[int, int, str, int], List[float]] = {}

        for k in self.kunjungan_data:
            if k.waktu_selesai is None:
                continue  # Skip kunjungan aktif

            # durasi_mangkal sudah dalam menit
            durasi_menit = k.durasi_mangkal or 0.0
            durasi_bin = discretize_durasi(durasi_menit)

            # Hitung total penjualan selama kunjungan ini
            total_penjualan = sum(t.jumlah_terjual for t in k.transaksi)

            key = (k.lokasi_id, durasi_bin, k.kondisi_cuaca or "cerah", k.hari_kuliah)
            if key not in aggregates:
                aggregates[key] = []
            aggregates[key].append(float(total_penjualan))

        # Convert to averages
        lookup = {}
        for key, values in aggregates.items():
            lookup[key] = sum(values) / len(values) if values else 0.0

        return lookup

    def _compute_distances(self) -> Dict[Tuple[int, int], float]:
        """Compute distances between all location pairs (in km)."""
        distances = {}
        for i, loc1 in enumerate(self.lokasi_list):
            for j, loc2 in enumerate(self.lokasi_list):
                if i != j:
                    dist = self._haversine_distance(
                        loc1.latitude, loc1.longitude,
                        loc2.latitude, loc2.longitude
                    )
                    distances[(loc1.id, loc2.id)] = dist
                else:
                    distances[(loc1.id, loc2.id)] = 0.0
        return distances

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in km."""
        R = 6371  # Earth's radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _get_travel_time(self, from_id: int, to_id: int) -> int:
        """Estimate travel time in minutes (assuming 15 km/h average speed)."""
        distance = self.distances.get((from_id, to_id), 1.0)
        speed_kmh = 15  # Average speed for a vendor on foot/bicycle
        time_hours = distance / speed_kmh
        return max(int(time_hours * 60), 5)  # Minimum 5 minutes

    def get_initial_state(self) -> State:
        """
        Get the initial state for an episode.
        Implements getInitialState() from Class Diagram.
        """
        if self.start_lokasi_id is not None and self.start_lokasi_id in self.lokasi_ids:
            initial_lokasi = self.start_lokasi_id
        else:
            initial_lokasi = self.lokasi_ids[0] if self.lokasi_ids else 0
            
        self._visited = [initial_lokasi]
        self._waktu_tersisa = self.waktu_operasional

        return State(
            lokasi_saat_ini=initial_lokasi,
            kondisi_cuaca=self.kondisi_cuaca,
            hari_kuliah=self.hari_kuliah
        )

    def execute_action(
        self, state: State, action: CompositeAction
    ) -> Tuple[State, float]:
        """
        Execute a composite action and return new state and reward.

        Setiap action = 1 step Q-Learning = 1 kunjungan lokasi.

        Args:
            state: Current state
            action: CompositeAction (lokasi_tujuan, durasi_mangkal)

        Returns:
            Tuple of (new_state, reward_per_lokasi)
        """
        lokasi_tujuan = action.lokasi_tujuan
        durasi_mangkal = action.durasi_mangkal

        # Hitung waktu perjalanan ke lokasi tujuan
        if lokasi_tujuan != state.lokasi_saat_ini:
            travel_time = self._get_travel_time(state.lokasi_saat_ini, lokasi_tujuan)
        else:
            travel_time = 0

        # Total waktu yang dihabiskan: perjalanan + mangkal
        total_time = travel_time + durasi_mangkal

        # Kurangi waktu tersisa
        self._waktu_tersisa -= total_time

        # Tambahkan ke visited
        if lokasi_tujuan not in self._visited:
            self._visited.append(lokasi_tujuan)

        # Hitung reward per lokasi
        reward = self.calculate_reward(state, action, travel_time)

        # State baru: lokasi berubah, cuaca & hari tetap
        new_state = State(
            lokasi_saat_ini=lokasi_tujuan,
            kondisi_cuaca=state.kondisi_cuaca,
            hari_kuliah=state.hari_kuliah
        )

        return new_state, reward

    def calculate_reward(
        self, state: State, action: CompositeAction, travel_time: int
    ) -> float:
        """
        Calculate reward based on actual historical sales data PER LOKASI.

        Reward components:
        1. Penjualan historis di lokasi tujuan untuk durasi & kondisi yang dipilih
        2. Penalty perjalanan (opportunity cost)
        3. Bonus eksplorasi untuk lokasi yang belum dikunjungi

        Args:
            state: Current state
            action: CompositeAction being executed
            travel_time: Travel time in minutes

        Returns:
            Reward value (float) — represents per-location sales
        """
        lokasi_tujuan = action.lokasi_tujuan
        durasi_mangkal = action.durasi_mangkal
        cuaca = state.kondisi_cuaca
        hari = state.hari_kuliah

        # 1. Reward utama: rata-rata penjualan historis per lokasi
        #    Lookup exact match: (lokasi, durasi_bin, cuaca, hari)
        reward_key = (lokasi_tujuan, durasi_mangkal, cuaca, hari)
        reward = self.reward_lookup.get(reward_key, 0.0)

        if reward == 0.0:
            # Fallback 1: Coba tanpa filter cuaca/hari — rata-rata lokasi secara umum
            fallback_rewards = [
                v for k, v in self.reward_lookup.items()
                if k[0] == lokasi_tujuan
            ]
            if fallback_rewards:
                reward = sum(fallback_rewards) / len(fallback_rewards)
            else:
                # Fallback 2: Gunakan avg_terjual dari lokasi_stats
                stats = self.lokasi_stats.get(lokasi_tujuan, {})
                reward = stats.get("avg_terjual", 0.0)

        # 2. Weather & calendar modifier
        weather_modifier = {
            "cerah": 1.0,
            "mendung": 0.8,
            "hujan": 0.5
        }.get(cuaca, 1.0)

        calendar_modifier = 1.2 if hari else 0.7

        reward *= weather_modifier * calendar_modifier

        # 3. Penalty perjalanan (opportunity cost — bisa jualan selama waktu perjalanan)
        if travel_time > 0:
            reward -= travel_time * 0.5

        # 4. Bonus eksplorasi
        if lokasi_tujuan not in self._visited:
            reward += 5.0

        return reward

    def is_terminal_state(self) -> bool:
        """
        Check if current episode should end.
        Terminal when: waktu habis atau semua lokasi sudah dikunjungi.
        """
        if self._waktu_tersisa <= 0:
            return True
        # Juga terminal jika semua lokasi sudah dikunjungi
        # dan waktu tidak cukup untuk kunjungan terpendek
        if len(self._visited) >= len(self.lokasi_ids):
            return True
        return False

    def get_available_actions(self, state: State) -> List[int]:
        """
        Get list of available action indices in current state.

        Hanya action yang feasible (cukup waktu untuk perjalanan + mangkal).
        """
        available = []

        for idx, action in enumerate(self.action_space.actions):
            # Hitung waktu yang dibutuhkan
            if action.lokasi_tujuan != state.lokasi_saat_ini:
                travel = self._get_travel_time(
                    state.lokasi_saat_ini, action.lokasi_tujuan
                )
            else:
                travel = 0

            total_needed = travel + action.durasi_mangkal

            # Hanya tambahkan jika cukup waktu
            if total_needed <= self._waktu_tersisa:
                available.append(idx)

        # Jika tidak ada action yang feasible, kembalikan semua
        # (terminal state akan ditangani oleh is_terminal_state)
        if not available and self.action_space.size > 0:
            # Return action durasi terpendek sebagai fallback
            min_dur_actions = [
                idx for idx, a in enumerate(self.action_space.actions)
                if a.durasi_mangkal == DURASI_VALUES[0]
            ]
            return min_dur_actions[:1] if min_dur_actions else [0]

        return available


# =========================================================
# Q-LEARNING AGENT
# =========================================================

class QLearningAgent:
    """
    Q-Learning Agent for route optimization.
    Implements: chooseAction(), updateQ()
    Uses ε-greedy exploration strategy and Bellman update rule.
    """

    def __init__(
        self,
        alpha: float = None,
        gamma: float = None,
        epsilon: float = None,
        q_table: QTable = None,
        action_space: ActionSpace = None
    ):
        """
        Initialize Q-Learning agent.

        Args:
            alpha: Learning rate (default from settings)
            gamma: Discount factor (default from settings)
            epsilon: Exploration rate for ε-greedy (default from settings)
            q_table: Q-Table instance
            action_space: ActionSpace instance
        """
        self.alpha = alpha if alpha is not None else settings.ALPHA
        self.gamma = gamma if gamma is not None else settings.GAMMA
        self.epsilon = epsilon if epsilon is not None else settings.EPSILON
        self.q_table = q_table or QTable()
        self.action_space = action_space

    def choose_action(self, state: State, available_actions: List[int]) -> int:
        """
        Choose action using ε-greedy strategy.
        Implements chooseAction() from Class Diagram.

        With probability ε: explore (random action)
        With probability 1-ε: exploit (best known action)

        Returns:
            Action index (int) for the composite action
        """
        if random.random() < self.epsilon:
            # Exploration: random action from available
            return random.choice(available_actions)
        else:
            # Exploitation: best action based on Q-values
            best_action = available_actions[0]
            best_value = float("-inf")

            for action_idx in available_actions:
                q_value = self.q_table.get_q_value(state, action_idx)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action_idx

            return best_action

    def update_q(
        self, state: State, action_index: int, reward: float, next_state: State
    ):
        """
        Update Q-value using Bellman equation.
        Implements updateQ() from Class Diagram.

        Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        # Current Q-value
        current_q = self.q_table.get_q_value(state, action_index)

        # Maximum Q-value for next state
        num_actions = self.action_space.size if self.action_space else 0
        max_next_q = self.q_table.get_max_q_value(next_state, num_actions)

        # Bellman update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Update Q-table
        self.q_table.update_q_value(state, action_index, new_q)

    def get_optimal_route(
        self, env: QLearningEnvironment
    ) -> Tuple[List[int], float, List[Dict], List[Dict]]:
        """
        Get optimal route by following the greedy policy (no exploration).

        Returns:
            Tuple of:
            - route: List of location IDs in visit order
            - total_reward: Cumulative reward
            - action_details: List of step details
            - durasi_per_lokasi: List of {lokasi_id, durasi_mangkal} per stop
        """
        state = env.get_initial_state()
        route = [state.lokasi_saat_ini]
        total_reward = 0.0
        action_details = []
        durasi_per_lokasi = []

        # Record durasi for starting location (will be updated on first MOVE)
        max_steps = len(env.lokasi_ids) * 2  # Safety limit
        step_count = 0

        while not env.is_terminal_state() and step_count < max_steps:
            available_actions = env.get_available_actions(state)
            if not available_actions:
                break

            # Always exploit (no exploration during inference)
            best_action_idx = available_actions[0]
            best_value = float("-inf")

            for action_idx in available_actions:
                q_value = self.q_table.get_q_value(state, action_idx)
                if q_value > best_value:
                    best_value = q_value
                    best_action_idx = action_idx

            action = self.action_space.index_to_action(best_action_idx)
            next_state, reward = env.execute_action(state, action)
            total_reward += reward

            action_details.append({
                "lokasi_asal": state.lokasi_saat_ini,
                "lokasi_tujuan": action.lokasi_tujuan,
                "durasi_mangkal": action.durasi_mangkal,
                "reward": reward,
                "waktu_tersisa": env._waktu_tersisa
            })

            # Track route (hanya tambah jika lokasi baru)
            if action.lokasi_tujuan not in route:
                route.append(action.lokasi_tujuan)

            # Track durasi + reward per lokasi
            durasi_per_lokasi.append({
                "lokasi_id": action.lokasi_tujuan,
                "durasi_mangkal": action.durasi_mangkal,
                "reward": reward
            })

            state = next_state
            step_count += 1

        return route, total_reward, action_details, durasi_per_lokasi


# =========================================================
# SISTEM CONTROLLER
# =========================================================

class SistemController:
    """
    System Controller for Q-Learning optimization.
    Implements: jalankanQLearning(), simpanHasil()
    """

    def __init__(
        self,
        db: Session,
        db_manager,
        pedagang_id: int
    ):
        """Initialize sistem controller."""
        self.db = db
        self.db_manager = db_manager
        self.pedagang_id = pedagang_id

    def jalankan_q_learning(
        self,
        max_episodes: int = None,
        kondisi_cuaca: str = "cerah",
        hari_kuliah: int = 1,
        start_lokasi_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Q-Learning training and return optimal route + durasi per lokasi.
        Implements jalankanQLearning() from Class Diagram.

        Setiap kunjungan lokasi dianggap sebagai 1 step dalam Q-Learning.
        Q-value diupdate berdasarkan:
            Q(s,a) = Q(s,a) + α[reward + γ max Q(s',a') - Q(s,a)]

        Returns:
            Dict with keys:
            - success, message
            - rute_optimal: List[Dict] with location details
            - durasi_per_lokasi: List[Dict] with {lokasi_id, nama, durasi_menit}
            - total_reward, total_jarak
            - episode_rewards
            - lokasi_stats, route_lokasi_ids (for reward_service)
        """
        max_episodes = max_episodes or settings.MAX_EPISODES

        # Get data from database
        lokasi_list = self.db_manager.get_lokasi(self.db, self.pedagang_id)
        kunjungan_data = self.db_manager.get_kunjungan_list(
            db=self.db,
            pedagang_id=self.pedagang_id,
            limit=1000,
            hanya_selesai=True
        )

        if not lokasi_list:
            return {
                "success": False,
                "message": "Tidak ada lokasi terdaftar. Tambahkan lokasi terlebih dahulu.",
                "rute_optimal": [],
                "durasi_per_lokasi": [],
                "total_reward": 0.0,
                "rekomendasi": ""
            }

        # Build action space from available locations
        lokasi_ids = [l.id for l in lokasi_list]
        action_space = ActionSpace(lokasi_ids)

        # Initialize Q-Table with database persistence
        q_table = QTable(self.db, self.pedagang_id, self.db_manager)

        # Initialize agent
        agent = QLearningAgent(
            q_table=q_table,
            action_space=action_space
        )

        # Initialize environment
        env = QLearningEnvironment(
            lokasi_list=lokasi_list,
            kunjungan_data=kunjungan_data,
            action_space=action_space,
            kondisi_cuaca=kondisi_cuaca,
            hari_kuliah=hari_kuliah,
            start_lokasi_id=start_lokasi_id
        )

        # ========================
        # Training loop
        # ========================
        episode_rewards = []
        for episode in range(max_episodes):
            state = env.get_initial_state()
            total_reward = 0.0
            total_waktu = 0
            max_steps = len(lokasi_ids) * 2  # Safety limit
            step_count = 0

            while not env.is_terminal_state() and step_count < max_steps:
                # Choose action (ε-greedy)
                available_actions = env.get_available_actions(state)
                if not available_actions:
                    break

                action_idx = agent.choose_action(state, available_actions)
                action = action_space.index_to_action(action_idx)

                # Execute action — 1 step = 1 kunjungan lokasi
                next_state, reward = env.execute_action(state, action)

                # Update Q-value (Bellman)
                agent.update_q(state, action_idx, reward, next_state)

                total_reward += reward
                total_waktu += action.durasi_mangkal
                state = next_state
                step_count += 1

            episode_rewards.append(total_reward)

            # Save episode to database
            self.db_manager.save_episode(
                self.db, self.pedagang_id, total_reward, total_waktu
            )

        # ========================
        # Get optimal route after training
        # ========================
        # Re-init environment for inference
        env_inference = QLearningEnvironment(
            lokasi_list=lokasi_list,
            kunjungan_data=kunjungan_data,
            action_space=action_space,
            kondisi_cuaca=kondisi_cuaca,
            hari_kuliah=hari_kuliah
        )

        route, final_reward, action_details, durasi_raw = agent.get_optimal_route(
            env_inference
        )

        # Build route with location details
        route_with_details = []
        total_jarak = 0.0

        for i, lokasi_id in enumerate(route):
            lokasi = next((l for l in lokasi_list if l.id == lokasi_id), None)
            if lokasi:
                jarak_step = 0.0
                if i > 0:
                    prev_id = route[i - 1]
                    jarak_step = env_inference.distances.get(
                        (prev_id, lokasi_id), 0.0
                    )
                    total_jarak += jarak_step

                route_with_details.append({
                    "urutan": i + 1,
                    "lokasi_id": lokasi.id,
                    "nama": lokasi.nama,
                    "latitude": lokasi.latitude,
                    "longitude": lokasi.longitude,
                    "jarak_dari_sebelumnya": round(jarak_step, 2)
                })

        # Build durasi_per_lokasi output
        # Aggregate durasi & reward per lokasi (jika ada duplikat, ambil yang terakhir)
        durasi_map: Dict[int, int] = {}
        reward_map: Dict[int, float] = {}
        for d in durasi_raw:
            durasi_map[d["lokasi_id"]] = d["durasi_mangkal"]
            reward_map[d["lokasi_id"]] = d.get("reward", 0.0)

        durasi_per_lokasi = []
        total_reward_lokasi = 0.0
        for lokasi_id in route:
            lokasi = next((l for l in lokasi_list if l.id == lokasi_id), None)
            if lokasi:
                durasi = durasi_map.get(lokasi_id, 30)  # Default 30 menit
                rwd = round(reward_map.get(lokasi_id, 0.0), 2)
                total_reward_lokasi += rwd
                durasi_per_lokasi.append({
                    "lokasi_id": lokasi.id,
                    "nama": lokasi.nama,
                    "durasi_menit": durasi,
                    "reward": rwd
                })

        # Generate recommendation text
        rekomendasi = self._generate_rekomendasi(
            route_with_details, final_reward, kondisi_cuaca, hari_kuliah,
            durasi_per_lokasi=durasi_per_lokasi
        )

        # Save optimal route
        self.simpan_hasil(route, total_jarak, final_reward, rekomendasi)

        return {
            "success": True,
            "message": f"Optimasi selesai setelah {max_episodes} episode training.",
            "rute_optimal": route_with_details,
            "durasi_per_lokasi": durasi_per_lokasi,
            "total_reward": final_reward,
            "total_reward_lokasi": round(total_reward_lokasi, 2),
            "total_jarak": round(total_jarak, 2),
            "rekomendasi": rekomendasi,
            "episode_rewards": episode_rewards[-10:],  # Last 10 episodes
            # Data tambahan untuk reward_service
            "lokasi_stats": env_inference.lokasi_stats,
            "route_lokasi_ids": route,
            "kondisi_cuaca": kondisi_cuaca,
            "hari_kuliah": hari_kuliah,
        }

    def simpan_hasil(
        self, urutan_lokasi: List[int], total_jarak: float,
        total_reward: float, rekomendasi: str
    ):
        """
        Save optimization results to database.
        Implements simpanHasil() from Class Diagram.
        """
        self.db_manager.save_rute_optimal(
            self.db,
            self.pedagang_id,
            urutan_lokasi,
            total_jarak,
            total_reward,
            rekomendasi
        )

    def _generate_rekomendasi(
        self, route: List[Dict], reward: float, cuaca: str, hari: int,
        perkiraan_penghasilan: int = 0, kategori: str = "",
        durasi_per_lokasi: List[Dict] = None
    ) -> str:
        """Generate recommendation text based on optimization results."""
        if not route:
            return "Tidak ada rute yang direkomendasikan."

        cuaca_text = {
            "cerah": "cerah ☀️",
            "mendung": "mendung ☁️",
            "hujan": "hujan 🌧️"
        }.get(cuaca, cuaca)

        hari_text = "hari kuliah 📚" if hari else "akhir pekan 🏖️"

        reko = f"📍 Rute Optimal untuk kondisi {cuaca_text} dan {hari_text}:\n\n"

        for loc in route:
            reko += f"{loc['urutan']}. {loc['nama']}"
            # Tambahkan durasi rekomendasi jika tersedia
            if durasi_per_lokasi:
                durasi_info = next(
                    (d for d in durasi_per_lokasi if d["lokasi_id"] == loc["lokasi_id"]),
                    None
                )
                if durasi_info:
                    reko += f" — ⏱️ {durasi_info['durasi_menit']} menit"
            reko += "\n"

        # Tampilkan perkiraan penghasilan jika tersedia
        if perkiraan_penghasilan > 0:
            reko += f"\n💰 Perkiraan Penghasilan: Rp {perkiraan_penghasilan:,}\n".replace(",", ".")
            if kategori:
                reko += f"📊 Kategori: {kategori}\n"
        else:
            reko += f"\n💰 Estimasi reward: {reward:.2f}\n"

        reko += "\n💡 Tips: Catat hasil penjualan di setiap lokasi untuk meningkatkan akurasi rekomendasi!"

        return reko
