"""
Q-Learning Agent for Sate Keliling Route Optimization.
Implements the QLearningAgent, QLearningEnvironment, and QTable from the Class Diagram.

State: lokasi_saat_ini, waktu_tersisa, lokasi_dikunjungi
Action: STAY (0) or MOVE (1)
Reward: Based on sales effectiveness and travel time
Update Rule: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
"""

import random
import json
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Lokasi, Penjualan, Kunjungan


class Action(IntEnum):
    """
    Action space for the Q-Learning agent.
    STAY: Tetap di lokasi saat ini
    MOVE: Pindah ke lokasi berikutnya
    """
    STAY = 0
    MOVE = 1

@dataclass
class ActionDetail:
    action_type: str
    from_lokasi: int
    to_lokasi: Optional[int]

@dataclass(frozen=True)
class State:
    """
    State representation for Q-Learning.
    Attributes:
        lokasi_saat_ini: Current location ID
        waktu_tersisa: Remaining operational time in minutes
        lokasi_dikunjungi: Tuple of visited location IDs
    """
    lokasi_saat_ini: int
    waktu_tersisa: int
    lokasi_dikunjungi: Tuple[int, ...]
    
    @staticmethod
    def discretize_time(waktu: int) -> str:
        if waktu > 120:
            return "banyak"
        elif waktu > 60:
            return "sedang"
        else:
            return "sedikit"
        
    def to_key(self) -> str:
        """Convert state to a unique string key for Q-table lookup."""
        return json.dumps({
            "lokasi": self.lokasi_saat_ini,
            "waktu": State.discretize_time(self.waktu_tersisa),
            "dikunjungi": sorted(self.lokasi_dikunjungi)
        }, sort_keys=True)
    
    @classmethod
    def from_key(cls, key: str) -> "State":
        """Create state from a string key."""
        data = json.loads(key)
        return cls(
            lokasi_saat_ini = data["lokasi"],
            waktu_tersisa = data["waktu"],
            lokasi_dikunjungi = tuple(data["dikunjungi"])
        )


class QTable:
    """
    Q-Table for storing Q-values.
    Supports both in-memory and database-backed storage.
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
    
    def get_q_value(self, state: State, action: Action) -> float:
        """
        Get Q-value for a state-action pair.
        Returns 0.0 if not found (optimistic initialization).
        """
        state_key = state.to_key()
        if state_key in self.q_values:
            return self.q_values[state_key].get(int(action), 0.0)
        return 0.0
    
    def update_q_value(self, state: State, action: Action, value: float):
        """Update Q-value for a state-action pair."""
        state_key = state.to_key()
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        self.q_values[state_key][int(action)] = value
        
        # Persist to database if available
        if self.db and self.pedagang_id and self.db_manager:
            self.db_manager.update_q_value(
                self.db, self.pedagang_id, state_key, int(action), value
            )
    
    def get_best_action(self, state: State) -> Tuple[Action, float]:
        """Get the best action and its Q-value for a given state."""
        state_key = state.to_key()
        if state_key not in self.q_values:
            return Action.STAY, 0.0
        
        best_action = Action.STAY
        best_value = float("-inf")
        
        for action in Action:
            q_val = self.q_values[state_key].get(int(action), 0.0)
            if q_val > best_value:
                best_value = q_val
                best_action = action
        
        return best_action, best_value if best_value != float("-inf") else 0.0
    
    def get_max_q_value(self, state: State) -> float:
        """Get the maximum Q-value for any action in a given state."""
        _, max_q = self.get_best_action(state)
        return max_q


class QLearningEnvironment:
    """
    Environment for Q-Learning training.
    Implements: getInitialState(), executeAction(), calculateReward(), isTerminalState()
    """
    
    def __init__(
        self,
        lokasi_list: List[Lokasi],
        kunjungan_data: List[Kunjungan],
        waktu_operasional: int = None,
        kondisi_cuaca: str = "cerah",
        hari_kuliah: int = 1
    ):
        """
        Initialize the environment.
        
        Args:
            lokasi_list: List of locations available
            kunjungan_data: Historical visit data for reward calculation
            waktu_operasional: Total operational time in minutes
            kondisi_cuaca: Weather condition (cerah, mendung, hujan)
            hari_kuliah: 1 if weekday/school day, 0 if weekend/holiday
        """
        self.lokasi_list = lokasi_list
        self.lokasi_ids = [l.id for l in lokasi_list]
        self.kunjungan_data = kunjungan_data
        self.waktu_operasional = waktu_operasional or settings.WAKTU_OPERASIONAL
        self.kondisi_cuaca = kondisi_cuaca
        self.hari_kuliah = hari_kuliah
        
        # Pre-compute location statistics for reward calculation
        self.lokasi_stats = self._compute_lokasi_stats()
        
        # Pre-compute distances between locations
        self.distances = self._compute_distances()
        
        # Current state
        self.current_state: Optional[State] = None
    
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
                # k.durasi_mangkal is in hours, so we multiply by 60
                durasi_per_kunjungan = [
                    (k.durasi_mangkal or 0.0) * 60.0 
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
        # Start at the first location (basecamp) with full operational time
        initial_lokasi = self.lokasi_ids[0] if self.lokasi_ids else 0
        self.current_state = State(
            lokasi_saat_ini=initial_lokasi,
            waktu_tersisa=self.waktu_operasional,
            lokasi_dikunjungi=(initial_lokasi,)
        )
        return self.current_state
    
    def execute_action(self, state: State, action: Action) -> Tuple[State, float]:
        """
        Execute an action and return new state and reward.
        Implements executeAction() from Class Diagram.
        
        Args:
            state: Current state
            action: Action to take (STAY or MOVE)
            
        Returns:
            Tuple of (new_state, reward)
        """
        if action == Action.STAY:
            # Stay at current location - spend time selling
            # Average stay duration based on historical data
            avg_durasi = int(self.lokasi_stats.get(
                state.lokasi_saat_ini, {}
            ).get("avg_durasi", 30))
            
            time_spent = min(avg_durasi, state.waktu_tersisa)
            
            new_state = State(
                lokasi_saat_ini=state.lokasi_saat_ini,
                waktu_tersisa=state.waktu_tersisa - time_spent,
                lokasi_dikunjungi=state.lokasi_dikunjungi
            )
            
            reward = self.calculate_reward(state, action, time_spent)
            
        else:  # MOVE
            # Find next unvisited location or cycle back
            unvisited = [
                lid for lid in self.lokasi_ids
                if lid not in state.lokasi_dikunjungi
            ]
            
            if unvisited:
                # Move to the nearest unvisited location
                next_lokasi = min(
                    unvisited,
                    key=lambda lid: self.distances.get(
                        (state.lokasi_saat_ini, lid), float("inf")
                    )
                )
            else:
                # All visited, go back to starting point
                next_lokasi = self.lokasi_ids[0] if self.lokasi_ids else state.lokasi_saat_ini
            
            travel_time = self._get_travel_time(state.lokasi_saat_ini, next_lokasi)
            time_spent = min(travel_time, state.waktu_tersisa)
            
            new_visited = state.lokasi_dikunjungi
            if next_lokasi not in new_visited:
                new_visited = state.lokasi_dikunjungi + (next_lokasi,)
            
            new_state = State(
                lokasi_saat_ini=next_lokasi,
                waktu_tersisa=state.waktu_tersisa - time_spent,
                lokasi_dikunjungi=new_visited
            )
            
            reward = self.calculate_reward(state, action, time_spent, next_lokasi)
        
        self.current_state = new_state
        return new_state, reward
    
    def calculate_reward(
        self, state: State, action: Action, time_spent: int, next_lokasi: int = None
    ) -> float:
        """
        Calculate reward based on action and state.
        Implements calculateReward() from Class Diagram.
        
        Reward components:
        - STAY: Positive reward based on sales effectiveness
        - MOVE: Negative for travel time, positive for reaching high-potential location
        - Bonus for visiting new locations
        - Weather and academic calendar adjustments
        """
        reward = 0.0
        
        # Weather modifier
        weather_modifier = {
            "cerah": 1.0,
            "mendung": 0.8,
            "hujan": 0.5
        }.get(self.kondisi_cuaca, 1.0)
        
        # Academic calendar modifier
        calendar_modifier = 1.2 if self.hari_kuliah else 0.7
        
        if action == Action.STAY:
            # Reward for staying = effectiveness * modifiers
            stats = self.lokasi_stats.get(state.lokasi_saat_ini, {})
            effectiveness = stats.get("effectiveness", 0.1)
            
            # Expected sales during stay
            expected_sales = effectiveness * time_spent * weather_modifier * calendar_modifier
            reward = expected_sales
            
        else:  # MOVE
            # Penalty for travel time (opportunity cost)
            reward -= time_spent * 0.1
            
            if next_lokasi:
                # Bonus for reaching high-potential location
                target_stats = self.lokasi_stats.get(next_lokasi, {})
                potential = target_stats.get("effectiveness", 0.1)
                reward += potential * 10 * weather_modifier * calendar_modifier
                
                # Exploration bonus for new locations
                if next_lokasi not in state.lokasi_dikunjungi:
                    reward += 5.0  # Exploration bonus
        
        return reward
    
    def is_terminal_state(self, state: State) -> bool:
        """
        Check if current state is terminal (episode ends).
        Implements isTerminalState() from Class Diagram.
        """
        # Episode ends when no time remaining
        return state.waktu_tersisa <= 0
    
    def get_available_actions(self, state: State) -> List[Action]:
        """Get list of available actions in current state."""
        actions = [Action.STAY]
        
        # Can only MOVE if there's enough time for travel
        if state.waktu_tersisa > 5:  # Minimum 5 minutes to move
            actions.append(Action.MOVE)
        
        return actions


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
        q_table: QTable = None
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            alpha: Learning rate (default from settings)
            gamma: Discount factor (default from settings)
            epsilon: Exploration rate for ε-greedy (default from settings)
            q_table: Q-Table instance
        """
        self.alpha = alpha if alpha is not None else settings.ALPHA
        self.gamma = gamma if gamma is not None else settings.GAMMA
        self.epsilon = epsilon if epsilon is not None else settings.EPSILON
        self.q_table = q_table or QTable()
    
    def choose_action(self, state: State, available_actions: List[Action]) -> Action:
        """
        Choose action using ε-greedy strategy.
        Implements chooseAction() from Class Diagram.
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (best known action)
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: best action based on Q-values
            best_action = available_actions[0]
            best_value = float("-inf")
            
            for action in available_actions:
                q_value = self.q_table.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            return best_action
    
    def update_q(
        self, state: State, action: Action, reward: float, next_state: State
    ):
        """
        Update Q-value using Bellman equation.
        Implements updateQ() from Class Diagram.
        
        Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        # Current Q-value
        current_q = self.q_table.get_q_value(state, action)
        
        # Maximum Q-value for next state
        max_next_q = self.q_table.get_max_q_value(next_state)
        
        # Bellman update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.q_table.update_q_value(state, action, new_q)
    
    def get_optimal_route(
        self, env: QLearningEnvironment
    ) -> Tuple[List[int], float, List[Dict]]:
        """
        Get optimal route by following the greedy policy.
        
        Returns:
            Tuple of (route as list of location IDs, total reward, action details)
        """
        state = env.get_initial_state()
        route = [state.lokasi_saat_ini]
        total_reward = 0.0
        action_details = []
        
        while not env.is_terminal_state(state):
            available_actions = env.get_available_actions(state)
            
            # Always exploit (no exploration during inference)
            best_action = available_actions[0]
            best_value = float("-inf")
            
            for action in available_actions:
                q_value = self.q_table.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            next_state, reward = env.execute_action(state, best_action)
            total_reward += reward
            
            action_details.append({
                "lokasi": state.lokasi_saat_ini,
                "action": best_action,
                "tujuan": next_state.lokasi_saat_ini if best_action == Action.MOVE else None,
                "reward": reward,
                "waktu_tersisa": next_state.waktu_tersisa
            })
            
            if best_action == Action.MOVE and next_state.lokasi_saat_ini not in route:
                route.append(next_state.lokasi_saat_ini)
            
            state = next_state
        
        return route, total_reward, action_details


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
        hari_kuliah: int = 1
    ) -> Dict[str, Any]:
        """
        Run Q-Learning training and return optimal route.
        Implements jalankanQLearning() from Class Diagram.
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
                "total_reward": 0.0,
                "rekomendasi": ""
            }
        
        # Initialize Q-Table with database persistence
        q_table = QTable(self.db, self.pedagang_id, self.db_manager)
        
        # Initialize agent
        agent = QLearningAgent(q_table=q_table)
        
        # Initialize environment
        env = QLearningEnvironment(
            lokasi_list=lokasi_list,
            kunjungan_data=kunjungan_data,
            kondisi_cuaca=kondisi_cuaca,
            hari_kuliah=hari_kuliah
        )
        
        # Training loop
        episode_rewards = []
        for episode in range(max_episodes):
            state = env.get_initial_state()
            total_reward = 0.0
            total_waktu = 0
            
            while not env.is_terminal_state(state):
                # Choose action
                available_actions = env.get_available_actions(state)
                action = agent.choose_action(state, available_actions)
                
                # Execute action
                next_state, reward = env.execute_action(state, action)
                
                # Update Q-value
                agent.update_q(state, action, reward, next_state)
                
                total_reward += reward
                total_waktu += (state.waktu_tersisa - next_state.waktu_tersisa)
                state = next_state
            
            episode_rewards.append(total_reward)
            
            # Save episode to database
            self.db_manager.save_episode(
                self.db, self.pedagang_id, total_reward, total_waktu
            )
        
        # Get optimal route after training
        route, final_reward, action_details = agent.get_optimal_route(env)
        
        # Build route with location details
        route_with_details = []
        total_jarak = 0.0
        
        for i, lokasi_id in enumerate(route):
            lokasi = next((l for l in lokasi_list if l.id == lokasi_id), None)
            if lokasi:
                jarak_step = 0.0
                if i > 0:
                    prev_id = route[i - 1]
                    jarak_step = env.distances.get((prev_id, lokasi_id), 0.0)
                    total_jarak += jarak_step

                route_with_details.append({
                    "urutan": i + 1,
                    "lokasi_id": lokasi.id,
                    "nama": lokasi.nama,
                    "latitude": lokasi.latitude,
                    "longitude": lokasi.longitude,
                    "jarak_dari_sebelumnya": round(jarak_step, 2)
                })
        
        # Generate recommendation text
        rekomendasi = self._generate_rekomendasi(
            route_with_details, final_reward, kondisi_cuaca, hari_kuliah
        )
        
        # Save optimal route
        self.simpan_hasil(route, total_jarak, final_reward, rekomendasi)
        
        return {
            "success": True,
            "message": f"Optimasi selesai setelah {max_episodes} episode training.",
            "rute_optimal": route_with_details,
            "total_reward": final_reward,
            "total_jarak": round(total_jarak, 2),
            "rekomendasi": rekomendasi,
            "episode_rewards": episode_rewards[-10:],  # Last 10 episodes
            # Data tambahan untuk reward_service
            "lokasi_stats": env.lokasi_stats,
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
        perkiraan_penghasilan: int = 0, kategori: str = ""
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
            reko += f"{loc['urutan']}. {loc['nama']}\n"
        
        # Tampilkan perkiraan penghasilan jika tersedia
        if perkiraan_penghasilan > 0:
            reko += f"\n💰 Perkiraan Penghasilan: Rp {perkiraan_penghasilan:,}\n".replace(",", ".")
            if kategori:
                reko += f"📊 Kategori: {kategori}\n"
        else:
            reko += f"\n💰 Estimasi reward: {reward:.2f}\n"
        
        reko += "\n💡 Tips: Catat hasil penjualan di setiap lokasi untuk meningkatkan akurasi rekomendasi!"
        
        return reko
