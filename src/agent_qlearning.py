from collections import defaultdict
import random
import numpy as np
from typing import Tuple
from src.env_grid import ACTIONS, State


class QLearningAgent:
    """
    Tabular Q-learning agent for a discrete grid world.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.alpha = learning_rate
        self.gamma = discount_factor

        self.epsilon = 1.0
        self.epsilon_decay = 0.997  
        self.epsilon_min = 0.05

        # Q-table: (state, action) -> value
        self.q = defaultdict(float)

    def _key(self, state: State, action: int) -> Tuple[int, int, int]:
        return (state.r, state.c, action)

    def get_q(self, state: State, action: int) -> float:
        return self.q[self._key(state, action)]

    def choose_action(self, state, epsilon_override=None):
        """
        Choose an action using epsilon-greedy.
        If epsilon_override is provided, use that instead of self.epsilon.
        """
        eps = self.epsilon if epsilon_override is None else epsilon_override

        if np.random.rand() < eps:
            return np.random.choice(ACTIONS)
        return self.best_action(state)
    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """
        Core Q-learning update rule.
        """
        old_q = self.get_q(state, action)

        if done:
            target = reward
        else:
            next_max = max(self.get_q(next_state, a) for a in ACTIONS)
            target = reward + self.gamma * next_max

        new_q = old_q + self.alpha * (target - old_q)
        self.q[self._key(state, action)] = new_q

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state: State) -> int:
        """
        Choose the best action without exploration (used for visualization).
        """
        q_values = [self.get_q(state, a) for a in ACTIONS]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def copy(self):
        """
        Create a deep copy of the agent (for snapshotting learning progress).
        """
        new_agent = QLearningAgent(
            learning_rate=self.alpha,
            discount_factor=self.gamma,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
        )
        new_agent.q = self.q.copy()
        return new_agent
    
    def get_state_value(self, state: State) -> float:
        """
        Return max Q-value over actions for a given state.
        Used for heatmap visualization.
        """
        return max(self.get_q(state, a) for a in ACTIONS)