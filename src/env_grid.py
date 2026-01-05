from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTIONS = (0, 1, 2, 3)

@dataclass(frozen=True)
class State:
    r: int
    c: int

class GridWorld:
    """
    Simple grid-world environment for reinforcement learning.

    Grid cells:
      0 = empty
      1 = wall

    The agent starts at `start` and tries to reach `goal`.
    Rewards:
      - step_penalty each move
      - wall_penalty if you try to move into a wall or outside grid (agent stays put)
      + goal_reward if you reach the goal (episode ends)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        step_penalty: float = -1.0,
        wall_penalty: float = -10.0,
        goal_reward: float = 100.0,
        max_steps: int = 500,
        grid: Optional[List[List[int]]] = None,
        start: State = State(0, 0),
        goal: Optional[State] = None,
    ):
        self.rows = rows
        self.cols = cols

        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.grid = grid if grid is not None else [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = start
        self.goal = goal if goal is not None else State(rows - 1, cols - 1)

        self.agent = State(self.start.r, self.start.c)
        self.steps_taken = 0

        # safety: ensure start/goal not walls
        self._ensure_valid_positions()

    def copy(self):
        new_env = GridWorld(
            rows=self.rows,
            cols=self.cols,
            step_penalty=self.step_penalty,
            wall_penalty=self.wall_penalty,
            goal_reward=self.goal_reward,
            max_steps=self.max_steps,
            start=State(self.start.r, self.start.c),
            goal=State(self.goal.r, self.goal.c),
        )
        new_env.grid = [row[:] for row in self.grid]
        return new_env
    

    def _ensure_valid_positions(self) -> None:
        if self.is_wall(self.start):
            raise ValueError("Start position is on a wall.")
        if self.is_wall(self.goal):
            raise ValueError("Goal position is on a wall.")

    def in_bounds(self, s: State) -> bool:
        return 0 <= s.r < self.rows and 0 <= s.c < self.cols

    def is_wall(self, s: State) -> bool:
        return self.grid[s.r][s.c] == 1

    def reset(self) -> State:
        self.agent = State(self.start.r, self.start.c)
        self.steps_taken = 0
        return self.agent

    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        """
        Take one action in the environment.

        Returns:
          next_state, reward, done, info
        """
        self.steps_taken += 1

        dr, dc = self._delta(action)
        candidate = State(self.agent.r + dr, self.agent.c + dc)

        # hit border or wall => stay put + penalty
        if (not self.in_bounds(candidate)) or self.is_wall(candidate):
            reward = self.wall_penalty
            done = False
            info = {"hit_wall": True}
            return self.agent, reward, done, info

        # move
        self.agent = candidate

        # reached goal
        if self.agent == self.goal:
            return self.agent, self.goal_reward, True, {"reached_goal": True}

        # max steps reached => end episode (counts as fail)
        if self.steps_taken >= self.max_steps:
            return self.agent, self.step_penalty, True, {"max_steps": True}

        distance_penalty = -0.1 * (
            abs(self.agent.r - self.goal.r) +
            abs(self.agent.c - self.goal.c)
        )
        return self.agent, self.step_penalty + distance_penalty, False, {}

    def _delta(self, action: int) -> Tuple[int, int]:
        if action == 0:   # UP
            return (-1, 0)
        if action == 1:   # RIGHT
            return (0, 1)
        if action == 2:   # DOWN
            return (1, 0)
        if action == 3:   # LEFT
            return (0, -1)
        raise ValueError(f"Invalid action: {action}")

    def set_wall(self, r: int, c: int, value: bool = True) -> None:
        # Don't allow walls on start/goal
        if (r, c) == (self.start.r, self.start.c) or (r, c) == (self.goal.r, self.goal.c):
            return
        self.grid[r][c] = 1 if value else 0

    def randomize_walls(self, density: float = 0.20, seed: int | None = None):
        """
        Randomly generate walls with a given density.
        Ensures start and goal are always free.
        """
        import numpy as np

        if seed is not None:
            np.random.seed(seed)

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in [
                    (self.start.r, self.start.c),
                    (self.goal.r, self.goal.c),
                ]:
                    self.grid[r][c] = 0
                else:
                    self.grid[r][c] = 1 if np.random.rand() < density else 0
    
    

    def move_goal_randomly(self) -> None:
        """
        Move the goal to a random empty cell.
        """
        empty_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0 and (r, c) != (self.start.r, self.start.c)
        ]
        if empty_cells:
            r, c = empty_cells[np.random.randint(len(empty_cells))]
            self.goal = State(r, c)

    def render_ascii(self) -> str:
        """
        Quick text view (useful before we add matplotlib animation).
        """
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                s = State(r, c)
                if s == self.agent:
                    row_chars.append("A")
                elif s == self.start:
                    row_chars.append("S")
                elif s == self.goal:
                    row_chars.append("G")
                elif self.is_wall(s):
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)