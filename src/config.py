from dataclasses import dataclass

@dataclass
class EnvConfig:
    rows: int = 10
    cols: int = 10

    # Rewards (tunable)
    step_penalty: float = -1.0
    wall_penalty: float = -10.0
    goal_reward: float = 100.0

    # Episode limit (prevents infinite wandering)
    max_steps_per_episode: int = 300