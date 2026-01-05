# Q-Learning Pathfinder (GridWorld)

A grid-based reinforcement learning project where an agent learns to navigate from a start cell to a goal cell in randomly generated environments with obstacles. The agent uses **tabular Q-learning** (no neural networks) and is evaluated across multiple mazes per run.

## What this demonstrates
- Reinforcement learning (online learning from trial-and-error)
- Interpretable ML: Q-values, policies, and behavior
- Experimental evaluation across many environments (not just one “lucky” maze)
- Clear visualizations of learned behavior and failure modes

## How it works (high level)
- **Environment:** 2D grid, with empty cells and walls  
- **Actions:** up / right / down / left  
- **Rewards (typical):**
  - Negative step penalty (encourages shorter paths)
  - Larger penalty for attempting to move into walls
  - Large positive reward for reaching the goal
  - Optional shaping term based on distance to goal (to reduce sparse-reward difficulty)

- **Agent:** Q-learning with ε-greedy exploration  
  Updates a Q-table `Q(s, a)` after each step:
  \[
  Q(s,a) \leftarrow Q(s,a) + \alpha\left(r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right)
  \]

## Running the project

### 1) Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt