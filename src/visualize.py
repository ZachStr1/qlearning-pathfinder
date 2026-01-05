import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from typing import List


def moving_average(data: List[float], window: int = 20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")

def plot_final_path(env, agent, save_path, title="Final Learned Path"):
    """
    Plot the environment and overlay the agent's final greedy-policy path in red.
    """
    import matplotlib.pyplot as plt
    from src.utils import rollout_policy

    path, success = rollout_policy(env, agent)

    plt.figure(figsize=(6, 6))
    plt.imshow(env.grid, cmap="gray_r")

    xs = [s.c for s in path]
    ys = [s.r for s in path]

    color = "green" if success else "orange"
    label = "Success Path" if success else "Failed Path"

    plt.plot(xs, ys, color=color, linewidth=3, label=label)

    plt.scatter(env.start.c, env.start.r, c="cyan", s=100, label="Start")
    plt.scatter(env.goal.c, env.goal.r, c="lime", s=120, label="Goal")

    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(True)
    plt.legend()
    plt.title(title)

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_learning_curves(
    rewards: List[float],
    steps: List[int],
    successes: List[int],
    save_path="outputs/policy_comparison.png",
):
    episodes = np.arange(1, len(rewards) + 1)

    rewards_ma = moving_average(rewards)
    steps_ma = moving_average(steps)
    success_ma = moving_average(successes)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Reward curve ---
    axs[0].plot(episodes, rewards, alpha=0.3, label="Reward per Episode")
    axs[0].plot(
        episodes[len(episodes) - len(rewards_ma):],
        rewards_ma,
        linewidth=2,
        label="Moving Average",
    )
    axs[0].set_ylabel("Total Reward")
    axs[0].set_title("Learning Curve: Reward vs Episode")
    axs[0].legend()
    axs[0].grid(True)

    # --- Steps curve ---
    axs[1].plot(episodes, steps, alpha=0.3, label="Steps per Episode")
    axs[1].plot(
        episodes[len(episodes) - len(steps_ma):],
        steps_ma,
        linewidth=2,
        label="Moving Average",
    )
    axs[1].set_ylabel("Steps")
    axs[1].set_title("Path Efficiency Improvement")
    axs[1].legend()
    axs[1].grid(True)

    # --- Success rate ---
    axs[2].plot(
        episodes[len(episodes) - len(success_ma):],
        success_ma,
        linewidth=2,
        label="Success Rate (Moving Avg)",
    )
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Success Rate")
    axs[2].set_ylim(0, 1.05)
    axs[2].set_title("Goal Reach Success Rate")
    axs[2].legend()
    axs[2].grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(0.1)

def plot_convergence_distribution(convergence_episodes, save_path="outputs/convergence_distribution.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.hist(convergence_episodes, bins=8, edgecolor="black")
    plt.xlabel("Episodes to Convergence")
    plt.ylabel("Number of Mazes")
    plt.title("Convergence Distribution Across Random Mazes")

    plt.savefig(save_path)
    plt.close()



def plot_q_heatmap(agent, env, title="Q-value Heatmap", save_path=None):
    """
    Plot a heatmap of max Q-values per grid cell.
    """
    heatmap = np.zeros((env.rows, env.cols))

    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r][c] == 1:
                heatmap[r, c] = np.nan  # walls
            else:
                s = State(r, c)
                heatmap[r, c] = agent.get_state_value(s)

    plt.figure(figsize=(6, 6))
    im = plt.imshow(heatmap, cmap="inferno")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # overlay goal and start
    plt.scatter(env.start.c, env.start.r, c="cyan", s=100, label="Start")
    plt.scatter(env.goal.c, env.goal.r, c="lime", s=120, label="Goal")

    plt.title(title)
    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(True)
    plt.legend(loc="upper right")

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()

from src.env_grid import State

def animate_agents_side_by_side(
    env,
    agents_dict,
    labels,
    save_path="outputs/policy_comparison.gif",
    interval=300,
):
    """
    Animate multiple agents side-by-side on the same environment.
    agents_dict: dict[label -> QLearningAgent]
    labels: list of labels in display order
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from src.env_grid import State

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    # Prepare per-panel artists
    agent_dots = []
    frames_per_agent = []

    for ax, label in zip(axes, labels):
        ax.imshow(env.grid, cmap="gray_r")
        ax.set_title(label)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        # Goal
        ax.plot(env.goal.c, env.goal.r, "go", markersize=12)

        # Agent dot
        dot, = ax.plot([], [], "ro", markersize=10)
        agent_dots.append(dot)

        # Build frames for this agent
        state = env.reset()
        done = False
        frames = [State(state.r, state.c)]

        # Safety cap to avoid infinite loops
        steps = 0
        while not done and steps < env.max_steps:
            action = agents_dict[label].best_action(state)
            state, _, done, _ = env.step(action)
            frames.append(State(state.r, state.c))
            steps += 1

        frames_per_agent.append(frames)

    max_len = max(len(f) for f in frames_per_agent)

    def update(frame_idx):
        artists = []
        for dot, frames in zip(agent_dots, frames_per_agent):
            idx = min(frame_idx, len(frames) - 1)
            s = frames[idx]
            dot.set_data([s.c], [s.r])
            artists.append(dot)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_len,
        interval=interval,
        blit=True,
        repeat=False,
    )

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ani.save(
        save_path.replace(".gif", ".mp4"),
        writer="ffmpeg",
        fps=3
    )
    

    plt.show(block=False)
    plt.pause(0.2)
    plt.close()