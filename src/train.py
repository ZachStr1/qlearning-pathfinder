from src.config import EnvConfig
from src.env_grid import GridWorld, State
from src.agent_qlearning import QLearningAgent
from src.visualize import plot_learning_curves, plot_final_path
import numpy as np



def train(num_episodes: int = 500, run_id: int | None = None, save_path_image: bool = False):

    snapshot_episodes = [50, 150, 300, num_episodes - 1]
    agent_snapshots = {}
    cfg = EnvConfig()

    
    env = GridWorld(
        rows=cfg.rows,
        cols=cfg.cols,
        step_penalty=cfg.step_penalty,
        wall_penalty=cfg.wall_penalty,
        goal_reward=cfg.goal_reward,
        max_steps=cfg.max_steps_per_episode,
        start=State(0, 0),
        goal=State(cfg.rows - 1, cfg.cols - 1),
    )

    env.randomize_walls(density=0.20)

    for r in range(env.rows):
        env.grid[r][0] = 0
    env.randomize_walls(density=0.20)

    agent = QLearningAgent()
    agent_snapshots = {}
    agent_snapshots["untrained"] = (agent.copy(), env.copy())

    episode_rewards = []
    episode_steps = []
    successes = []

    for ep in range(num_episodes):
        
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        successes.append(1 if reward == cfg.goal_reward else 0)

        if (ep + 1) % 50 == 0:
            avg_steps = np.mean(episode_steps[-50:])
            success_rate = np.mean(successes[-50:]) * 100
            print(
                f"Episode {ep+1:4d} | "
                f"Avg Steps (last 50): {avg_steps:6.1f} | "
                f"Success: {success_rate:5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        if ep in snapshot_episodes:
            agent_snapshots[f"ep_{ep}"] = (agent.copy(), env.copy())

        if save_path_image and run_id is not None:
            plot_final_path(
                env,
                agent,
                save_path=f"outputs/final_path_run_{run_id}.png",
                title=f"Final Path (Run {run_id})",
            )

        return episode_rewards, episode_steps, successes, agent, env

def run_multiple_mazes(num_runs=5, max_episodes=800):
    convergence_episodes = []

    for run in range(1, num_runs + 1):
        print(f"\n=== Maze {run}/{num_runs} ===")

        rewards, steps, successes, agent, env = train(
            num_episodes=max_episodes,
            run_id=run,
            save_path_image=True,
        )

        # convergence = first episode index where last-50 avg steps < 25
        window = 50
        converged_at = None
        for i in range(window, len(steps)):
            avg_steps = sum(steps[i - window:i]) / window
            if avg_steps < 25:
                converged_at = i
                break

        if converged_at is None:
            converged_at = max_episodes

        convergence_episodes.append(converged_at)

    return convergence_episodes

from src.visualize import plot_q_heatmap
from src.visualize import animate_agents_side_by_side



if __name__ == "__main__":
    import os
    from src.visualize import plot_convergence_distribution

    # clear outputs (png only)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(output_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(output_dir, f))

    # Run 5 mazes and save 5 final_path_run_*.png
    convergence_episodes = run_multiple_mazes(num_runs=5, max_episodes=500)

    # Save the histogram
    plot_convergence_distribution(convergence_episodes, save_path="outputs/convergence_distribution.png")

    print("\nSaved:")
    print("- outputs/final_path_run_1.png ... final_path_run_5.png")
    print("- outputs/convergence_distribution.png")