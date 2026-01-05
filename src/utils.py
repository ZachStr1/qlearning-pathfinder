def rollout_policy(env, agent):
    from src.env_grid import State

    state = env.reset()
    done = False
    path = [State(state.r, state.c)]

    for _ in range(env.max_steps):
        action = agent.choose_action(state, epsilon_override=0.20)
        next_state, reward, done, _ = env.step(action)
        path.append(State(next_state.r, next_state.c))
        state = next_state

        if done and reward > 0:
            return path, True  # success

    return path, False  # failed to reach goal