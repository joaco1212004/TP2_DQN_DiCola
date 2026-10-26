import time
import numpy as np

def train_q_learning(
    env,
    num_episodes=20000,
    alpha=0.1,
    gamma=0.99,
    epsilon_max=1.0,
    epsilon_min=0.05,
    decay=0.0002,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Inicializar Q(s,a) = 0
    q_table = np.zeros((n_states, n_actions))

    rewards = []
    lengths = []
    epsilons = []

    start_time = time.perf_counter()

    for episode in range(num_episodes):

        state, _ = env.reset(
            seed=seed if episode == 0 else None
        )

        epsilon = epsilon_min + (
            epsilon_max - epsilon_min
        ) * np.exp(-decay * episode)

        total_reward = 0.0
        steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Epsilon-greedy
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                q_values = q_table[state]

                # Desempatar aleatoriamente entre
                # acciones con el mismo Q-value
                best_actions = np.flatnonzero(
                    np.isclose(q_values, q_values.max())
                )
                action = int(rng.choice(best_actions))

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            # Bellman target
            if terminated:
                td_target = reward
            else:
                td_target = (
                    reward
                    + gamma * np.max(q_table[next_state])
                )

            # Q-learning update
            q_table[state, action] += alpha * (
                td_target - q_table[state, action]
            )

            state = next_state

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        epsilons.append(epsilon)

    elapsed_time = time.perf_counter() - start_time

    history = {
        "rewards": np.array(rewards),
        "lengths": np.array(lengths),
        "epsilons": np.array(epsilons),
        "wall_time": elapsed_time,
    }

    return q_table, history


def evaluate_q_learning(
    env,
    q_table,
    num_episodes=100,
    epsilon=0.0,
    seed=1234,
):
    rng = np.random.default_rng(seed)

    rewards = []
    lengths = []

    for episode in range(num_episodes):

        state, _ = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):

            if rng.random() < epsilon:
                action = int(
                    rng.integers(env.action_space.n)
                )
            else:
                action = int(
                    np.argmax(q_table[state])
                )

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "success_rate": 100 * np.mean(rewards),
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
        "rewards": np.array(rewards),
    }