import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from models import QNetwork

def encode_observation(obs, observation_space):
    """
    Convierte observaciones discretas a one-hot
    y observaciones continuas a vectores float32.
    """
    if hasattr(observation_space, "n"):
        index = int(obs) - int(
            getattr(observation_space, "start", 0)
        )

        vector = np.zeros(
            observation_space.n,
            dtype=np.float32
        )
        vector[index] = 1.0

        return vector

    return np.asarray(
        obs,
        dtype=np.float32
    ).reshape(-1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated):
        self.memory.append((
            state.copy(),
            action,
            reward,
            next_state.copy(),
            terminated
        ))

    def sample(self, batch_size, rng):
        indices = rng.choice(
            len(self.memory),
            size=batch_size,
            replace=False
        )

        transitions = [
            self.memory[int(i)]
            for i in indices
        ]

        states, actions, rewards, next_states, terminated = zip(
            *transitions
        )

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(terminated, dtype=np.float32)
        )

    def __len__(self):
        return len(self.memory)


def train_dqn(
    env,
    num_episodes=500,
    capacity=10_000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    epsilon_max=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.005,
    target_update_freq=500,
    seed=42,
    tracked_states=None,
    device="cpu",
    eval_env=None,
    eval_interval=50,
    eval_episodes=30,
    eval_seed=10000,
    checkpoint_path=None,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    observation_space = env.observation_space

    if hasattr(observation_space, "n"):
        input_dim = observation_space.n
    else:
        input_dim = int(np.prod(observation_space.shape))

    output_dim = env.action_space.n

    # Online Q-network
    online_net = QNetwork(
        input_dim,
        output_dim
    ).to(device)

    # Target Q-network
    target_net = QNetwork(
        input_dim,
        output_dim
    ).to(device)

    target_net.load_state_dict(
        online_net.state_dict()
    )
    target_net.eval()

    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=lr
    )

    buffer = ReplayBuffer(capacity)

    rewards = []
    lengths = []
    epsilons = []
    episode_times = []

    episode_losses = []
    losses = []
    loss_steps = []

    q_history = {
        name: []
        for name in (tracked_states or {})
    }

    global_steps = 0
    start_time = time.perf_counter()

    best_checkpoint_reward = -float("inf")
    best_checkpoint_episode = None

    checkpoint_episodes = []
    checkpoint_rewards = []

    if eval_env is not None and checkpoint_path is None:
        raise ValueError(
            "A checkpoint_path is required when using eval_env."
        )

    if eval_env is not None and eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")

    for episode in range(num_episodes):

        obs, _ = env.reset(
            seed=seed if episode == 0 else None
        )

        state = encode_observation(
            obs,
            observation_space
        )

        epsilon = epsilon_min + (
            epsilon_max - epsilon_min
        ) * np.exp(-epsilon_decay * episode)

        total_reward = 0.0
        steps = 0
        losses_this_episode = []
        done = False

        while not done:

            # Selección epsilon-greedy
            if rng.random() < epsilon:
                action = int(
                    rng.integers(output_dim)
                )
            else:
                with torch.no_grad():
                    q = online_net(
                        torch.as_tensor(
                            state,
                            dtype=torch.float32,
                            device=device
                        )
                    )

                    action = int(q.argmax().item())

            # Interacción con el entorno
            next_obs, reward, terminated, truncated, _ = (
                env.step(action)
            )

            next_state = encode_observation(
                next_obs,
                observation_space
            )

            # Guardar transición
            buffer.add(
                state,
                action,
                float(reward),
                next_state,
                terminated
            )

            done = terminated or truncated

            state = next_state
            steps += 1
            global_steps += 1
            total_reward += float(reward)

            # Entrenamiento con minibatch
            if len(buffer) >= batch_size:

                batch = buffer.sample(
                    batch_size,
                    rng
                )

                (
                    b_states,
                    b_actions,
                    b_rewards,
                    b_next,
                    b_terminated
                ) = batch

                states_t = torch.as_tensor(
                    b_states,
                    dtype=torch.float32,
                    device=device
                )

                actions_t = torch.as_tensor(
                    b_actions,
                    dtype=torch.int64,
                    device=device
                )

                rewards_t = torch.as_tensor(
                    b_rewards,
                    dtype=torch.float32,
                    device=device
                )

                next_t = torch.as_tensor(
                    b_next,
                    dtype=torch.float32,
                    device=device
                )

                terminated_t = torch.as_tensor(
                    b_terminated,
                    dtype=torch.float32,
                    device=device
                )

                # Q(s,a) para la acción ejecutada
                q_selected = online_net(
                    states_t
                ).gather(
                    1,
                    actions_t[:, None]
                ).squeeze(1)

                # Bellman target usando target network
                with torch.no_grad():
                    q_next = target_net(
                        next_t
                    ).max(dim=1).values

                    targets = rewards_t + gamma * (
                        1.0 - terminated_t
                    ) * q_next

                # Mean squared error
                loss = F.mse_loss(
                    q_selected,
                    targets
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))
                loss_steps.append(global_steps)

                losses_this_episode.append(
                    float(loss.item())
                )

            # Actualizar target network cada C pasos
            if global_steps % target_update_freq == 0:
                target_net.load_state_dict(
                    online_net.state_dict()
                )

        # Registrar métricas del episodio
        rewards.append(total_reward)
        lengths.append(steps)
        epsilons.append(epsilon)

        episode_times.append(
            time.perf_counter() - start_time
        )

        episode_losses.append(
            float(np.mean(losses_this_episode))
            if losses_this_episode else np.nan
        )

        # Registrar valores Q para los gráficos
        if tracked_states:
            with torch.no_grad():
                for label, tracked_obs in tracked_states.items():

                    vec = encode_observation(
                        tracked_obs,
                        observation_space
                    )

                    q_vals = online_net(
                        torch.as_tensor(
                            vec,
                            dtype=torch.float32,
                            device=device
                        )
                    )

                    q_history[label].append(
                        q_vals.cpu().numpy().copy()
                    )

        # =====================================================
        # NUEVO: Evaluación periódica y guardado de checkpoints
        # =====================================================

        if eval_env is not None and (
            (episode + 1) % eval_interval == 0
            or episode + 1 == num_episodes
        ):

            evaluation = evaluate_dqn(
                eval_env,
                online_net,
                num_episodes=eval_episodes,
                epsilon=0.0,
                seed=eval_seed,
                device=device
            )

            score = evaluation["mean_reward"]

            checkpoint_episodes.append(episode + 1)
            checkpoint_rewards.append(score)

            # Guardar la mejor política encontrada
            if score > best_checkpoint_reward:

                best_checkpoint_reward = score
                best_checkpoint_episode = episode + 1

                torch.save(
                    online_net.state_dict(),
                    checkpoint_path
                )

            # Volver al modo entrenamiento
            online_net.train()

            print(
                f"Episode {episode + 1} | "
                f"Eval reward: {score:.2f} | "
                f"Best: {best_checkpoint_reward:.2f} "
                f"(episode {best_checkpoint_episode})"
            )

    history = {
        "rewards": np.asarray(rewards),
        "lengths": np.asarray(lengths),
        "epsilons": np.asarray(epsilons),
        "episode_losses": np.asarray(episode_losses),
        "losses": np.asarray(losses),
        "loss_steps": np.asarray(loss_steps),
        "q_values": {
            key: np.stack(values)
            for key, values in q_history.items()
        },
        "wall_time": time.perf_counter() - start_time,
        "episode_times": np.asarray(episode_times),
        "checkpoint_episodes": np.asarray(checkpoint_episodes),
        "checkpoint_rewards": np.asarray(checkpoint_rewards),
        "best_checkpoint_episode": best_checkpoint_episode,
        "best_checkpoint_reward": best_checkpoint_reward,
    }

    return online_net, history


def evaluate_dqn(
    env,
    policy,
    num_episodes=100,
    epsilon=0.0,
    seed=1234,
    device="cpu",
):
    rng = np.random.default_rng(seed)

    returns = []
    lengths = []

    policy.eval()

    for episode in range(num_episodes):

        obs, _ = env.reset(
            seed=seed + episode
        )

        state = encode_observation(
            obs,
            env.observation_space
        )

        done = False
        total_reward = 0.0
        steps = 0

        while not done:

            if rng.random() < epsilon:
                action = int(
                    rng.integers(env.action_space.n)
                )
            else:
                with torch.no_grad():
                    q = policy(
                        torch.as_tensor(
                            state,
                            dtype=torch.float32,
                            device=device
                        )
                    )

                    action = int(q.argmax().item())

            obs, reward, terminated, truncated, _ = env.step(
                action
            )

            state = encode_observation(
                obs,
                env.observation_space
            )

            done = terminated or truncated

            total_reward += float(reward)
            steps += 1

        returns.append(total_reward)
        lengths.append(steps)

    return {
        "mean_reward": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "rewards": np.asarray(returns),
        "lengths": np.asarray(lengths),
    }