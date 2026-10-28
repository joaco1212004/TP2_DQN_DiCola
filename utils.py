import gymnasium as gym
import numpy as np

class NormalizeMountainCarObservation(gym.ObservationWrapper):
    """
    Normaliza posicion y velocidad al rango [-1, 1].
    No modifica las recompensas ni las transiciones.
    """

    def __init__(self, env):
        super().__init__(env)

        low = env.observation_space.low.astype(np.float32)
        high = env.observation_space.high.astype(np.float32)

        self.center = (high + low) / 2.0
        self.scale = (high - low) / 2.0

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)

        normalized = (obs - self.center) / self.scale

        return normalized.astype(np.float32)

class MountainCarPotentialShaping(gym.Wrapper):
    """
    Potential-based reward shaping para MountainCar.

    Incentiva cambios en altura y velocidad,
    manteniendo la recompensa original como base.
    """

    def __init__(self, env, gamma=0.99, scale=20.0):
        super().__init__(env)

        self.gamma = gamma
        self.scale = scale
        self.previous_obs = None

    def potential(self, obs):
        position, velocity = obs

        # Altura normalizada aproximadamente entre 0 y 1
        height = (np.sin(3.0 * position) + 1.0) / 2.0

        # Energia cinetica aproximada y normalizada
        kinetic = (velocity / 0.07) ** 2

        return self.scale * (
            height + 0.5 * kinetic - 1.5
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(
            seed=seed,
            options=options
        )

        self.previous_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        phi_current = self.potential(self.previous_obs)

        # El estado terminal tiene potencial cero
        phi_next = (
            0.0 if terminated
            else self.potential(obs)
        )

        shaped_reward = (
            reward
            + self.gamma * phi_next
            - phi_current
        )

        self.previous_obs = obs.copy()

        return (
            obs,
            float(shaped_reward),
            terminated,
            truncated,
            info
        )


class MountainCarSuccessCounter(gym.Wrapper):
    """Cuenta llegadas a la meta durante evaluacion."""

    def __init__(self, env):
        super().__init__(env)
        self.successes = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        if terminated:
            self.successes += 1

        return obs, reward, terminated, truncated, info