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