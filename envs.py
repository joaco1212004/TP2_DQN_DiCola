import gymnasium as gym
import numpy as np

from gymnasium import spaces


class ConstantRewardEnv(gym.Env):
    """
    Un estado, una acción y un paso.
    Recompensa constante de +1.

    Q(0, 0) = 1
    """

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 0, 1.0, True, False, {}


class RandomObsBinaryRewardEnv(gym.Env):
    """
    Observación aleatoria: -1 o +1.
    Una sola acción y un solo paso.
    La recompensa coincide con la observación.

    Q(-1, 0) = -1
    Q(+1, 0) = +1
    """

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(1)

        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = float(
            self.np_random.choice([-1.0, 1.0])
        )

        obs = np.array(
            [self.state],
            dtype=np.float32
        )

        return obs, {}

    def step(self, action):
        reward = self.state

        obs = np.array(
            [self.state],
            dtype=np.float32
        )

        return obs, reward, True, False, {}


class TwoStepDelayedRewardEnv(gym.Env):
    """
    Dos estados y dos pasos.

    Estado 0:
        Recompensa 0 y transición al estado 1.

    Estado 1:
        Recompensa +1 y termina.

    Con gamma = 0.99:
        Q(0, 0) = 0.99
        Q(1, 0) = 1
    """

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)

        self.state = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = 0

        return self.state, {}

    def step(self, action):

        if self.state == 0:
            self.state = 1

            return 1, 0.0, False, False, {}

        else:
            return 1, 1.0, True, False, {}


class TwoActionRewardEnv(gym.Env):
    """
    Un estado y dos acciones.
    Un solo paso.

    Acción 0 -> recompensa +1
    Acción 1 -> recompensa -1

    Q(0, 0) = +1
    Q(0, 1) = -1
    """

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        return 0, {}

    def step(self, action):

        reward = 1.0 if action == 0 else -1.0

        return 0, reward, True, False, {}