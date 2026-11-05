"""Tabular Q-learning: model-free control from sampled transitions only."""


from __future__ import annotations

import time

import numpy as np

from .env import CleaningEnv, State
from .model import n_states, state_index
from .policy import PolicyMixin

EPSILON = 0.1


class QLearning(PolicyMixin):
    """Q-learning com política e-gulosa e alpha = 1/m."""

    def __init__(self, env: CleaningEnv, gamma: float, n_steps: int, seed=None, q_init: float = 0.0):
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        Q = np.full((n_states(env), 5), q_init)
        m = np.zeros((n_states(env), 5), dtype=np.int32)  # visitas a cada par (s, a)

        # sem estado terminal: um estado inicial sorteado e uma única trajetória
        state = env.random_init(rng)
        s = state_index(env, state)
        for t in range(n_steps):
            # e-gulosa
            if rng.random() < EPSILON:
                a = int(rng.integers(5))
            else:
                a = int(Q[s].argmax())

            r = env.reward(state)
            state_prox = env.next_state_sample(state, a, rng)
            s_prox = state_index(env, state_prox)

            m[s, a] += 1
            alpha = 1.0 / m[s, a]
            Q[s, a] += alpha * (r + gamma * Q[s_prox].max() - Q[s, a])

            state, s = state_prox, s_prox

        self.env = env
        self.Q = Q
        self.visitas = m
        self.n_steps = n_steps
        self.time = time.perf_counter() - t0

    def act(self, state: State, seed=None) -> int:
        return int(self.Q[state_index(self.env, state)].argmax())
