"""Tabular Q-learning: model-free control from sampled transitions only."""

from __future__ import annotations

import time

import numpy as np

from .env import CleaningEnv, State
from .model import n_states, state_index
from .policy import PolicyMixin

EPSILON = 0.1  # política e-gulosa com e fixo


class QLearning(PolicyMixin):
    """Tabular Q-learning.

    Learns only from interaction with the environment: env.random_init, env.next_state_sample
    and env.reward. Do not use the model (P and R). state_index(env, state) from
    cleaning_robot.model gives a row of the Q table for each state.

    act() must return the greedy action of the learned Q.
    """

    def __init__(self, env: CleaningEnv, gamma: float, n_steps: int, seed=None, q_init: float = 0.0):
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Q(s, a) inicial; nas aulas, Q(s, a) = 0 para todo par (s, a)
        Q = np.full((n_states(env), 5), q_init)
        m = np.zeros((n_states(env), 5), dtype=np.int32)  # m(s, a): número de visitas ao par

        # Ambiente sem estado terminal: escolhe s_0 aleatoriamente e segue uma única trajetória
        state = env.random_init(rng)
        s = state_index(env, state)
        for t in range(n_steps):
            # Escolhe a_t pela política e-gulosa usando Q(s_t, a)
            if rng.random() < EPSILON:
                a = int(rng.integers(5))
            else:
                a = int(Q[s].argmax())

            # Observa (r_t, s_{t+1})
            r = env.reward(state)
            state_prox = env.next_state_sample(state, a, rng)
            s_prox = state_index(env, state_prox)

            # Q(s_t, a_t) = Q(s_t, a_t) + alpha_m (r_t + gamma max_a Q(s_{t+1}, a) - Q(s_t, a_t)), alpha_m = 1/m
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
