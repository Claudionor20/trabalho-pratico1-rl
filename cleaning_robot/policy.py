from __future__ import annotations

import numpy as np

from .env import ACTIONS, STAY, CleaningEnv, State


class PolicyMixin:
    def act(self, state: State, seed=None) -> int:
        raise NotImplementedError


class RandomPolicy(PolicyMixin):
    def act(self, state: State, seed=None) -> int:
        return int(np.random.default_rng(seed).choice(ACTIONS))


class GreedyPolicy(PolicyMixin):
    """Walks along a shortest path to the mess and waits when the room is clean.
    Ignores the dog completely."""

    def __init__(self, env: CleaningEnv):
        self.env = env

    def act(self, state: State, seed=None) -> int:
        if state.mess_pos is None:
            return STAY
        env = self.env
        robot, mess = env.index[state.robot_pos], env.index[state.mess_pos]
        moves = [(env.grid_dist[n, mess], a) for a, n in enumerate(env.move[robot]) if n >= 0]
        return min(moves)[1]
