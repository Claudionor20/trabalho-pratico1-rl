"""Tabular model: every state gets an integer index and the dynamics become sparse matrices.

Only the public MDP interface is used (State, env.cells, env.reward, env.next_state_probs).
"""

from __future__ import annotations

import itertools
import time
from array import array
from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp

from .env import ACTIONS, State


def n_states(env) -> int:
    F = len(env.cells)
    return 2 * F * F * 4 * (F + 1)


def all_states(env) -> Iterator[State]:
    """Every state (reachable or not), in index order."""
    cells = env.cells
    for playing, robot, dog, face, mess in itertools.product(
        (False, True), cells, cells, range(4), [None, *cells]
    ):
        yield State(robot, dog, face, playing, mess)


def state_index(env, state: State) -> int:
    """Position of `state` in all_states(env): mixed-radix number with digits
    (playing, robot cell, dog cell, face, mess cell + 1)."""
    F, cell = len(env.cells), env.index
    mess = 0 if state.mess_pos is None else cell[state.mess_pos] + 1
    return (
        ((int(state.dog_playing) * F + cell[state.robot_pos]) * F + cell[state.dog_pos]) * 4 + state.dog_face
    ) * (F + 1) + mess


def build_transition_matrices(env) -> list[sp.csr_matrix]:
    """P[a][i, j] = Pr(next state j | state i, action a), one sparse matrix per action."""
    n = n_states(env)
    rows = [array("l") for _ in ACTIONS]
    cols = [array("l") for _ in ACTIONS]
    vals = [array("d") for _ in ACTIONS]
    for i, s in enumerate(all_states(env)):
        for a in ACTIONS:
            for s2, pr in env.next_state_probs(s, a).items():
                rows[a].append(i)
                cols[a].append(state_index(env, s2))
                vals[a].append(pr)
    return [
        sp.csr_matrix(
            (np.frombuffer(v), (np.frombuffer(r, dtype=np.int64), np.frombuffer(c, dtype=np.int64))), shape=(n, n)
        )
        for r, c, v in zip(rows, cols, vals)
    ]


def reward_vector(env) -> np.ndarray:
    return np.fromiter((env.reward(s) for s in all_states(env)), float, count=n_states(env))


class TabularModel:
    """The whole MDP as arrays: P[a] (n x n sparse) and R (n,)."""

    def __init__(self, env, P: list[sp.csr_matrix] | None = None, R: np.ndarray | None = None, verbose: bool = True):
        self.env = env
        self.n = n_states(env)
        if P is not None and R is not None:
            self.P, self.R = P, R
            return
        t0 = time.perf_counter()
        self.P = build_transition_matrices(env)
        self.R = reward_vector(env)
        self.build_time = time.perf_counter() - t0
        if verbose:
            nnz = sum(m.nnz for m in self.P)
            print(f"model: {self.n:,} states, {nnz:,} transitions, built in {self.build_time:.1f}s")

    def state_index(self, state: State) -> int:
        return state_index(self.env, state)
