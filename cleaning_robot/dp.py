"""Métodos baseados em modelo (programação dinâmica)."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .env import State
from .model import TabularModel, all_states
from .policy import PolicyMixin

ERRO = 1e-10  # para quando max |V_k - V_{k-1}| < ERRO
MELHORA_MIN = 1e-6  # diferenças de Q menores que isso contam como empate


class TabularPolicy(PolicyMixin):
    """Política determinística guardada como uma ação por estado."""

    def __init__(self, model: TabularModel, actions: np.ndarray):
        self.model = model
        self.actions = actions

    def act(self, state: State, seed=None) -> int:
        return int(self.actions[self.model.state_index(state)])


def tabulate(model: TabularModel, policy: PolicyMixin) -> np.ndarray:
    """Ação de uma política determinística em cada estado."""
    return np.fromiter((policy.act(s) for s in all_states(model.env)), np.int64, count=model.n)


def policy_evaluation(model: TabularModel, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Valor V de uma política: uma ação por estado ou matriz de probabilidades n x 5."""
    if policy.ndim == 1:
        probs = np.eye(5)[policy]
    else:
        probs = policy

    # P_pi(s, s') = soma_a pi(a|s) p(s'|s, a)
    P_pi = sp.csr_matrix(model.P[0].shape)
    for a in range(5):
        P_pi = P_pi + sp.diags(probs[:, a]) @ model.P[a]

    # V_k = r + gamma P_pi V_{k-1}, com V_0 = 0
    V = np.zeros(model.n)
    erro = 1.0
    while erro > ERRO:
        V_novo = model.R + gamma * (P_pi @ V)
        erro = np.max(np.abs(V_novo - V))
        V = V_novo
    return V


class ValueIteration(TabularPolicy):
    """Iteração por valor."""

    def __init__(self, model: TabularModel, gamma: float):
        t0 = time.perf_counter()

        # V_{k+1}(s) = max_a ( r(s) + gamma soma_o p(o|s, a) V_k(o) )
        V = np.zeros(model.n)
        erro = 1.0
        n_iter = 0
        while erro > ERRO:
            Q = np.stack([model.R + gamma * (model.P[a] @ V) for a in range(5)], axis=1)
            V_novo = Q.max(axis=1)
            erro = np.max(np.abs(V_novo - V))
            V = V_novo
            n_iter += 1

        # política: ação de maior Q com o V final
        Q = np.stack([model.R + gamma * (model.P[a] @ V) for a in range(5)], axis=1)
        actions = Q.argmax(axis=1)

        self.V = V
        self.n_iter = n_iter
        self.time = time.perf_counter() - t0
        super().__init__(model, actions)


class PolicyIteration(TabularPolicy):
    """Iteração por política."""

    def __init__(self, model: TabularModel, gamma: float):
        t0 = time.perf_counter()

        # política inicial sorteada
        rng = np.random.default_rng(0)
        actions = rng.integers(5, size=model.n)
        n_iter = 0
        while True:
            # avaliação
            V = policy_evaluation(model, actions, gamma)
            n_iter += 1

            # melhoria: troca a ação só se Q melhora mais que MELHORA_MIN (evita oscilar em empates)
            Q = np.stack([model.R + gamma * (model.P[a] @ V) for a in range(5)], axis=1)
            Q_atual = Q[np.arange(model.n), actions]
            troca = Q.max(axis=1) > Q_atual + MELHORA_MIN
            if not troca.any():
                break
            actions = np.where(troca, Q.argmax(axis=1), actions)

        self.V = V
        self.n_iter = n_iter
        self.time = time.perf_counter() - t0
        super().__init__(model, actions)
