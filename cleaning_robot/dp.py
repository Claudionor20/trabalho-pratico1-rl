"""Model-based methods (dynamic programming)."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .env import State
from .model import TabularModel, all_states
from .policy import PolicyMixin

ERRO = 1e-10  # critério de convergência: para quando max_s |V_k(s) - V_{k-1}(s)| < ERRO
MELHORA_MIN = 1e-6  # iteração por política: só troca a ação se Q melhora mais que isso


class TabularPolicy(PolicyMixin):
    """Deterministic policy stored as one action per state index."""

    def __init__(self, model: TabularModel, actions: np.ndarray):
        self.model = model
        self.actions = actions

    def act(self, state: State, seed=None) -> int:
        return int(self.actions[self.model.state_index(state)])


def tabulate(model: TabularModel, policy: PolicyMixin) -> np.ndarray:
    """Action of a deterministic policy in every state, in index order."""
    return np.fromiter((policy.act(s) for s in all_states(model.env)), np.int64, count=model.n)


def policy_evaluation(model: TabularModel, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Value V^pi of a policy, computed with the model.

    `policy` is either one action per state index (deterministic, shape (n,)) or the
    probability of each action in each state (shape (n, n_actions)).
    Returns V with V[i] the value of the state of index i.

    The model is model.P (one sparse n x n matrix per action, model.P[a][i, j] = P(j | i, a))
    and model.R (reward of each state).
    """
    # Política determinística: pi(a|s) = 1 para a ação escolhida e 0 para as outras
    if policy.ndim == 1:
        probs = np.eye(5)[policy]
    else:
        probs = policy

    # MDP + política = MRP: P_pi(s, s') = soma_a pi(a|s) p(s'|s, a)
    P_pi = sp.csr_matrix(model.P[0].shape)
    for a in range(5):
        P_pi = P_pi + sp.diags(probs[:, a]) @ model.P[a]

    # Método iterativo (programação dinâmica): V_k = r + gamma * P_pi V_{k-1}, com V_0 = 0
    V = np.zeros(model.n)
    erro = 1.0
    while erro > ERRO:
        V_novo = model.R + gamma * (P_pi @ V)
        erro = np.max(np.abs(V_novo - V))
        V = V_novo
    return V


class ValueIteration(TabularPolicy):
    """Value iteration.

    After construction, self.actions must hold the greedy action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        t0 = time.perf_counter()

        # Iteração por valor: V_{k+1}(s) = max_a ( r(s) + gamma * soma_o p(o|s, a) V_k(o) ), com V_0 = 0
        V = np.zeros(model.n)
        erro = 1.0
        n_iter = 0
        while erro > ERRO:
            Q = np.stack([model.R + gamma * (model.P[a] @ V) for a in range(5)], axis=1)
            V_novo = Q.max(axis=1)
            erro = np.max(np.abs(V_novo - V))
            V = V_novo
            n_iter += 1

        # Política ótima: argmax_a ( r(s) + gamma * soma_o p(o|s, a) V_f(o) )
        Q = np.stack([model.R + gamma * (model.P[a] @ V) for a in range(5)], axis=1)
        actions = Q.argmax(axis=1)

        self.V = V
        self.n_iter = n_iter
        self.time = time.perf_counter() - t0
        super().__init__(model, actions)


class PolicyIteration(TabularPolicy):
    """Policy iteration.

    After construction, self.actions must hold the action of every state index:
    call super().__init__(model, actions) at the end, and act() comes for free.
    """

    def __init__(self, model: TabularModel, gamma: float):
        t0 = time.perf_counter()

        # Política inicial determinística, sorteada (semente fixa para reprodutibilidade)
        rng = np.random.default_rng(0)
        actions = rng.integers(5, size=model.n)
        n_iter = 0
        while True:
            # Avaliação da política
            V = policy_evaluation(model, actions, gamma)
            n_iter += 1

            # Melhoria: Q(s, a) = r(s) + gamma * soma_o p(o|s, a) V(o) e pi_{i+1}(s) = argmax_a Q(s, a)
            Q = np.stack([model.R + gamma * (model.P[a] @ V) for a in range(5)], axis=1)

            # A política ótima não é única: só troca a ação quando a melhora é real,
            # para que empates (e erros numéricos da avaliação) não façam a política oscilar
            Q_atual = Q[np.arange(model.n), actions]
            troca = Q.max(axis=1) > Q_atual + MELHORA_MIN
            if not troca.any():
                break
            actions = np.where(troca, Q.argmax(axis=1), actions)

        self.V = V
        self.n_iter = n_iter
        self.time = time.perf_counter() - t0
        super().__init__(model, actions)
