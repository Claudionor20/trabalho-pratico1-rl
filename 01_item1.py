# %% Configuração
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cleaning_robot.animate import rollout
from cleaning_robot.dp import ERRO, policy_evaluation, tabulate
from cleaning_robot.env import SMALL_ROOM, CleaningEnv, Room
from cleaning_robot.model import TabularModel, state_index
from cleaning_robot.policy import GreedyPolicy, RandomPolicy

SALA = SMALL_ROOM
K_LISTA = [0, 1, 2, 4, 6, 10]
GAMMAS = [0.7, 0.99]
PASSOS_TRAJETORIA = 20000

os.makedirs("resultados", exist_ok=True)
os.makedirs("figuras", exist_ok=True)

# %% Avaliação das políticas gulosa e aleatória
# Médias sobre os estados iniciais: random_init só sorteia estados com o cachorro
# em modo normal, que são a primeira metade dos índices.
cabecalho = [
    "k", "gamma", "V_gulosa", "V_aleatoria", "dV_medio", "frac_gulosa_melhor",
    "S_gulosa", "S_aleatoria", "capturas_gulosa", "capturas_aleatoria",
    "tempo_aval_gulosa_s", "tempo_aval_aleatoria_s",
]
linhas = []

for k in K_LISTA:
    env = CleaningEnv(room=SALA, p=0.2, k=k)
    P = [sp.load_npz(f"cache/P_k{k}_a{a}.npz") for a in range(5)]
    R = np.load(f"cache/R_k{k}.npy")
    model = TabularModel(env, P=P, R=R, verbose=False)

    n = model.n
    metade = n // 2
    assert state_index(env, env.random_init(0)) < metade

    politica_gulosa = GreedyPolicy(env)
    politica_aleatoria = RandomPolicy()
    acoes_gulosa = tabulate(model, politica_gulosa)
    probs_aleatoria = np.full((n, 5), 0.2)

    # Trajetória longa: -S_k é a fração de passos com a sala suja
    S = {}
    capturas = {}
    for nome, politica in (("gulosa", politica_gulosa), ("aleatoria", politica_aleatoria)):
        estados, acoes, recompensas, eventos, reboots = rollout(env, politica, PASSOS_TRAJETORIA, seed=0)
        S[nome] = np.mean(recompensas)
        capturas[nome] = eventos.count("captured") / PASSOS_TRAJETORIA

    for gamma in GAMMAS:
        t0 = time.perf_counter()
        V_gulosa = policy_evaluation(model, acoes_gulosa, gamma)
        t_gulosa = time.perf_counter() - t0

        t0 = time.perf_counter()
        V_aleatoria = policy_evaluation(model, probs_aleatoria, gamma)
        t_aleatoria = time.perf_counter() - t0

        np.savez_compressed(
            f"resultados/V_k{k}_g{gamma}.npz",
            gulosa=V_gulosa.astype(np.float32),
            aleatoria=V_aleatoria.astype(np.float32),
        )

        # diferença estado a estado (abaixo de 1e-6 conta como empate)
        dV = (V_gulosa - V_aleatoria)[:metade]
        linhas.append([
            k, gamma, V_gulosa[:metade].mean(), V_aleatoria[:metade].mean(), dV.mean(), (dV > 1e-6).mean(),
            S["gulosa"], S["aleatoria"], capturas["gulosa"], capturas["aleatoria"], t_gulosa, t_aleatoria,
        ])
        print(f"k={k:2d} gamma={gamma}: V gulosa={linhas[-1][2]:8.3f}  aleatoria={linhas[-1][3]:8.3f}  "
              f"(avaliação: {t_gulosa:.0f}s e {t_aleatoria:.0f}s)")

with open("resultados/item1.csv", "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(cabecalho)
    escritor.writerows(linhas)

# %% Gráficos do item 1
tab = np.array(linhas)
col = {nome: i for i, nome in enumerate(cabecalho)}

# V médio contra k (linha tracejada: pior caso -1/(1-gamma))
fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
for eixo, gamma in zip(eixos, GAMMAS):
    sel = tab[:, col["gamma"]] == gamma
    eixo.plot(tab[sel, col["k"]], tab[sel, col["V_gulosa"]], "o-", label="gulosa")
    eixo.plot(tab[sel, col["k"]], tab[sel, col["V_aleatoria"]], "s-", label="aleatória")
    eixo.axhline(-1 / (1 - gamma), color="gray", linestyle="--", label="pior caso, -1/(1-γ)")
    eixo.set_title(f"γ = {gamma}")
    eixo.set_xlabel("k")
    eixo.set_ylabel("valor médio do estado inicial")
    eixo.legend()
plt.tight_layout()
plt.savefig("figuras/item1_V_vs_k.png", dpi=150)

# diferença média gulosa - aleatória contra k
fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
for eixo, gamma in zip(eixos, GAMMAS):
    sel = tab[:, col["gamma"]] == gamma
    eixo.plot(tab[sel, col["k"]], tab[sel, col["dV_medio"]], "o-")
    eixo.set_title(f"γ = {gamma}")
    eixo.set_xlabel("k")
    eixo.set_ylabel("V gulosa - V aleatória (média)")
plt.tight_layout()
plt.savefig("figuras/item1_diferenca_vs_k.png", dpi=150)

# simulação: capturas por passo e fração com sala suja
sel = tab[:, col["gamma"]] == GAMMAS[0]
fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
eixos[0].plot(tab[sel, col["k"]], tab[sel, col["capturas_gulosa"]], "o-", label="gulosa")
eixos[0].plot(tab[sel, col["k"]], tab[sel, col["capturas_aleatoria"]], "s-", label="aleatória")
eixos[0].set_xlabel("k")
eixos[0].set_ylabel("capturas por passo")
eixos[0].legend()
eixos[1].plot(tab[sel, col["k"]], -tab[sel, col["S_gulosa"]], "o-", label="gulosa")
eixos[1].plot(tab[sel, col["k"]], -tab[sel, col["S_aleatoria"]], "s-", label="aleatória")
eixos[1].set_xlabel("k")
eixos[1].set_ylabel("-S_k (fração de passos com sala suja)")
eixos[1].legend()
plt.tight_layout()
plt.savefig("figuras/item1_simulacao.png", dpi=150)

# %% 1c: método direto contra método iterativo
# Direto: resolve (I - gamma P_pi) V = r com splu. Iterativo: repete V = r + gamma P_pi V.
# Os dois últimos casos (sala do enunciado) usam bastante memória no método direto.
casos = [
    (Room.centered(5, 3, 1, 1), False, 2),
    (Room.centered(7, 3, 1, 1), False, 2),
    (Room.centered(9, 3, 3, 1), False, 2),
    (Room.centered(9, 4, 3, 1), False, 2),
    (SALA, True, 0),
    (SALA, True, 2),
]
gamma = 0.99

print(f"{'n':>8} {'k':>3} {'nnz(P_pi)':>11} {'iter':>6} {'t_iter(s)':>10} {'t_direto(s)':>12} "
      f"{'MB iter':>8} {'MB direto':>10} {'GB densa':>9} {'|dif|max':>9}")
for sala, do_cache, k in casos:
    env_c = CleaningEnv(room=sala, p=0.2, k=k)
    if do_cache:
        P_c = [sp.load_npz(f"cache/P_k{k}_a{a}.npz") for a in range(5)]
        R_c = np.load(f"cache/R_k{k}.npy")
        model_c = TabularModel(env_c, P=P_c, R=R_c, verbose=False)
    else:
        model_c = TabularModel(env_c, verbose=False)
    n_c = model_c.n
    acoes_c = tabulate(model_c, GreedyPolicy(env_c))
    probs_c = np.eye(5)[acoes_c]

    P_pi = sp.csr_matrix((n_c, n_c))
    for a in range(5):
        P_pi = P_pi + sp.diags(probs_c[:, a]) @ model_c.P[a]

    # Iterativo
    t0 = time.perf_counter()
    V_iter = np.zeros(n_c)
    n_iter = 0
    erro = 1.0
    while erro > ERRO:
        V_novo = model_c.R + gamma * (P_pi @ V_iter)
        erro = np.max(np.abs(V_novo - V_iter))
        V_iter = V_novo
        n_iter += 1
    t_iter = time.perf_counter() - t0

    # Direto
    A = (sp.identity(n_c, format="csc") - gamma * P_pi).tocsc()
    t0 = time.perf_counter()
    lu = spla.splu(A)
    V_direto = lu.solve(model_c.R)
    t_direto = time.perf_counter() - t0

    # memória: 12 bytes por não nulo; a matriz densa teria n^2 valores de 8 bytes
    mb_iter = P_pi.nnz * 12 / 1e6
    mb_direto = (lu.L.nnz + lu.U.nnz) * 12 / 1e6
    gb_densa = n_c**2 * 8 / 1e9
    print(f"{n_c:8d} {k:3d} {P_pi.nnz:11d} {n_iter:6d} {t_iter:10.2f} {t_direto:12.2f} "
          f"{mb_iter:8.1f} {mb_direto:10.1f} {gb_densa:9.1f} {np.abs(V_direto - V_iter).max():9.1e}")
    del lu, A, P_pi, model_c
