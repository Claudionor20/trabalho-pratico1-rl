# %% Configuração
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from cleaning_robot.dp import MELHORA_MIN, policy_evaluation
from cleaning_robot.env import SMALL_ROOM, CleaningEnv
from cleaning_robot.model import TabularModel, n_states
from cleaning_robot.qlearning import QLearning

SALA = SMALL_ROOM
CENARIOS = [(4, 0.7), (10, 0.99)]  # (k, gamma): ajustar depois de ver os itens 1 e 2
N_PASSOS = [1_000_000, 3_000_000, 10_000_000]
SEEDS = [0, 1, 2]
N_PASSOS_INICIAL = 3_000_000  # passos usados na comparação das inicializações de Q (3a)

os.makedirs("resultados", exist_ok=True)
os.makedirs("figuras", exist_ok=True)

# %% Treino e avaliação do Q-learning
# O treino usa só random_init, next_state_sample e reward. O modelo é usado depois,
# apenas para avaliar a política aprendida, pi(s) = argmax_a Q(s, a), com policy_evaluation.
cabecalho = [
    "k", "gamma", "q_init", "n_passos", "seed", "tempo_s", "V_medio", "perda_media",
    "frac_nao_otima", "frac_pares_visitados",
]
linhas = []
referencia = {}  # por cenário: valor médio da ótima e perda média da gulosa

for k, gamma in CENARIOS:
    env = CleaningEnv(room=SALA, p=0.2, k=k)
    P = [sp.load_npz(f"cache/P_k{k}_a{a}.npz") for a in range(5)]
    R = np.load(f"cache/R_k{k}.npy")
    model = TabularModel(env, P=P, R=R, verbose=False)
    n = model.n
    metade = n // 2
    todos = np.arange(n)

    V_otimo = np.load(f"resultados/otima_k{k}_g{gamma}.npz")["V"]
    V_gulosa = np.load(f"resultados/V_k{k}_g{gamma}.npz")["gulosa"].astype(np.float64)
    Q_otimo = np.stack([model.R + gamma * (model.P[a] @ V_otimo) for a in range(5)], axis=1)
    melhor = Q_otimo.max(axis=1)
    referencia[(k, gamma)] = [V_otimo[:metade].mean(), (V_otimo - V_gulosa)[:metade].mean()]

    # Q inicial = 0 (como nas aulas) com vários tamanhos de trajetória, e
    # Q inicial = -1/(1-gamma) (o menor valor possível) para comparar na 3a
    execucoes = [(0.0, m) for m in N_PASSOS] + [(-1 / (1 - gamma), N_PASSOS_INICIAL)]
    for q_init, n_passos in execucoes:
        for seed in SEEDS:
            ql = QLearning(env, gamma, n_passos, seed, q_init=q_init)
            acoes_q = ql.Q.argmax(axis=1)
            V_q = policy_evaluation(model, acoes_q, gamma)
            nao_otima = melhor - Q_otimo[todos, acoes_q] > MELHORA_MIN
            linhas.append([
                k, gamma, q_init, n_passos, seed, ql.time,
                V_q[:metade].mean(),
                (V_otimo - V_q)[:metade].mean(),
                nao_otima[:metade].mean(),
                # Condição de convergência: todo par (s, a) visitado. Medido nos estados iniciais,
                # porque muitos estados com o cachorro brincando nunca são alcançados
                (ql.visitas[:metade] > 0).mean(),
            ])
            print(f"k={k} gamma={gamma} q_init={q_init:.1f} passos={n_passos:,} seed={seed}: "
                  f"{ql.time:.0f}s, perda média={linhas[-1][7]:.3f}")
            del ql

with open("resultados/item3.csv", "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(cabecalho)
    escritor.writerows(linhas)

# %% 3a: Q inicial 0 contra Q inicial -1/(1-gamma), e qualidade contra passos de interação
tab = np.array(linhas)
col = {nome: i for i, nome in enumerate(cabecalho)}

print("\n3a: perda média (V° - V da política de Q) e pares (s, a) visitados, média sobre as sementes")
for k, gamma in CENARIOS:
    sel = (tab[:, col["k"]] == k) & (tab[:, col["gamma"]] == gamma) & (tab[:, col["n_passos"]] == N_PASSOS_INICIAL)
    zero = sel & (tab[:, col["q_init"]] == 0.0)
    menor = sel & (tab[:, col["q_init"]] < 0.0)
    print(f"  k={k} gamma={gamma}: Q0 = 0 -> perda {tab[zero, col['perda_media']].mean():.3f}, "
          f"pares {tab[zero, col['frac_pares_visitados']].mean():.1%} | "
          f"Q0 = -1/(1-γ) -> perda {tab[menor, col['perda_media']].mean():.3f}, "
          f"pares {tab[menor, col['frac_pares_visitados']].mean():.1%}")

fig, eixos = plt.subplots(1, len(CENARIOS), figsize=(5 * len(CENARIOS), 4))
for eixo, (k, gamma) in zip(np.atleast_1d(eixos), CENARIOS):
    base = (tab[:, col["k"]] == k) & (tab[:, col["gamma"]] == gamma) & (tab[:, col["q_init"]] == 0.0)
    medias = []
    desvios = []
    for m in N_PASSOS:
        valores = tab[base & (tab[:, col["n_passos"]] == m), col["perda_media"]]
        medias.append(valores.mean())
        desvios.append(valores.std())
    eixo.errorbar(N_PASSOS, medias, yerr=desvios, fmt="o-", capsize=3, label="Q-learning (média ± dp)")
    eixo.axhline(referencia[(k, gamma)][1], color="C1", linestyle="--", label="política gulosa")
    eixo.axhline(0, color="k", linewidth=0.8, label="política ótima")
    eixo.set_xscale("log")
    eixo.set_xlabel("passos de interação")
    eixo.set_ylabel("perda média V° - V")
    eixo.set_title(f"k = {k}, γ = {gamma}")
    eixo.legend()
plt.tight_layout()
plt.savefig("figuras/item3_perda_vs_passos.png", dpi=150)

# %% 3b: Q-learning contra os métodos baseados em modelo
with open("resultados/item2.csv") as f:
    item2 = list(csv.DictReader(f))

print("\n3b: comparação (Q-learning com o maior número de passos, média sobre as sementes)")
for k, gamma in CENARIOS:
    linha = [r for r in item2 if int(r["k"]) == k and float(r["gamma"]) == gamma][0]
    tempo_modelo = float(np.load(f"cache/tempo_k{k}.npy"))
    n = n_states(CleaningEnv(room=SALA, p=0.2, k=k))
    sel = ((tab[:, col["k"]] == k) & (tab[:, col["gamma"]] == gamma)
           & (tab[:, col["q_init"]] == 0.0) & (tab[:, col["n_passos"]] == N_PASSOS[-1]))
    print(f"\n  k={k}, gamma={gamma}")
    print(f"    construção do modelo: {tempo_modelo:.0f}s ({n * 5:,} consultas a next_state_probs)")
    print(f"    iteração por valor: {linha['iter_vi']} iterações, {float(linha['tempo_vi_s']):.0f}s, "
          f"{int(linha['iter_vi']) * n * 5:,} cálculos de Q(s, a), perda = 0")
    print(f"    iteração por política: {linha['iter_pi']} iterações, {float(linha['tempo_pi_s']):.0f}s, perda = 0")
    print(f"    Q-learning: {N_PASSOS[-1]:,} passos, {tab[sel, col['tempo_s']].mean():.0f}s, "
          f"perda média = {tab[sel, col['perda_media']].mean():.3f}, "
          f"valor médio = {tab[sel, col['V_medio']].mean():.3f} (ótima: {referencia[(k, gamma)][0]:.3f}), "
          f"pares (s, a) visitados = {tab[sel, col['frac_pares_visitados']].mean():.1%}, "
          f"estados com ação não ótima = {tab[sel, col['frac_nao_otima']].mean():.1%}")
    print(f"    política gulosa (referência): perda média = {referencia[(k, gamma)][1]:.3f}")
