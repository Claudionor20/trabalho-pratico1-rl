# %% Configuração
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from cleaning_robot.animate import animate
from cleaning_robot.dp import MELHORA_MIN, PolicyIteration, TabularPolicy, ValueIteration, policy_evaluation, tabulate
from cleaning_robot.env import SMALL_ROOM, CleaningEnv
from cleaning_robot.model import TabularModel
from cleaning_robot.policy import GreedyPolicy

SALA = SMALL_ROOM
K_LISTA = [0, 1, 2, 4, 6, 10]
GAMMAS = [0.7, 0.99]
K_GIF, GAMMA_GIF = 10, 0.99  # cenário da animação

os.makedirs("resultados", exist_ok=True)
os.makedirs("figuras", exist_ok=True)

# %% Iteração por valor e iteração por política para cada (k, gamma)
cabecalho = ["k", "gamma", "iter_vi", "tempo_vi_s", "iter_pi", "tempo_pi_s", "max_dif_V_vi_pi"]
linhas = []

for k in K_LISTA:
    env = CleaningEnv(room=SALA, p=0.2, k=k)
    P = [sp.load_npz(f"cache/P_k{k}_a{a}.npz") for a in range(5)]
    R = np.load(f"cache/R_k{k}.npy")
    model = TabularModel(env, P=P, R=R, verbose=False)

    for gamma in GAMMAS:
        vi = ValueIteration(model, gamma)
        pi = PolicyIteration(model, gamma)
        np.savez_compressed(
            f"resultados/otima_k{k}_g{gamma}.npz",
            V=vi.V,
            acoes_vi=vi.actions.astype(np.int8),
            acoes_pi=pi.actions.astype(np.int8),
        )
        linhas.append([k, gamma, vi.n_iter, vi.time, pi.n_iter, pi.time, np.abs(vi.V - pi.V).max()])
        print(f"k={k:2d} gamma={gamma}: valor {vi.n_iter} it, {vi.time:.0f}s | política {pi.n_iter} it, {pi.time:.0f}s")

with open("resultados/item2.csv", "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(cabecalho)
    escritor.writerows(linhas)

# %% 2a: tempo e iterações contra k
tab = np.array(linhas)
col = {nome: i for i, nome in enumerate(cabecalho)}

fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
for eixo, gamma in zip(eixos, GAMMAS):
    sel = tab[:, col["gamma"]] == gamma
    eixo.plot(tab[sel, col["k"]], tab[sel, col["tempo_vi_s"]], "o-", label="iteração por valor")
    eixo.plot(tab[sel, col["k"]], tab[sel, col["tempo_pi_s"]], "s-", label="iteração por política")
    eixo.set_title(f"γ = {gamma}")
    eixo.set_xlabel("k")
    eixo.set_ylabel("tempo de convergência (s)")
    eixo.legend()
plt.tight_layout()
plt.savefig("figuras/item2_tempo_vs_k.png", dpi=150)

# Iterações: uma linha de painéis por método, porque as escalas são muito diferentes
fig, eixos = plt.subplots(2, 2, figsize=(10, 7))
for j, gamma in enumerate(GAMMAS):
    sel = tab[:, col["gamma"]] == gamma
    eixos[0, j].plot(tab[sel, col["k"]], tab[sel, col["iter_vi"]], "o-")
    eixos[0, j].set_title(f"γ = {gamma}")
    eixos[0, j].set_ylabel("iterações (iteração por valor)")
    eixos[1, j].plot(tab[sel, col["k"]], tab[sel, col["iter_pi"]], "s-", color="C1")
    eixos[1, j].set_ylabel("iterações (iteração por política)")
    for eixo in eixos[:, j]:
        eixo.set_xlabel("k")
        eixo.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figuras/item2_iteracoes_vs_k.png", dpi=150)

# %% 2b e 2c: comparações da política ótima
# Q(s, a) = r(s) + gamma * soma_o p(o|s, a) V°(o), com V° o valor ótimo do modelo k.
# Uma ação a é ótima em s quando Q(s, a) = max_a Q(s, a). Como a política ótima não é única,
# diferenças de Q menores que MELHORA_MIN são tratadas como empate.
# Médias e frações sobre os estados iniciais (primeira metade dos índices).
acoes_otimas = {}
for k in K_LISTA:
    for gamma in GAMMAS:
        acoes_otimas[(k, gamma)] = np.load(f"resultados/otima_k{k}_g{gamma}.npz")["acoes_vi"].astype(np.int64)

frac_dif = {gamma: np.zeros((len(K_LISTA), len(K_LISTA))) for gamma in GAMMAS}  # [modelo k, política de k']
perda_cruzada = {gamma: np.zeros((len(K_LISTA), len(K_LISTA))) for gamma in GAMMAS}

cabecalho_c = [
    "k", "gamma", "V_otima", "V_gulosa", "V_aleatoria", "perda_media_gulosa", "perda_max_gulosa",
    "frac_gulosa_subotima", "perda_media_aleatoria",
]
linhas_c = []

for i, k in enumerate(K_LISTA):
    env = CleaningEnv(room=SALA, p=0.2, k=k)
    P = [sp.load_npz(f"cache/P_k{k}_a{a}.npz") for a in range(5)]
    R = np.load(f"cache/R_k{k}.npy")
    model = TabularModel(env, P=P, R=R, verbose=False)
    n = model.n
    metade = n // 2
    todos = np.arange(n)
    acoes_gulosa = tabulate(model, GreedyPolicy(env))

    for gamma in GAMMAS:
        V_otimo = np.load(f"resultados/otima_k{k}_g{gamma}.npz")["V"]
        Q = np.stack([model.R + gamma * (model.P[a] @ V_otimo) for a in range(5)], axis=1)
        melhor = Q.max(axis=1)

        # 2b: política ótima de k' (coluna j) usada no modelo k (linha i)
        for j, k2 in enumerate(K_LISTA):
            nao_otima = melhor - Q[todos, acoes_otimas[(k2, gamma)]] > MELHORA_MIN
            frac_dif[gamma][i, j] = nao_otima[:metade].mean()
            # Avaliação da política de k' no modelo k: valor perdido em relação ao ótimo de k
            if k2 != k:
                V_cruzado = policy_evaluation(model, acoes_otimas[(k2, gamma)], gamma)
                perda_cruzada[gamma][i, j] = (V_otimo - V_cruzado)[:metade].mean()

        # 2c: ótima contra gulosa e aleatória (valores do item 1)
        dados = np.load(f"resultados/V_k{k}_g{gamma}.npz")
        V_gulosa = dados["gulosa"].astype(np.float64)
        V_aleatoria = dados["aleatoria"].astype(np.float64)
        gulosa_nao_otima = melhor - Q[todos, acoes_gulosa] > MELHORA_MIN
        linhas_c.append([
            k, gamma,
            V_otimo[:metade].mean(), V_gulosa[:metade].mean(), V_aleatoria[:metade].mean(),
            (V_otimo - V_gulosa)[:metade].mean(), (V_otimo - V_gulosa)[:metade].max(),
            gulosa_nao_otima[:metade].mean(),
            (V_otimo - V_aleatoria)[:metade].mean(),
        ])
    print(f"k={k}: comparações concluídas")

with open("resultados/item2_comparacao.csv", "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(cabecalho_c)
    escritor.writerows(linhas_c)

# Matrizes da 2b em CSV (linha = modelo k, coluna = política ótima de k')
with open("resultados/item2_mudanca_politica.csv", "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerow(["gamma", "medida", "k_modelo"] + [f"pol_k{k2}" for k2 in K_LISTA])
    for gamma in GAMMAS:
        for i, k in enumerate(K_LISTA):
            escritor.writerow([gamma, "frac_acao_nao_otima", k] + list(frac_dif[gamma][i]))
            escritor.writerow([gamma, "perda_media_valor", k] + list(perda_cruzada[gamma][i]))

# Mapas de calor da 2b: fração de estados e valor perdido
for medida, matrizes, rotulo in (("frac", frac_dif, "fração de estados com ação não ótima"),
                                 ("perda", perda_cruzada, "valor perdido médio, V° - V")):
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for eixo, gamma in zip(eixos, GAMMAS):
        m = matrizes[gamma]
        imagem = eixo.imshow(m, cmap="Blues")
        eixo.set_xticks(range(len(K_LISTA)), K_LISTA)
        eixo.set_yticks(range(len(K_LISTA)), K_LISTA)
        eixo.set_xlabel("política ótima de k'")
        eixo.set_ylabel("modelo k")
        eixo.set_title(f"γ = {gamma}")
        for a in range(len(K_LISTA)):
            for b in range(len(K_LISTA)):
                cor = "white" if m[a, b] > m.max() / 2 else "black"
                eixo.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", color=cor, fontsize=8)
        plt.colorbar(imagem, ax=eixo, label=rotulo)
    plt.tight_layout()
    plt.savefig(f"figuras/item2_mudanca_politica_{medida}.png", dpi=150)

# Gráfico da 2c: valor médio das três políticas contra k
tab_c = np.array(linhas_c)
col_c = {nome: i for i, nome in enumerate(cabecalho_c)}
fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
for eixo, gamma in zip(eixos, GAMMAS):
    sel = tab_c[:, col_c["gamma"]] == gamma
    eixo.plot(tab_c[sel, col_c["k"]], tab_c[sel, col_c["V_otima"]], "o-", label="ótima")
    eixo.plot(tab_c[sel, col_c["k"]], tab_c[sel, col_c["V_gulosa"]], "s-", label="gulosa")
    eixo.plot(tab_c[sel, col_c["k"]], tab_c[sel, col_c["V_aleatoria"]], "^-", label="aleatória")
    eixo.axhline(-1 / (1 - gamma), color="gray", linestyle="--", label="pior caso, -1/(1-γ)")
    eixo.set_title(f"γ = {gamma}")
    eixo.set_xlabel("k")
    eixo.set_ylabel("valor médio do estado inicial")
    eixo.legend()
plt.tight_layout()
plt.savefig("figuras/item2_comparacao_politicas.png", dpi=150)

# %% Animações: política ótima e gulosa, mesmo estado inicial (seed=0)
env = CleaningEnv(room=SALA, p=0.2, k=K_GIF)
P = [sp.load_npz(f"cache/P_k{K_GIF}_a{a}.npz") for a in range(5)]
R = np.load(f"cache/R_k{K_GIF}.npy")
model = TabularModel(env, P=P, R=R, verbose=False)

politica_otima = TabularPolicy(model, acoes_otimas[(K_GIF, GAMMA_GIF)])
animate(politica_otima, env, steps=100, seed=0, filename=f"figuras/otima_k{K_GIF}_g{GAMMA_GIF}.gif")
animate(GreedyPolicy(env), env, steps=100, seed=0, filename=f"figuras/gulosa_k{K_GIF}.gif")
