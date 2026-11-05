# %% Constrói o MDP de cada k e guarda em cache/
import os

import numpy as np
import scipy.sparse as sp

from cleaning_robot.env import SMALL_ROOM, CleaningEnv
from cleaning_robot.model import TabularModel

SALA = SMALL_ROOM
K_LISTA = [0, 1, 2, 4, 6, 10]

os.makedirs("cache", exist_ok=True)

for k in K_LISTA:
    if os.path.exists(f"cache/tempo_k{k}.npy"):
        print(f"k={k}: já existe, pulando")
        continue
    env = CleaningEnv(room=SALA, p=0.2, k=k)
    model = TabularModel(env)
    for a in range(5):
        sp.save_npz(f"cache/P_k{k}_a{a}.npz", model.P[a])
    np.save(f"cache/R_k{k}.npy", model.R)
    np.save(f"cache/tempo_k{k}.npy", model.build_time)  # salvo por último: marca o k como pronto
    del model
