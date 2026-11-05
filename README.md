# Exercício 1: robô de limpeza

O enunciado está no PDF do exercício.

## Instalação

```bash
uv sync
```

## O que é fornecido

- `cleaning_robot/env.py`: o ambiente (sala, cachorro, robô) com a interface do MDP:
  `reward`, `next_state_probs`, `next_state_sample`, `random_init`.
- `cleaning_robot/model.py`: enumeração dos estados (`all_states`, `state_index`) e o modelo
  em matrizes esparsas (`TabularModel`: `P[a]` e `R`).
- `cleaning_robot/policy.py`: `PolicyMixin`, `GreedyPolicy` e `RandomPolicy`.
- `cleaning_robot/animate.py`: animação de uma política (`animate`) e simulação (`rollout`).

## O que implementar

- `cleaning_robot/dp.py`: `policy_evaluation`, `ValueIteration` e `PolicyIteration`.
- `cleaning_robot/qlearning.py`: `QLearning`.

## Exemplo

```python
from cleaning_robot.env import CleaningEnv
from cleaning_robot.model import TabularModel
from cleaning_robot.policy import GreedyPolicy
from cleaning_robot.animate import animate

env = CleaningEnv(k=2)
model = TabularModel(env)  # P e R; para k=10 leva ~2 min e ~1.6 GB
animate(GreedyPolicy(env), env, steps=80, seed=3, filename="greedy.gif")
```
