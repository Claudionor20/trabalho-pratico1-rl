# Exercício 1: Robô de Limpeza e Cachorro
COS767, Tópicos Especiais em Aprendizado por Reforço (2026/3). O enunciado está em `enunciado/`.

## Estrutura

- `cleaning_robot/`: ambiente, modelo, políticas e animação (fornecidos), e as implementações
  - `dp.py`: `policy_evaluation`, `ValueIteration` e `PolicyIteration`
  - `qlearning.py`: `QLearning`
- `00_modelos.py`: constrói o MDP de cada k e guarda em `cache/`
- `01_item1.py`, `02_item2.py`, `03_item3.py`: experimentos de cada item
- `resultados/`: tabelas geradas pelos experimentos
- `figuras/`: gráficos e animações geradas pelos experimentos
- `relatorio/`: relatório em LaTeX e PDF

## Como rodar

```bash
uv sync
uv run python 00_modelos.py
uv run python 01_item1.py
uv run python 02_item2.py
uv run python 03_item3.py
```

Os scripts devem rodar nessa ordem, porque cada um usa resultados dos anteriores.
O `00_modelos.py` leva alguns minutos por k, e o modelo de k = 10 usa mais de 1 GB de memória.
O `02_item2.py` e o `03_item3.py` levam da ordem de uma hora cada.
