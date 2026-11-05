from cleaning_robot.env import CleaningEnv
from cleaning_robot.model import TabularModel, state_index
from cleaning_robot.policy import GreedyPolicy
from cleaning_robot.animate import animate, rollout

a = 0
seed = 0
env = CleaningEnv(k=2) # p = 0.2
s = env.random_init(seed=0) # estado inicial aleatorio
env.reward(s) # r(s)
env.next_state_probs(s, a) # {s’: P(s’|s,a)}
env.next_state_sample(s, a, seed) # s’ ~ P(.|s,a)
model = TabularModel(env) # model.P[a][i, j] = P(j|i,a), model.R[i] = r(i)
i = state_index(env, s) # linha de s em P e R
animate(GreedyPolicy(env), env, steps=80, filename="greedy.gif")