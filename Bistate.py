import random
import time

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit
from tqdm import trange


# Initializing the agent's strategy
@jit(nopython=True)
def initialize_strategy(total_attributes,size):
    for i in range(size):
        if random.random() > 0.5:
            total_attributes[0][i] = 0
        else:
            total_attributes[0][i] = 1
# Initializing the agent's age
@jit(nopython=True)
def initialize_age(total_attributes,size):
    for i in range(size):
        total_attributes[1][i] = 0
# Calculate rewards
@jit(nopython=True)
def update_reward(total_attributes, i, r):
    Px = 0
    neighbors = neighborsArray[i]
    for j in neighbors:
        if total_attributes[0][i] == 1 and total_attributes[0][j] == 1:  # 都合作
            Px += 1
        elif total_attributes[0][i] == 1 and total_attributes[0][j] == 0:  # 玩家合作，邻居背叛
            Px -= r
        elif total_attributes[0][i] == 0 and total_attributes[0][j] == 1:  # 玩家背叛，邻居合作
            Px += 1 + r
        elif total_attributes[0][i] == 0 and total_attributes[0][j] == 0:  # 都背叛
            Px += 0
    return Px
# Update actions and Q-table
@jit(nopython=True)
def update(total_attributes, state_num, i, epsilon, alpha, gamma, Q_table, r):
    if total_attributes[1][i] < state_num:
        state = total_attributes[0][i]
    else:
        state = total_attributes[0][i] + 2
    old_strategy = total_attributes[0][i]
    # Select an action at probability epsilon
    if random.random() <= epsilon:
        if random.random() < 0.5:
            action = 1
            total_attributes[0][i] = 1
        else:
            action = 0
            total_attributes[0][i] = 0
    # With probability 1-epsilon, select the action corresponding to the maximum value in the Q table row containing the current state.
    else:
        if Q_table[i][state][0] == Q_table[i][state][1]:
            if random.random() > 0.5:
                action = 1
                total_attributes[0][i] = 1
            else:
                action = 0
                total_attributes[0][i] = 0
        elif Q_table[i][state][0] > Q_table[i][state][1]:
            total_attributes[0][i] = 0
            action = 0
        else:
            action = 1
            total_attributes[0][i] = 1
    ri = update_reward(total_attributes, i, r)
    # If the current strategy is maintained and the age threshold is not reached, the strategy age increases by one, and the status remains unchanged.
    if action == old_strategy:
        total_attributes[1][i] += 1
        if total_attributes[1][i] < state_num:
            max_Q_value = np.max(Q_table[i][action])
            Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
        # If the current strategy is maintained and the age threshold is reached, the strategy age increases by one, and transfers to state DH/CH.
        else:
            if action == 0:
                max_Q_value = np.max(Q_table[i][2])
                Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
            else:
                max_Q_value = np.max(Q_table[i][3])
                Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
    else:
        total_attributes[1][i] = 0
        max_Q_value = np.max(Q_table[i][action])
        Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))

@jit(nopython=True)
def fraction_of_cooperators(total_attributes,size):
    cooperator_num = 0
    for i in range(size):
        if total_attributes[0][i] == 1:
            cooperator_num += 1
    return cooperator_num / size

@jit(nopython=True)
def MCS(total_attributes, Q_table, epsilon, alpha, gamma, size, steps, state_num, r):
    # Monte Carlo simulation
    temp = 0
    for step in range(steps):
        for _ in range(size):
            i = random.randint(0, size - 1)
            update(total_attributes, state_num, i, epsilon, alpha, gamma, Q_table, r)
        if step >= steps - 100:
            temp += fraction_of_cooperators(total_attributes,size)
    return temp / 100

if __name__ == '__main__':
    L = 100
    size = L * L
    steps = 500000
    epsilon = 0.02
    alpha = 0.8
    gamma = 0.8
    r = []
    for i in range(21):
        r.append(i*0.01)
    state_num = [8]
    avg = 10

    #初始化网络结构
    neighborsArray = []
    G = nx.grid_2d_graph(int(L), int(L), periodic=True)
    for i in G.nodes():
        temp = []
        for j in list(G.neighbors(i)):
            temp.append(j[0] * L + j[1])
        neighborsArray.append(temp)
    neighborsArray = np.asarray(neighborsArray, dtype='int32')

    # 0:strategy 1:strategy_age
    total_attributes = np.zeros((2, size), dtype='int32')

    data = np.zeros((len(state_num), len(r)), dtype='float32')
    for i in range(len(state_num)):
        for j in trange(len(r)):
            for k in range(avg):
                initialize_strategy(total_attributes, size)
                initialize_age(total_attributes, size)
                # Q_table[0][0]:Q(D0,D) Q_table[0][1]:Q(D0,C) Q_table[1][0]:Q(C0,D) Q_table[1][1]:Q(C0,C)
                # Q_table[2][0]:Q(DH,D) Q_table[2][1]:Q(DH,C) Q_table[3][0]:Q(CH,D) Q_table[3][1]:Q(CH,C)
                Q_table = np.array([np.zeros((4, 2)) for _ in range(size)])
                data[i][j] += MCS(total_attributes, Q_table, epsilon, alpha, gamma, size, steps, state_num[i], r[j])
            data[i][j] /= avg
            print(f"state_num={state_num[i]} r={r[j]} cooperation_rate={data[i][j]}")

    row_names = [f"{s}" for s in state_num]
    col_names = [f"{val}" for val in r]
    df = pd.DataFrame(data, index=row_names, columns=col_names)
    csv_file_path = f'basedir/file_name'
    df.to_csv(csv_file_path, encoding='utf_8_sig')