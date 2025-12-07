import random
import networkx as nx
import numpy as np
import pandas as pd
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
# Initializing the agent's Q-value
@jit(nopython=True)
def initialize_Q_table(Q_table):
    for i in range(size):
        for j in range(2):
            for k in range(2):
                Q_table[i][j][k] = 0
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
    age = total_attributes[1][i]
    strategy = total_attributes[0][i]
    state = int(age * 2 + strategy)
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
    if state % 2 == 0:
        if action == 0:
            # Persist with the defection strategy. If the age threshold is not reached, increment the strategy age by one and transition to the next state Dk+1.
            if total_attributes[1][i] + 1 < state_num:
                total_attributes[1][i] += 1
                if Q_table[i][state + 2][0] == -99 and Q_table[i][state + 3][0] == -99: #如果当前状态没有被开通过，那么赋初始值为0；否则，直接用原始Q值即可
                    Q_table[i][state + 2] = 0
                    Q_table[i][state + 3] = 0
                max_Q_value = np.max(Q_table[i][state + 2])
                Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
            # Persist with the defection strategy until the age threshold is reached. If the strategy age remains unchanged, remain in the current state.
            else:
                max_Q_value = np.max(Q_table[i][state])
                Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
        # Select cooperation strategy, set strategy age to 0, transition to state C0
        else:
            total_attributes[1][i] = 0
            max_Q_value = np.max(Q_table[i][1])
            Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
    else:
        if action == 1:
            # Persist with the cooperation strategy. If the age threshold is not reached, increment the strategy age by one and transition to the next state Ck+1.
            if total_attributes[1][i] + 1 < state_num:
                total_attributes[1][i] += 1
                if Q_table[i][state + 1][0] == -99 and Q_table[i][state + 2][0] == -99:
                    Q_table[i][state + 1] = 0
                    Q_table[i][state + 2] = 0
                max_Q_value = np.max(Q_table[i][state + 2])
                Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
            # Persist with the cooperation strategy until the age threshold is reached. If the strategy age remains unchanged, remain in the current state.
            else:
                max_Q_value = np.max(Q_table[i][state])
                Q_table[i][state][action] = Q_table[i][state][action] + (alpha * (ri + (gamma * max_Q_value) - Q_table[i][state][action]))
        # Select defection strategy, set strategy age to 0, transition to state D0
        else:
            total_attributes[1][i] = 0
            max_Q_value = np.max(Q_table[i][0])
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
    # 初始化参数
    L = 100
    size = L * L
    steps = 500000
    epsilon = 0.02
    alpha = 0.8
    gamma = 0.8
    r = []
    for i in range(21):
        r.append(i*0.01)
    state_num = 10
    avg = 10

    # Initialize Network Structure
    neighborsArray = []
    G = nx.grid_2d_graph(int(L), int(L), periodic=True)
    for i in G.nodes():
        temp = []
        for j in list(G.neighbors(i)):
            temp.append(j[0] * L + j[1])
        neighborsArray.append(temp)
    neighborsArray = np.asarray(neighborsArray, dtype='int32')

    # 0:strategy 1:strategy_age
    total_attributes = np.zeros((2, size), dtype='float32')

    data = np.asarray([0 for _ in range(len(r))], dtype='float32')
    for i in trange(len(r)):
        for k in range(avg):
            initialize_strategy(total_attributes, size)
            initialize_age(total_attributes, size)
            # Q_table[0][0]:Q(D0,D) Q_table[0][1]:Q(D0,C) Q_table[1][0]:Q(C0,D) Q_table[1][1]:Q(C0,C)
            # Q_table[2][0]:Q(D1,D) Q_table[2][1]:Q(D1,C) Q_table[3][0]:Q(C1,D) Q_table[3][1]:Q(C1,C)
            Q_table = np.full((size, state_num * 2, 2), -99.0, dtype=np.float32)
            initialize_Q_table(Q_table)
            data[i] += MCS(total_attributes, Q_table, epsilon, alpha, gamma, size, steps, state_num, r[i])
        data[i] /= avg
        print("r=",r[i],"data=",data[i])

    row_names = [f"{r[i]}" for i in range(len(r))]
    df = pd.DataFrame(data, index=row_names)
    bisedir = ''
    csv_file_path = f'bisedir/file_name'
    df.to_csv(csv_file_path, encoding='utf_8_sig')