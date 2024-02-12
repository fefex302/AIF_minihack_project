import time
import matplotlib.pyplot as plt
import IPython.display as display
import random
import numpy as np
from queue import PriorityQueue
from typing import List

N_ARR = np.array([0, 1])
S_ARR = np.array([0, -1])
W_ARR = np.array([-1, 0])
E_ARR = np.array([1, 0])
NE_ARR = N_ARR + E_ARR
SE_ARR = S_ARR + E_ARR
SW_ARR = S_ARR + W_ARR
NW_ARR = N_ARR + W_ARR
MOVES = [N_ARR, E_ARR, S_ARR, W_ARR, NE_ARR, SE_ARR, SW_ARR, NW_ARR]

N = 0
E = 1
S = 2
W = 3
NE = 3 + N + E
SE = 2 + S + E
SW = 1 + S + W
NW = 4 + N + W
ACTIONS = [N, E, S, W, NE, SE, SW, NW]

ACTION_NAMES = ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW']


def move_to_action(move: np.ndarray[int]) -> List[int]:
    if np.array_equal(move, N_ARR):
        return N
    elif np.array_equal(move, S_ARR):
        return S
    elif np.array_equal(move, W_ARR):
        return W
    elif np.array_equal(move, E_ARR):
        return E
    elif np.array_equal(move, NE_ARR):
        return NE
    elif np.array_equal(move, SE_ARR):
        return SE
    elif np.array_equal(move, NW_ARR):
        return NW
    elif np.array_equal(move, SW_ARR):
        return SW



def action_to_move(action: int) -> np.ndarray[int]:
    if action == N:
        return N_ARR
    elif action == S:
        return S_ARR
    elif action == W:
        return W_ARR
    elif action == E:
        return E_ARR
    elif action == NE:
        return NE_ARR
    elif action == SE:
        return SE_ARR
    elif action == NW:
        return NW_ARR
    elif action == SW:
        return SW_ARR


def action_to_string(action: int):
    return ACTION_NAMES[action]


#def random_step(env:MiniHackGoldRoom):
#    return env.mystep(action=random.sample(env.possible_actions(), 1)[0])
#
#
#def greedy_stair_step(env:MiniHackGoldRoom):
#    agent_point = np.array(env.get_agent_coords())
#    stair_point = np.array(env.get_stair_coords())
#    reachables = [agent_point + move for move in MOVES]
#    scores = []
#    for reachable_point, move in zip(reachables, ACTIONS):
#        if move in env.possible_actions():
#            scores.append(-np.linalg.norm(stair_point - reachable_point))
#        else:
#            scores.append(-np.inf)
#    move = ACTIONS[np.argmax(scores)]
#    return env.mystep(action=move)
#
##TBR
#def apply_greedy_strategy(env:MiniHackGoldRoom):
#    init_s, init_r = env.myreset()
#    states = [init_s]
#    rewards = [float(init_r)]
#    cumulative_rewards = []
#    state, reward, done, info = greedy_stair_step(env)
#    states.append(state)
#    for i in range(0, env.max_episode_steps):
#        if not done:
#            state, reward, done, info = greedy_stair_step(env)
#            rewards.append(reward)
#            cumulative_rewards.append(np.sum(rewards))
#            states.append(state)
#        else:
#            rewards.append(float(env.stair_score))
#            break
#    return states, rewards, cumulative_rewards, done, i


def show_episode(states, clear_output=True):
    image = plt.imshow(states[0]['pixel'][100:300, 500:750])
    for state in states[1:]:
        display.display(plt.gcf())
        if clear_output:
            display.clear_output(wait=True)
        image.set_data(np.array(state['pixel'])[100:300, 500:750])
        time.sleep(0.3)