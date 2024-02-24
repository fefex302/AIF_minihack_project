import time
import matplotlib.pyplot as plt
import IPython.display as display
import random
import numpy as np
from queue import PriorityQueue
from typing import List, Tuple

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
DIAGONAL_ACTIONS = [NE, SE, SW, NW]

ACTION_NAMES = ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW']


class AllowedMovesFunction:

    def __init__(self):
        pass

    def __call__(self, state: dict):
        pass


class AllowedSimpleMovesFunction(AllowedMovesFunction):

    def __init__(
        self,
        width: int = None,
        height: int = None,
        to_avoid: List[Tuple[int, int]] = []
        ):

        self.width = width
        self.height = height
        self.to_avoid = to_avoid
    
    def __call__(self, state: dict) -> List[np.ndarray[int]]:
        return allowed_moves(
            width=self.width,
            height=self.height,
            state=state,
            to_avoid=self.to_avoid
        )


class AllowedCompositeMovesFunction(AllowedMovesFunction):

    def __init__(self):
        pass
    
    def __call__(self, state: dict) -> List[np.ndarray[int]]:
        return \
            [np.array(state['stair_coord']) - np.array(state['agent_coord'])] \
                + [np.array(g_coord) - np.array(state['agent_coord']) for g_coord in state['gold_coords'] if state['agent_coord'] != g_coord]

ALLOWED_SIMPLE_MOVES = AllowedSimpleMovesFunction()
ALLOWED_COMPOSITE_MOVES = AllowedCompositeMovesFunction()


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


def action_to_string(action: int) -> str:
    return ACTION_NAMES[action]


def show_episode(states: dict, clear_output: bool = True) -> None:
    image = plt.imshow(states[0]['pixel'][100:300, 500:750])
    for state in states[1:]:
        display.display(plt.gcf())
        if clear_output:
            display.clear_output(wait=True)
        image.set_data(np.array(state['pixel'])[100:300, 500:750])
        time.sleep(0.3)


def allowed_moves(width: int, height: int, state: dict, to_avoid: List[Tuple[int, int]] = []) -> List[np.ndarray[int]]:
    x, y = state['agent_coord']
    n = 0b1000
    s = 0b0100
    w = 0b0010
    e = 0b0001
    b = 0b0000
    nw = n | w
    ne = n | e
    sw = s | w
    se = s | e
    moves = []
    if x-1 >= 0:
        b |= w
        moves.append(W_ARR)
    if x+1 < width:
        b |= e
        moves.append(E_ARR)
    if y-1 >= 0:
        b |= s
        moves.append(S_ARR)
    if y+1 < height:
        b |= n
        moves.append(N_ARR)
    if b & (n | w) == nw:
        moves.append(NW_ARR)
    if b & (n | e) == ne:
        moves.append(NE_ARR)
    if b & (s | w) == sw:
        moves.append(SW_ARR)
    if b & (s | e) == se:
        moves.append(SE_ARR)
    moves = [m for m in moves if tuple(np.array(state['agent_coord']) + m) not in (to_avoid + state['leprechaun_coords'])]

    return moves


def is_composite(move: np.ndarray[int]) -> bool:
    for m in MOVES:
        if np.array_equal(move, m):
            return False
    return True


def default_heuristic(state: dict, env: dict):
    agent_stair_dist = np.linalg.norm(np.array(state['agent_coord']) - np.array(state['stair_coord']))
    actual_golds = [g for g in state['gold_coords'] if g != state['agent_coord'] and g != state['stair_coord']]
    gold_in_stair = (state['stair_coord'] in state['gold_coords'])
    n_golds = len(actual_golds)
    if agent_stair_dist == 0:
        return 0
    strategy1_score = env['time_penalty'] * agent_stair_dist + env['stair_score'] + env['gold_score'] * gold_in_stair
    if n_golds == 0:
        return strategy1_score
    agent_gold_dists = [np.linalg.norm(np.array(state['agent_coord']) - np.array(gold_coord)) for gold_coord in actual_golds]
    gold_stair_dists = [np.linalg.norm(np.array(state['stair_coord']) - np.array(gold_coord)) for gold_coord in actual_golds]
    path_lenghts = [ag + gs for ag, gs in zip(agent_gold_dists, gold_stair_dists)]
    min_path_len = min(path_lenghts)
    strategy2_score = env['time_penalty'] * (min(path_lenghts)) + env['gold_score'] * n_golds + env['stair_score'] + env['gold_score'] * gold_in_stair
    return max([strategy1_score, strategy2_score])


def default_score(env: dict, next_state: dict, curr_state: dict = None, curr_g: float = 0.0) -> float:
    if curr_state == None:
        return curr_g + env['gold_score'] * (next_state['agent_coord'] in next_state['gold_coords'])
    else:
        return curr_g + \
            env['gold_score'] * (next_state['agent_coord'] in next_state['gold_coords']) + \
            env['stair_score'] * (next_state['agent_coord'] == next_state['stair_coord']) + \
            env['time_penalty'] * np.linalg.norm(np.array(next_state['agent_coord']) - np.array(curr_state['agent_coord']))


def scaled_default_heuristic(state: dict, env: dict):
    agent_stair_dist = np.linalg.norm(np.array(state['agent_coord']) - np.array(state['stair_coord']))
    if state['gold_coords'] == []:
        return env['time_penalty'] * agent_stair_dist + env['stair_score']
    gold_stair_dists = [np.linalg.norm(np.array(state['stair_coord']) - np.array(gold_coord)) for gold_coord in env['gold_coords']]
    max_h_value = max(
        env['time_penalty'] + env['time_penalty'] * min(gold_stair_dists) + env['gold_score'] * len(env['gold_coords']) + env['stair_score'], \
            env['time_penalty'] * agent_stair_dist + env['stair_score']
        )
    vertices = [(0, 0), (0, env['height'] - 1), (env['width'] - 1, 0), (env['width'] - 1, env['height'] - 1)]
    min_h_value = env['time_penalty'] * max([np.linalg.norm(np.array(env['stair_coord']) - np.array(v)) for v in vertices])
    return (default_heuristic(state=state, env=env) - min_h_value) / (max_h_value - min_h_value)


def scaled_default_score(next_state: dict, curr_state: dict, env: dict, curr_g: float = 0.0) -> float:
    max_g_value = env['time_penalty'] + env['gold_score'] + env['stair_score']
    min_g_value = env['time_penalty']
    return (default_score(next_state=next_state, curr_state=curr_state, env=env, curr_g=curr_g) - min_g_value) / (max_g_value - min_g_value)


#def scaled_score2(next_state: dict, curr_state: dict, env: dict, prev_g: float = 0.0) -> float:
#    max_g_value = env['time_penalty'] + env['gold_score'] + env['stair_score']
#    min_g_value = env['time_penalty']
#    mod_state = next_state
#    mod_state['gold_coords'] = [coord for coord in next_state['gold_coords'] if coord != next_state['agent_coord']]
#    return (default_score(next_state=mod_state, curr_state=curr_state, env=env, prev_g=prev_g) - min_g_value) / (max_g_value - min_g_value)