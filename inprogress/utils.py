import time
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import IPython.display as display
import random
import numpy as np
from queue import PriorityQueue
from typing import List, Tuple, Callable
import json

import gym
from tqdm import tqdm

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
    strategy2_score = env['time_penalty'] * min(path_lenghts) + env['gold_score'] * n_golds + env['stair_score'] + env['gold_score'] * gold_in_stair
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
    gold_stair_dists = [np.linalg.norm(np.array(state['stair_coord']) - np.array(gold_coord)) for gold_coord in state['gold_coords']]
    max_h_value = max(
        env['time_penalty'] + env['time_penalty'] * min(gold_stair_dists) + env['gold_score'] * len(env['gold_coords']) + env['stair_score'], \
            env['time_penalty'] * agent_stair_dist + env['stair_score']
        )
    vertices = [(0, 0), (0, env['height'] - 1), (env['width'] - 1, 0), (env['width'] - 1, env['height'] - 1)]
    min_h_value = env['time_penalty'] * max([np.linalg.norm(np.array(env['stair_coord']) - np.array(v)) for v in vertices])
    if max_h_value == min_h_value:
        return 0
    return (default_heuristic(state=state, env=env) - min_h_value) / (max_h_value - min_h_value)


def scaled_default_score(next_state: dict, curr_state: dict, env: dict, curr_g: float = 0.0) -> float:
    max_g_value = env['time_penalty'] + env['gold_score'] + env['stair_score']
    min_g_value = env['time_penalty']
    if max_g_value == min_g_value:
        return 0
    return (default_score(next_state=next_state, curr_state=curr_state, env=env, curr_g=curr_g) - min_g_value) / (max_g_value - min_g_value)

def run_episodes(
    widths: List[int],
    heights: List[int],
    n_golds: List[int],
    n_leps: List[int],
    gold_scores: List[float],
    stair_scores: List[float],
    time_penalties: List[float],
    algorithms: List[Callable],
    alg_paramss: List[List[dict]],
    max_steps: int,
    n_episodes: int,
    max_fraction = 0.8,
    return_states: bool = False
    ) -> List[dict]:

    episodes = []

    for width, height in zip(widths, heights):
        for time_penalty in tqdm(time_penalties, total=len(time_penalties), desc=f'Size: {width}x{height}'):
            for gold_score in gold_scores:
                for stair_score in stair_scores:
                    for nl in n_leps:
                        if nl < max_fraction*width*height:
                            for ng in tqdm(n_golds, total=len(n_golds), desc=f'size: {width}x{height}, time_penalty: {time_penalty}, gold_score: {gold_score}, n_leps: {nl}'):
                                if ng < max_fraction*width*height:
  
                                    init = {
                                        'width': width,
                                        'height': height,
                                        'n_golds': ng,
                                        'n_leps': nl,
                                        'gold_score': gold_score,
                                        'time_penalty': time_penalty
                                    }

                                    for _ in range(n_episodes):
                                        env = gym.make(
                                            'MiniHack-MyTask-Custom-v0',
                                            width=width,
                                            height=height,
                                            n_leps=nl,
                                            n_golds=ng,
                                            max_episode_steps=max_steps,
                                            gold_score=gold_score,
                                            stair_score=stair_score,
                                            time_penalty=time_penalty
                                            )

                                        for search_algorithm, alg_params in zip(algorithms, alg_paramss):
                                            for kwargs in alg_params:

                                                algorithm = {
                                                    'name': search_algorithm.__name__,
                                                    'params': [(key, str(value)) for key, value in kwargs.items()]
                                                }

                                                env.myreset()
                                                states, rewards, done, iters, steps = search_algorithm(env=env, max_steps=max_steps, **kwargs)

                                                gold_gains = [states[0]['gold']]
                                                gold_thefts = [0]
                                                for state in states[1:]:
                                                    prev = gold_gains[-1]
                                                    diff = state['gold']-prev
                                                    if diff >= 0:
                                                        gold_gains.append(diff)
                                                        gold_thefts.append(0)
                                                    else:
                                                        gold_gains.append(0)
                                                        gold_thefts.append(-diff)

                                                results = {
                                                    'rewards':rewards,
                                                    'steps': steps,
                                                    'iters': iters,
                                                    'gold_thefts': gold_thefts,
                                                    'gold_gains': gold_gains,
                                                    'done': done
                                                }

                                                if return_states:
                                                    results['states'] = states

                                                episode = {
                                                    'init': init,
                                                    'algorithm': algorithm,
                                                    'results': results
                                                }
                                                episodes.append(episode)

                                    with open(f'episodes.json', 'w') as f:
                                        json.dump(episodes, f)
    return episodes     


def get_plot(
    episodes: List[dict],
    fixed: List[Tuple[str, float]],
    x_variable: str,
    y_variable: str,
    ax: Axes,
    f: Callable = (lambda x: x),
    algorithms: List[dict] = None,
    aggregation_function: Callable[List[float], float] = np.mean
    ):

    if algorithms == None:
        algorithms = []
        for episode in episodes:
            if episode['algorithm'] not in algorithms:
                algorithms.append(episode['algorithm'])
    
    plots = []

    for algorithm in algorithms:

        alg_string = algorithm['name']
        for key, value in algorithm['params']:
            if 'SimpleMoves' in value:
                alg_string += ', simple moves'
            elif 'CompositeMoves' in value:
                alg_string += ', composite moves'
            else:
                alg_string += f', {key}={value}'

        x_values = [episode['init'][x_variable] for episode in episodes if np.all([episode['init'][fixed_variable] == fixed_value and episode['algorithm']['name'] == algorithm['name'] and len([1 for p in algorithm['params'] if p in episode['algorithm']['params']])==len(algorithm['params']) for fixed_variable, fixed_value in fixed])]
        x_values.sort()
        y_values = []
        for x_val in x_values:
            y_values.append(aggregation_function([f(episode['results'][y_variable]) for episode in episodes if episode['init'][x_variable] == x_val and np.all([episode['init'][fixed_variable] == fixed_value and episode['algorithm'] == algorithm for fixed_variable, fixed_value in fixed])]))
        plots.append({
            'algorithm': alg_string,
            'x_values': x_values,
            'y_values': y_values
        })

    handles = []
    labels = []
    for p in plots:
        handle, = ax.plot(p['x_values'], p['y_values'], label=p['algorithm'], marker='o')
        handles.append(handle)
        labels.append(p['algorithm'])
    
    ax.set_xlabel(x_variable)
    ax.set_ylabel(f'{aggregation_function.__name__} {y_variable}')

    return handles, labels


def design_plan(
    widths: List[int],
    heights: List[int],
    n_golds: List[int],
    n_leps: List[int],
    gold_scores: List[float],
    stair_scores: List[float],
    time_penalties: List[float],
    algorithms: List[Callable],
    alg_paramss: List[List[dict]],
    max_steps: int,
    n_episodes: int,
    max_fraction = 0.8,
    return_states: bool = False
    ) -> List[dict]:

    plans = []

    i = 0
    for width, height in zip(widths, heights):
        for stair_score in stair_scores:
            for time_penalty in time_penalties:
                for gold_score in gold_scores:
                    for nl in n_leps:
                        if nl < max_fraction*width*height:
                            for ng in tqdm(n_golds, total=len(n_golds), desc=f'size: {width}x{height}, time_penalty: {time_penalty}, gold_score: {gold_score}, n_leps: {nl}'):
                                if ng < max_fraction*width*height:
                            
                                    init = {
                                        'width': width,
                                        'height': height,
                                        'n_golds': ng,
                                        'n_leps': nl,
                                        'gold_score': gold_score,
                                        'time_penalty': time_penalty
                                    }

                                    for _ in range(n_episodes):

                                        env = gym.make(
                                            'MiniHack-MyTask-Custom-v0',
                                            width=width,
                                            height=height,
                                            n_leps=nl,
                                            n_golds=ng,
                                            max_episode_steps=max_steps,
                                            gold_score=gold_score,
                                            stair_score=stair_score,
                                            time_penalty=time_penalty
                                            )

                                    
                                        for search_algorithm, alg_params in zip(algorithms, alg_paramss):
                                            for kwargs in alg_params:

                                                params = []
                                                for key, value in kwargs.items():
                                                    if 'Moves' not in str(value):
                                                        params.append((key, str(value)))
                                                    elif 'Simple' in str(value):
                                                        params.append((key, 'simple_moves'))
                                                    else:
                                                        params.append((key, 'composite_moves'))
                                                        

                                                algorithm = {
                                                    'name': search_algorithm.__name__,
                                                    'params': params
                                                }

                                                plan, expanded_nodes = search_algorithm(env=env, **kwargs)

                                                plan_stats = plan.stats(env=env)

                                                curr_plan = {
                                                    'init': init,
                                                    'algorithm': algorithm,
                                                    'results': {
                                                        'expanded_nodes': expanded_nodes,
                                                        'path_len': plan_stats['path_len'],
                                                        'score': plan_stats['score']
                                                    }
                                                }

                                                plans.append(curr_plan)

                                    with open(f'plans.json', 'w') as f:
                                        json.dump(plans, f)

    return plans        