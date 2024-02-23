from gold_room_env import MiniHackGoldRoom
from utils import allowed_moves, move_to_action, scaled_default_heuristic, scaled_default_score, default_heuristic, default_score, scaled_score2, ACTIONS
from typing import Callable, List, Tuple
import random
import numpy as np


#def greedy_action(starting_state: dict, value_function1: Callable[dict, float], value_function2: Callable[dict, float], conditions: Callable[List[dict], bool]):
#    moves = allowed_moves_function(state=starting_state)
#
#    if moves == []:
#        action = random.sample(population=ACTIONS, k=1)[0]
#    else:
#        next_agent_coords = [tuple(np.array(starting_state['agent_coord']) + move) for move in moves]
#        next_states = [
#            {
#            'agent_coord': next_agent_coord,
#            'stair_coord': starting_state['stair_coord'],
#            'gold_coords': [coord for coord in starting_state['gold_coords']]
#            }
#        for next_agent_coord in next_agent_coords
#        ]
#        if conditions(next_states):
#            next_values = [value_function1(curr_state=next_state, prev_state=starting_state) for next_state in next_states]
#        else:
#            next_values = [value_function2(curr_state=next_state, prev_state=starting_state) for next_state in next_states]
#
#        max_value = max(next_values)
#        max_value_indexes = [i for i, value in enumerate(next_values) if value == max_value]
#        action = move_to_action(moves[random.sample(population=max_value_indexes, k=1)[0]])
#    return action


#def online_greedy_search(env: MiniHackGoldRoom, value_function1: Callable[[dict, dict], float] = None, value_function2: Callable[[dict, dict], float] = None, conditions: Callable[dict, bool] = None, max_steps: int = 1000):
#
#    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)
#    
#    if value_function1 == None and value_function2 == None and conditions == None:
#        h1 = lambda state: scaled_default_heuristic(state=state, env=env.to_dict())
#        g1 = lambda next_state, curr_state: scaled_default_score(next_state=next_state, curr_state=curr_state, env=env.to_dict())
#        value_function1 = lambda next_state, curr_state: g1(next_state=next_state, curr_state=curr_state) + h1(state=next_state)
#
#    if conditions == None or value_function2 == None:
#        conditions = lambda states: False
#
#        value_function2 = lambda next_state, curr_state: scaled_score2(next_state=next_state, curr_state=curr_state, env=env.to_dict())
#        
#        conditions = lambda state: state['stair_coord'] in state['gold_coords'] and state['stair_coord'] == state['agent_coord'] and len(state['gold_coords']) > 1
#    
#    
#    
#    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)
#
#    state, reward = env.myreset()
#    done = False
#    
#    rewards = []
#    states = []
#
#    for i in range(max_steps):
#        rewards.append(reward)
#        states.append(state)
#        if done:
#            break
#
#        moves = allowed_moves_function(state=state)
#
#        if moves == []:
#            action = random.sample(population=ACTIONS, k=1)[0]
#
#        else:
#            next_agent_coords = [tuple(np.array(state['agent_coord']) + move) for move in moves]
#
#            next_states = [
#                {
#                'agent_coord': next_agent_coord,
#                'stair_coord': state['stair_coord'],
#                'gold_coords': [coord for coord in state['gold_coords'] if coord != state['agent_coord']],
#                'gold': state['gold'] + env.gold_score * (next_agent_coord in state['gold_coords'])
#                }
#            for next_agent_coord in next_agent_coords
#            ]
#            
#            next_values = []
#            for next_state in next_states:
#                if conditions(next_state):
#                    next_values.append(value_function2(next_state=next_state, curr_state=state))
#                else:
#                    next_values.append(value_function1(next_state=next_state, curr_state=state))
#            
#            max_value = max(next_values)
#            max_value_indexes = [i for i, value in enumerate(next_values) if value == max_value]
#            action = move_to_action(moves[random.sample(population=max_value_indexes, k=1)[0]])
#
#        state, reward, done = env.mystep(action=action)
#
#    return states, rewards, done, i


def online_greedy_search(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000):

    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)

    env_dict = env.to_dict()
    env_dict['gold_coords'] = [coord for coord in env_dict['gold_coords'] if coord != env_dict['stair_coord']]
    
    if value_function == None:
        h = lambda state: scaled_default_heuristic(state=state, env=env_dict)
        g = lambda next_state, curr_state: scaled_default_score(next_state=next_state, curr_state=curr_state, env=env_dict)
        value_function = lambda next_state, curr_state: g(next_state=next_state, curr_state=curr_state) + h(state=next_state)

    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)

    state, reward = env.myreset()
    state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]
    done = False
    
    rewards = []
    states = []

    for i in range(max_steps):
        rewards.append(reward)
        states.append(state)
        if done:
            break

        moves = allowed_moves_function(state=state)

        if moves == []:
            action = random.sample(population=ACTIONS, k=1)[0]

        else:
            next_agent_coords = [tuple(np.array(state['agent_coord']) + move) for move in moves]

            next_states = [
                {
                'agent_coord': next_agent_coord,
                'stair_coord': state['stair_coord'],
                'gold_coords': [coord for coord in state['gold_coords'] if coord != state['agent_coord']],
                'gold': state['gold'] + env.gold_score * (next_agent_coord in state['gold_coords'])
                }
            for next_agent_coord in next_agent_coords
            ]
            
            next_values = []
            for next_state in next_states:
                next_values.append(value_function(next_state=next_state, curr_state=state))
            max_value = max(next_values)
            max_value_indexes = [i for i, value in enumerate(next_values) if value == max_value]
            action = move_to_action(moves[random.sample(population=max_value_indexes, k=1)[0]])

        state, reward, done = env.mystep(action=action)
        state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]

    return states, rewards, done, i


#def thr_online_greedy_search(env: MiniHackGoldRoom, gold_to_gather: int, max_steps: int = 1000):
#    
#    conditions = lambda curr_state, next_states: curr_state['gold'] >= gold_to_gather or state['gold_coords'] == []
#    
#    def value_function1(state: dict):
#        agent_stair_dist = np.linalg.norm(np.array(state['agent_coord']) - np.array(state['stair_coord']))
#        gold_stair_dists = [np.linalg.norm(np.array(state['stair_coord']) - np.array(gold_coord)) for gold_coord in env['gold_coords']]
#        max_h_value = env['time_penalty'] + env['time_penalty'] * max(gold_stair_dists) + env['gold_score'] * len(env['gold_coords']) + env['stair_score']
#        vertices = [(0, 0), (0, env['height'] - 1), (env['width'] - 1, 0), (env['width'] - 1, env['height'] - 1)]
#        min_h_value = env['time_penalty'] * max([np.linalg.norm(np.array(env['stair_coord']) - np.array(v)) for v in vertices])
#        return (default_heuristic(state=state, env=env) - min_h_value) / (max_h_value - min_h_value)
#
#    def value_function2(state: dict):
#        agent_stair_dist = np.linalg.norm(np.array(state['agent_coord']) - np.array(state['stair_coord']))
#        if state['gold_coords'] == []:
#            return env['time_penalty'] * agent_stair_dist + env['stair_score']
#    
#    h1 = lambda state: scaled_default_heuristic(state=state, env=env.to_dict())
#    g1 = lambda curr_state, prev_state: scaled_default_score(curr_state=curr_state, prev_state=prev_state, env=env.to_dict())
#    value_function1 = lambda curr_state, prev_state: g1(curr_state=curr_state, prev_state=prev_state) + h1(state=curr_state)