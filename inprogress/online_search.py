from gold_room_env import MiniHackGoldRoom
from utils import allowed_moves, move_to_action, scaled_default_heuristic, scaled_default_score, default_heuristic, default_score, ACTIONS
from typing import Callable, List, Tuple
import random
import numpy as np



def online_search_f(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000, selection_policy: Callable[List[float], int] = None, prob_rand_move: Callable[[int, float, float], float] = None):

    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)

    env_dict = env.to_dict()
    env_dict['gold_coords'] = [coord for coord in env_dict['gold_coords'] if coord != env_dict['stair_coord']]
    
    if value_function == None:
        h = lambda state: scaled_default_heuristic(state=state, env=env_dict)
        g = lambda next_state, curr_state: scaled_default_score(next_state=next_state, curr_state=curr_state, env=env_dict)
        value_function = lambda next_state, curr_state: g(next_state=next_state, curr_state=curr_state) + h(state=next_state)

    if selection_policy == None:
        def selection_policy(values: List[float]) -> int:
            max_value = max(next_values)
            max_value_indexes = [i for i, value in enumerate(next_values) if value == max_value]
            return random.sample(population=max_value_indexes, k=1)[0]


    if prob_rand_move == None:
        prob_rand_move = lambda t, curr_value, next_value: 0

    state, reward = env.myreset()
    state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]
    done = False
    
    rewards = []
    states = []
    curr_value = value_function(next_state=state, curr_state=None)

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

            next_value_index = selection_policy(next_values)
            next_value = next_values[next_value_index]

            if random.uniform(0, 1) < prob_rand_move(t=i, curr_value=curr_value, next_value=next_value):
                action = move_to_action(random.sample(population=moves, k=1)[0])
            else:
                action = move_to_action(moves[next_value_index])

        state, reward, done = env.mystep(action=action)
        state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]

    return states, rewards, done, i

def hill_climbing(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000):
    return online_search(env=env, value_function=value_function, max_steps=max_steps, prob_rand_move=lambda t, curr_value, next_value: np.exp((curr_value - next_value) / (max_steps - t)))