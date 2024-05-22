from gold_room_env import MiniHackGoldRoom
from utils import allowed_moves, move_to_action, scaled_default_heuristic, scaled_default_score, default_heuristic, default_score, ACTIONS
from typing import Callable, List, Tuple
import random
import numpy as np

def online_search_f(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000, selection_policy: Callable[List[float], int] = None, prob_move: Callable[[int, float, float], float] = None, greedy_alternative = False):

    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)

    env_dict = env.to_dict()
    env_dict['gold_coords'] = [coord for coord in env_dict['gold_coords'] if coord != env_dict['stair_coord']]
    
    if value_function == None:
        h = lambda state: scaled_default_heuristic(state=state, env=env_dict)
        g = lambda next_state, curr_state: scaled_default_score(next_state=next_state, curr_state=curr_state, env=env_dict)
        value_function = lambda next_state, curr_state: g(next_state=next_state, curr_state=curr_state) + h(state=next_state)
    
    def greedy_selection(values: List[float]) -> int:
        max_value = max(values)
        max_value_indexes = [i for i, value in enumerate(values) if value == max_value]
        return random.sample(population=max_value_indexes, k=1)[0]

    if selection_policy == None:
        selection_policy = greedy_selection

    if prob_move == None:
        prob_move = lambda t, curr_value, next_value: 1#0

    state, reward = env.myreset()
    state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]
    done = False
    
    rewards = [reward]
    states = [state]
    curr_value = value_function(next_state=state, curr_state=None)
    nop = 0

    for i in range(max_steps):
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

            if random.uniform(0, 1) <= prob_move(t=i, curr_value=curr_value, next_value=next_value):
                #action = move_to_action(random.sample(population=moves, k=1)[0])
            #else:
                action = move_to_action(moves[next_value_index])
                state, reward, done = env.mystep(action=action)
                state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]
                rewards.append(reward)
                states.append(state)

            elif greedy_alternative:
                next_value_index = greedy_selection(next_values)
                next_value = next_values[next_value_index]
                action = move_to_action(moves[next_value_index])
                state, reward, done = env.mystep(action=action)
                state['gold_coords'] = [coord for coord in state['gold_coords'] if coord != state['stair_coord']]
                rewards.append(reward)
                states.append(state)
            
            else:
                nop += 1

    return states, rewards, done, i, i-nop

#def hill_climbing(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000):
#    return online_search_f(env=env, value_function=value_function, max_steps=max_steps, prob_rand_move=lambda t, curr_value, next_value: np.exp((curr_value - next_value) / (max_steps - t)))

def online_greedy_search(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000):
    states, rewards, done, i, _ = online_search_f(env=env, value_function=value_function, max_steps=max_steps)
    return states, rewards, done, i

def simulated_annealing(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000, temperature: Callable[[int, float, float], float] = None, k = 1):
    
    def energy(curr_value, next_value):

        if next_value < curr_value:
            return next_value - curr_value
        else:
            return 0

    if temperature == None:
        temperature = lambda t: k*(1-(t+1)/max_steps) + 1

    def prob_move(t, curr_value, next_value):
        return np.exp(energy(curr_value=curr_value, next_value=next_value) / temperature(t))
    
    def selection_policy(values: List[float]) -> int:
        indexes = [i for i, value in enumerate(values)]
        return random.sample(population=indexes, k=1)[0]
    
    return online_search_f(env=env, value_function=value_function, max_steps=max_steps, selection_policy=selection_policy, prob_move=prob_move)

def online_random_greedy_search(env: MiniHackGoldRoom, value_function: Callable[[dict, dict], float] = None, max_steps: int = 1000, prob_rand_move: float = 0.5):

    def selection_policy(values: List[float]) -> int:
        indexes = [i for i, value in enumerate(values)]
        return random.sample(population=indexes, k=1)[0]
    
    def prob_move(t, curr_value, next_value):
        return prob_rand_move

    states, rewards, done, i, _ =  online_search_f(env=env, value_function=value_function, max_steps=max_steps, selection_policy=selection_policy, prob_move=prob_move, greedy_alternative=True)
    return states, rewards, done, i