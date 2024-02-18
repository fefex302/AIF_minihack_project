from gold_room_env import MiniHackGoldRoom
from utils import allowed_moves, move_to_action, scaled_default_heuristic, scaled_default_score, ACTIONS
from typing import Callable
import random
import numpy as np

def online_greedy_search(env: MiniHackGoldRoom, value_function: Callable[dict, float] = None, max_steps: int = 1000):
    state, reward = env.myreset()
    done = False

    if value_function == None:
        h = lambda state: scaled_default_heuristic(state=state, env=env.to_dict())
        g = lambda curr_state, prev_state: scaled_default_score(curr_state=curr_state, prev_state=prev_state, env=env.to_dict())

        value_function = lambda curr_state, prev_state: g(curr_state=curr_state, prev_state=prev_state) #+ h(state=curr_state)
    
    allowed_moves_function = lambda state: allowed_moves(width=env.width, height=env.height, state=state)

    current_value = value_function(curr_state=state, prev_state=state)
    
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
                'gold_coords': [coord for coord in state['gold_coords'] if coord != next_agent_coord]
                }
            for next_agent_coord in next_agent_coords
            ]

            next_values = [value_function(curr_state=next_state, prev_state=state) for next_state in next_states]
            max_value = max(next_values)
            max_value_index = next_values.index(max_value)
            action = move_to_action(moves[max_value_index])

        prev_state = state
        state, reward, done = env.mystep(action=action)
        current_value = value_function(curr_state=state, prev_state=prev_state)

    return states, rewards, done, i