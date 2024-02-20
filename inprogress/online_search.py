from gold_room_env import MiniHackGoldRoom
from utils import allowed_moves, move_to_action, scaled_default_heuristic, scaled_default_score, default_heuristic, default_score, ACTIONS
from typing import Callable
import random
import numpy as np

def online_greedy_search(env: MiniHackGoldRoom, value_function: Callable[dict, float] = None, max_steps: int = 1000):
    state, reward = env.myreset()
    done = False

    if value_function == None:
        h = lambda state: scaled_default_heuristic(state=state, env=env.to_dict())
        g = lambda curr_state, prev_state: scaled_default_score(curr_state=curr_state, prev_state=prev_state, env=env.to_dict())

        h2 = lambda state: default_heuristic(state=state, env=env.to_dict())
        g2 = lambda curr_state, prev_state: default_score(curr_state=curr_state, prev_state=prev_state, env=env.to_dict())

        value_function = lambda curr_state, prev_state: g(curr_state=curr_state, prev_state=prev_state) + h(state=curr_state)
        value_function2 = lambda curr_state, prev_state: g2(curr_state=curr_state, prev_state=prev_state) + h2(state=curr_state)
    else:
        value_function2 = value_function
    
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
                'gold_coords': [coord for coord in state['gold_coords']]# if coord != next_agent_coord]
                }
            for next_agent_coord in next_agent_coords
            ]
            
            if env.stair_coord in next_agent_coords and env.stair_coord in state['gold_coords']:# and len([gold_coord for gold_coord in state['gold_coords'] if gold_coord in next_agent_coords]) > 1:
                next_values = [value_function2(curr_state=next_state, prev_state=state) for next_state in next_states]
            else:       
                next_values = [value_function(curr_state=next_state, prev_state=state) for next_state in next_states]
            max_value = max(next_values)
            max_value_indexes = [i for i, value in enumerate(next_values) if value == max_value]
            action = move_to_action(moves[random.sample(population=max_value_indexes, k=1)[0]])

        prev_state = state
        state, reward, done = env.mystep(action=action)
        current_value = value_function(curr_state=state, prev_state=prev_state)

    return states, rewards, done, i

    # IDEA: iterative approach: fix a threshold score value and search a solution that reach that score with hill climbing, sim. ann., change the threshold