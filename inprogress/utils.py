import time
import matplotlib.pyplot as plt
import IPython.display as display
import random
import numpy as np
from gold_room_env import MOVES, MOVES_DIRS, MiniHackGoldRoom


def random_step(env:MiniHackGoldRoom):
    return env.mystep(action=random.sample(env.possible_actions(), 1)[0])


def greedy_stair_step(env:MiniHackGoldRoom):
    agent_point = np.array(env.get_agent_coords())
    stair_point = np.array(env.get_stair_coords())
    reachables = [agent_point + move for move in MOVES]
    scores = []
    for reachable_point, move in zip(reachables, MOVES_DIRS):
        if move in env.possible_actions():
            scores.append(-np.linalg.norm(stair_point - reachable_point))
        else:
            scores.append(-np.inf)
    move = MOVES_DIRS[np.argmax(scores)]
    return env.mystep(action=move)


def apply_greedy_strategy(env:MiniHackGoldRoom):
    init_s, init_r = env.myreset()
    states = [init_s]
    rewards = [float(init_r)]
    cumulative_rewards = []
    state, reward, done, info = greedy_stair_step(env)
    states.append(state)
    for i in range(0, env.max_episode_steps):
        if not done:
            state, reward, done, info = greedy_stair_step(env)
            rewards.append(reward)
            cumulative_rewards.append(np.sum(rewards))
            states.append(state)
        else:
            rewards.append(float(env.stair_score))
            break
    return states, rewards, cumulative_rewards, done, i


def show_episode(states):
    image = plt.imshow(states[0]['pixel'][100:300, 500:750])
    for state in states[1:]:
        display.display(plt.gcf())
        display.clear_output(wait=True)
        image.set_data(np.array(state['pixel'])[100:300, 500:750])
        time.sleep(0.3)