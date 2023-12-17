import gym
import time
import matplotlib.pyplot as plt
from pyswip import Prolog
import IPython.display as display
from minihack import LevelGenerator
#from minihack import RewardManager
#from minihack.reward_manager import SequentialRewardManager
import random
from nle import nethack
from nle.nethack import Command
import numpy as np

STATS = ['x', 'y', 'strength_percentage', 'strength', 'dexterity', 'constitution', 'intelligence',
    'wisdom', 'charisma', 'score', 'hitpoints (health)', 'max_hitpoints', 'depth', 'gold',
    'energy', 'max_energy', 'armor_class', 'monster_level', 'experience_level',
    'experience_points', 'time', 'hunger_state', 'carrying_capacity',
    'dungeon_number', 'level_number']

X_COORD = 0
Y_COORD = 1
STRENGTH_PERCENTAGE = 2
STRENGTH = 3
HEALTH = 10
GOLD = 13
TIME = 15
CARRYING_CAPACITY = 17

GOLD_CHAR = 36
LEPRECHAUN_CHAR = 108
AGENT_CHAR = 64
STAIR_CHAR = 62

N = 0
E = 1
S = 2
W = 3
NE = 4
SE = 5
SW = 6
NW = 7

def print_stats(blstats):
    stat_vals = list(blstats)
    i = 0
    for s in STATS:
        print(s + ': ' + str(stat_vals[i]))
        i += 1

'''
class GoldEvent(Event):

    def __init__(self):
        super.__init__(
            reward=0,
            repeatable=True,
            terminal_required=False,
            terminal_sufficient=False
            )
    
    def check(self, env, previous_observation, action, observation) -> float:
        previous_gold = previous_observation[env._original_observation_keys.index('blstats')][13]
        current_gold = observation[env._original_observation_keys.index('blstats')][13]
        return current_gold - previous_gold
'''

class GoldRoom():

    def __init__(self, w=2, h=2):
        self.w = w
        self.h = h
        self.level_generator = LevelGenerator(w=w, h=h)
        #self.reward_manager = RewardManager()
        #self.reward_manager.add_event(GoldEvent())
        self.level_generator.add_goal_pos()

    
    def add_leprechaun(self, x=-1, y=-1):
        if x < 0:
            x = random.randint(0, self.w-1)
        elif x >= self.h:
            raise IndexError('h index out of bound')
        if y < 0:
            y = random.randint(0, self.h-1)
        elif y >= self.w:
            raise IndexError('w index out of bound')

        self.level_generator.add_monster(name='leprechaun', place=(x, y), args=['awake', 'hostile'])

    
    def add_gold(self, amount=1, x=-1, y=-1):
        if x < 0:
            x = random.randint(0, self.w-1)
        elif x >= self.h:
            raise IndexError('h index out of bound')
        if y < 0:
            y = random.randint(0, self.h-1)
        elif y >= self.w:
            raise IndexError('w index out of bound')

        self.level_generator.add_gold(amount=amount, place = (x, y))
    

    def make(self):
        self.env = gym.make('MiniHack-Navigation-Custom-v0',
               character='sam-hum-neu-mal',
               observation_keys=('blstats', 'chars', 'pixel', 'message'),
               des_file=self.level_generator.get_des())#, reward_manager=self.reward_manager ,actions=...)
    
    def reset(self):
        state = self.env.reset()
        stats = state['blstats']
        stats_dict = {
            'x_coord': stats[X_COORD],
            'y_coord': stats[Y_COORD],
            'strength_percentage': stats[STRENGTH_PERCENTAGE],
            'strength': stats[STRENGTH],
            'health': stats[HEALTH],
            'gold': stats[GOLD],
            'time': stats[TIME],
            'carrying_capacity': stats[CARRYING_CAPACITY]
        }

        non_empty_rows = ~np.all(state['chars'] == 32, axis=1)
        non_empty_cols = ~np.all(state['chars'] == 32, axis=0)

        matrix_map = state['chars'][non_empty_rows][:, non_empty_cols]

        self.prev_state = None
        self.curr_state = {
            'stats': stats_dict,
            'map': matrix_map,
            'pixel': state['pixel'],
            'message': bytes(state['message']).decode('utf-8').rstrip('\x00')
        }
        return self.curr_state


    def step(self, action:int):
        state, reward, done, info = self.env.step(action)
        stats = state['blstats']
        stats_dict = {
            'x_coord': stats[X_COORD],
            'y_coord': stats[Y_COORD],
            'strength_percentage': stats[STRENGTH_PERCENTAGE],
            'strength': stats[STRENGTH],
            'health': stats[HEALTH],
            'gold': stats[GOLD],
            'time': stats[TIME],
            'carrying_capacity': stats[CARRYING_CAPACITY]
        }

        non_empty_rows = ~np.all(state['chars'] == 32, axis=1)
        non_empty_cols = ~np.all(state['chars'] == 32, axis=0)

        matrix_map = state['chars'][non_empty_rows][:, non_empty_cols]

        self.prev_state = self.curr_state
        self.curr_state = {
            'stats': stats_dict,
            'map': matrix_map,
            'pixel': state['pixel'],
            'message': bytes(state['message']).decode('utf-8').rstrip('\x00')
        }
        return self.curr_state, reward, done, info


    def agent_pos(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return x[0], y[0]


    def possible_actions(self) -> set:
        x, y = self.agent_pos()
        possible_actions = set()
        if x-1 >= 0:
            possible_actions.add(N)
        if x+1 < self.h:
            possible_actions.add(S)
        if y-1 >= 0:
            possible_actions.add(W)
            if N in possible_actions:
                possible_actions.add(NW)
            if S in possible_actions:
                possible_actions.add(SW)
        if y+1 < self.w:
            possible_actions.add(E)
            if N in possible_actions:
                possible_actions.add(NE)
            if S in possible_actions:
                possible_actions.add(SE)
        return possible_actions
    
    #TBR
    def dist_from_gold(self, x, y):
        curr_point = np.array(x, y)
        gold_x, gold_y = np.where(self.curr_state['map'] == GOLD_CHAR)
        distances = np.linalg.norm(np.array([gold_x, gold_y]).T - np.array([x, y]), axis=1)
        min_distance = np.min(distances)
        return min_distance

    #TBR
    def dist_from_leprechaun(self, x, y):
        curr_point = np.array(x, y)
        lep_x, lep_y = np.where(self.curr_state['map'] == LEPRECHAUN_CHAR)
        distances = np.linalg.norm(np.array([lep_x, lep_y]).T - np.array([x, y]), axis=1)
        min_distance = np.min(distances)
        return min_distance

    #TBR
    def dist_from_goal(self, x, y):
        if (x not in range(0, self.h)) or (y not in range(0, self.w)): return np.inf
        curr_point = np.array(x, y)
        goal_x, goal_y = np.where(self.curr_state['map'] == STAIR_CHAR)
        distances = np.linalg.norm(np.array([goal_x, goal_y]).T - np.array([x, y]), axis=1)
        min_distance = np.min(distances)
        return min_distance


def random_step(env:GoldRoom):
    return env.step(action=random.sample(env.possible_actions(), 1)[0])

#TBR
def greedy_goal_step(env:GoldRoom):
    x, y = env.agent_pos()
    coords = []
    dist = []
    for i in [x-1, x, x+1]:
        for j in [y-1, y, y+1]:
            if i != x or j != y:
                coords.append((i, j))
                dist.append(env.dist_from_goal(i, j))
    idx = np.argmin(dist)
    x_new, y_new = coords[idx]
    print(x_new, y_new)
    if x_new == x+1 and y_new == y-1: return env.step(action=NW)
    elif x_new == x+1 and y_new == y: return env.step(action=N)
    elif x_new == x+1 and y_new == y+1: return env.step(action=NE)
    elif x_new == x and y_new == y-1: return env.step(action=W)
    elif x_new == x and y_new == y+1: return env.step(action=E)
    if x_new == x-1 and y_new == y-1: return env.step(action=SW)
    elif x_new == x-1 and y_new == y: return env.step(action=S)
    elif x_new == x-1 and y_new == y+1: return env.step(action=SE)

def show_episode(states):
    image = plt.imshow(states[0]['pixel'][100:300, 500:750])
    for state in states:
        display.display(plt.gcf())
        display.clear_output(wait=True)
        image.set_data(np.array(state['pixel'])[100:300, 500:750])