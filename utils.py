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

MOVES = [np.array([-1, -1]), np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]), np.array([0, 1]), np.array([1, -1]), np.array([1, 0]), np.array([1, 1])]
MOVES_DIRS = [SW, W, NW, S, N, SE, E, NE]

def print_stats(blstats):
    stat_vals = list(blstats)
    for i, s in enumerate(STATS):
        print(s + ': ' + str(stat_vals[i]))

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
        self.env = gym.make(
            'MiniHack-Navigation-Custom-v0',
            character='sam-hum-neu-mal',
            observation_keys=('blstats', 'chars', 'pixel', 'message'),
            des_file=self.level_generator.get_des()
        )#, reward_manager=self.reward_manager, actions=...)
    
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


    def agent_idxs(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return x[0], y[0]

    def gold_idxs(self):
        xs, ys = np.where(self.curr_state['map'] == GOLD_CHAR)
        return xs, ys

    def goal_idxs(self):
        xs, ys = np.where(self.curr_state['map'] == STAIR_CHAR)
        return xs[0], ys[0]
    
    def agent_coords(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return y[0], self.w - x[0] - 1

    def gold_coords(self):
        xs, ys = np.where(self.curr_state['map'] == GOLD_CHAR)

        return list(zip(ys, [self.w - x - 1 for x in xs]))

    def goal_coords(self):
        xs, ys = np.where(self.curr_state['map'] == STAIR_CHAR)
        return ys[0], self.w - xs[0] - 1

    def possible_actions(self) -> set:
        x, y = self.agent_idxs()
        n = 0b1000
        s = 0b0100
        w = 0b0010
        e = 0b0001
        b = 0b0000
        nw = n & w
        ne = n & e
        sw = s & w
        se = s & e
        possible_actions = set()
        if x-1 >= 0:
            b |= n
        if x+1 < self.h:
            b |= s
        if y-1 >= 0:
            b |= w
        if y+1 < self.w:
            b |= e
        if b & n:
            possible_actions.add(N)
        if b & w:
            possible_actions.add(W)
        if b & e:
            possible_actions.add(E)
        if b & s:
            possible_actions.add(S)
        if b & n & w == nw:
            possible_actions.add(NW)
        if b & n & e == ne:
            possible_actions.add(NE)
        if b & s & w == sw:
            possible_actions.add(SW)
        if b & s & e == se:
            possible_actions.add(SE)
            
        return possible_actions
    
    #TBR
    def dist_from_gold(self, x, y):
        agent_x, agent_y = self.agent_coords()
        gold_coords = self.gold_coords()
        distances = [np.linalg.norm([gold_x - agent_x, gold_y - agent_y]) for gold_x, gold_y in gold_coords]
        min_distance = np.min(distances)
        return min_distance

    #TBR
    def dist_from_goal(self, x, y):
        if (x not in range(0, self.h)) or (y not in range(0, self.w)): return np.inf
        agent_x, agent_y = self.agent_coords()
        goal_x, goal_y = self.goal_coords()
        distances = np.linalg.norm([goal_x - agent_x, goal_y - agent_y])
        min_distance = np.min(distances)
        return min_distance


def random_step(env:GoldRoom):
    return env.step(action=random.sample(env.possible_actions(), 1)[0])


def greedy_goal_step(env:GoldRoom):
    agent_point = np.array(env.agent_coords())
    goal_point = np.array(env.goal_coords())
    reachables = [agent_point + move for move in MOVES]
    scores = []
    for reachable_point, move in zip(reachables, MOVES_DIRS):
        if move in env.possible_actions():
            scores.append(-np.linalg.norm(goal_point - reachable_point))
        else:
            scores.append(-np.inf)
    move = MOVES_DIRS[np.argmax(scores)]
    return env.step(action=move)


def show_episode(states):
    image = plt.imshow(states[0]['pixel'][100:300, 500:750])
    for state in states:
        display.display(plt.gcf())
        display.clear_output(wait=True)
        image.set_data(np.array(state['pixel'])[100:300, 500:750])