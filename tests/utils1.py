import gym
import time
import matplotlib.pyplot as plt
from pyswip import Prolog
import IPython.display as display
from minihack import LevelGenerator
import random
import numpy as np
from nle import nethack
import enum
import warnings
#from nle.nethack import Command, CompassCardinalDirection, CompassIntercardinalDirection


#class WaitAction(enum.IntEnum):
#    WAIT = ord(".")
#
#Actions = enum.IntEnum(
#    "Actions",
#    {
#        **CompassCardinalDirection.__members__,
#        **CompassIntercardinalDirection.__members__,
#        **WaitAction.__members__,
#    },
#)
#
#ACTIONS = tuple(Actions)


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


class GoldRoom():

    def __init__(self, w=2, h=2, seed=None, gold_score=1, stair_score=1, time_penalty=0):
        self.w = w
        self.h = h
        self.level_generator = LevelGenerator(w=w, h=h)
        self.level_generator.add_goal_pos()
        self.seed = seed
        self.gold_score = gold_score
        self.stair_score = stair_score
        self.time_penalty = time_penalty
        self.gold_coordss = []
        self.stair_coords = None
        self.collected_gold = 0
    

    @staticmethod
    def stateRapr(state):
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
        mystate = {
            'stats': stats_dict,
            'map': matrix_map,
            'pixel': state['pixel'],
            'message': bytes(state['message']).decode('utf-8').rstrip('\x00')
        }
        return mystate


    
    def add_leprechaun(self, x=-1, y=-1):
        col_idx = x
        row_idx = self.h - y - 1
        if x < 0:
            col_idx = random.randint(0, self.w-1)
        elif x >= self.w:
            raise IndexError('h index out of bound')
        if y < 0:
            row_idx = random.randint(0, self.h-1)
        elif y >= self.h:
            raise IndexError('w index out of bound')
        
        self.level_generator.add_monster(name='leprechaun', place=(col_idx, row_idx), args=['awake', 'hostile'])

    
    def add_gold(self, x=-1, y=-1):
        col_idx = x
        row_idx = self.h - y - 1
        if x < 0:
            col_idx = random.randint(0, self.w-1)
        elif x >= self.w:
            raise IndexError('x out of bound')
        if y < 0:
            row_idx = random.randint(0, self.h-1)
        elif y >= self.h:
            raise IndexError('y out of bound')
        if (x, y) in self.gold_coordss:
            warnings.warn('Gold already in this position, not added', Warning)
        else:
            self.gold_coordss.append((x, y))
            self.level_generator.add_gold(amount=1, place=(col_idx, row_idx))
    

    def set_start_position(self, x, y):
        col_idx = x
        row_idx = self.h - y - 1
        if x < 0:
            col_idx = random.randint(0, self.w-1)
        elif x >= self.w:
            raise IndexError('x out of bound')
        if y < 0:
            row_idx = random.randint(0, self.h-1)
        elif y >= self.h:
            raise IndexError('y out of bound')

        self.level_generator.set_start_pos((col_idx, row_idx))
    

    def make(self):
        self.env = gym.make(
            'MiniHack-Navigation-Custom-v0',
            character='sam-hum-neu-mal',
            observation_keys=('blstats', 'chars', 'pixel', 'message'),
            des_file=self.level_generator.get_des()
        )
        if self.seed is not None:
            self.env.seed(self.seed) #TBR


    def reset(self):
        self.prev_state = None
        state = self.env.reset()
        self.curr_state = GoldRoom.stateRapr(state)
        self.stair_coords = self.get_stair_coords()
        if self.get_agent_coords() in self.gold_coordss:
            reward = self.gold_score
            self.collected_gold += self.gold_score
        else:
            reward = 0
        return self.curr_state, reward


    def step(self, action:int):
        state, reward, done, info = self.env.step(action)
        self.prev_state = self.curr_state
        self.curr_state = GoldRoom.stateRapr(state)
        myreward = self.time_penalty
        if not done:
            if self.get_agent_coords() in self.gold_coordss:
                myreward += self.gold_score
                self.collected_gold += self.gold_score
            if self.get_agent_coords() == self.get_leprechaun_coords():
                myreward -= self.collected_gold
                self.collected_gold = 0
        if done:
            myreward += self.stair_score
        return self.curr_state, myreward, done, info


    def _agent_idxs(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return x[0], y[0]

#    def gold_idxs(self):
#        xs, ys = np.where(self.curr_state['map'] == GOLD_CHAR)
#        return xs, ys
#
#    def stair_idxs(self):
#        xs, ys = np.where(self.curr_state['map'] == STAIR_CHAR)
#        return xs[0], ys[0]

    def get_agent_coords(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return y[0], self.w - x[0] - 1
    

    def get_leprechaun_coords(self):
        x, y = np.where(self.curr_state['map'] == LEPRECHAUN_CHAR)
        return y[0], self.w - x[0] - 1
    

    def get_gold_coords(self):
        return self.gold_coordss
        #xs, ys = np.where(self.curr_state['map'] == GOLD_CHAR)
        #return list(zip(ys, [self.w - x - 1 for x in xs]))

    def get_stair_coords(self):
        xs, ys = np.where(self.curr_state['map'] == STAIR_CHAR)
        return ys[0], self.w - xs[0] - 1

    def possible_actions(self) -> set:
        x, y = self._agent_idxs()
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


def random_step(env:GoldRoom):
    return env.step(action=random.sample(env.possible_actions(), 1)[0])


def greedy_stair_step(env:GoldRoom):
    print(env.get_agent_coords())
    agent_point = np.array(env.get_agent_coords())
    stair_point = np.array(env.stair_coords)
    reachables = [agent_point + move for move in MOVES]
    scores = []
    for reachable_point, move in zip(reachables, MOVES_DIRS):
        if move in env.possible_actions():
            scores.append(-np.linalg.norm(stair_point - reachable_point))
        else:
            scores.append(-np.inf)
    move = MOVES_DIRS[np.argmax(scores)]
    return env.step(action=move)


def apply_greedy_strategy(env, max_steps):
    env.make()
    init_s, init_r = env.reset()
    states = [init_s]
    rewards = [init_r]
    cumulative_rewards = []
    state, reward, done, info = greedy_stair_step(env)
    states.append(state)
    for i in range(0, max_steps):
        if not done:
            state, reward, done, info = greedy_stair_step(env)
            rewards.append(reward)
            cumulative_rewards.append(np.sum(rewards))
            states.append(state)
        else:
            break
    return states, rewards, cumulative_rewards, done, i


def show_episode(states):
    image = plt.imshow(states[0]['pixel'][100:300, 500:750])
    for state in states[1:]:
        display.display(plt.gcf())
        display.clear_output(wait=True)
        image.set_data(np.array(state['pixel'])[100:300, 500:750])
        time.sleep(0.2)