#from pyswip import Prolog
from minihack import LevelGenerator, RewardManager, MiniHack
from minihack.envs import register
import random
import numpy as np
from nle import nethack
import enum
import warnings
from nle.nethack import Command, CompassCardinalDirection, CompassIntercardinalDirection


class WaitAction(enum.IntEnum):
    WAIT = ord(".")

Actions = enum.IntEnum(
    "Actions",
    {
        **CompassCardinalDirection.__members__,
        **CompassIntercardinalDirection.__members__,
        **WaitAction.__members__,
    },
)

ACTIONS = tuple(Actions)


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



class MiniHackGoldRoom(MiniHack):

    def __init__(
        self, *args, w=2, h=2, seeds=None, gold_score=1,
        stair_score=1, time_penalty=0, agent_coords=(-1, -1),
        stair_coords=(-1, -1), gold_coordss=None, leprechaun_coordss=None,
        n_golds=0, n_leps=0, max_episode_steps=100
    ):
        
        if max_episode_steps < 0:
            raise RuntimeError(f'Invalid argument max_episode_steps = {max_episode_steps}')

        if w < 0:
            raise RuntimeError(f'Invalid argument w = {w}')

        if h < 0:
            raise RuntimeError(f'Invalid argument h = {h}')

        if n_golds < 0:
            raise RuntimeError(f'Invalid argument n_stairs = {n_golds}')

        if n_leps < 0:
            raise RuntimeError(f'Invalid argument n_stairs = {n_leps}')
        
        if gold_coordss != None:
            seen = set()
            duplicate = False
            for item in gold_coordss:
                if item in seen:
                    duplicate = True
                    break
                seen.add(item)
            if duplicate:
                warnings.warn('Gold already in this position, not added', Warning)


        level_generator = LevelGenerator(w=w, h=h)
        self.w = w
        self.h = h
        self.collected_gold = 0
        self.gold_score = gold_score
        self.stair_score = stair_score
        self.time_penalty = time_penalty
        self.max_episode_steps = max_episode_steps
        
        coords = [(x, y) for x in range(w) for y in range(h)]
        if gold_coordss == None:
            gold_coordss = random.sample(coords, n_golds)

        if leprechaun_coordss == None:
            leprechaun_coordss = random.sample(coords, n_leps)

        self.start_coords = self._set_start_position(level_generator=level_generator, x=agent_coords[0], y=agent_coords[1])
        self.stair_coords = self._set_stair_position(level_generator=level_generator, x=stair_coords[0], y=stair_coords[1])
        self.gold_coordss = [self._add_gold(level_generator=level_generator, x=x, y=y) for x, y in gold_coordss]
        self.leprechaun_coordss = [self._add_leprechaun(level_generator=level_generator, x=x, y=y) for x, y in leprechaun_coordss]
        
        if self.stair_coords in self.gold_coordss:
            self.stair_score += self.gold_score
        reward_manager = RewardManager()

        def my_reward_function(env:MiniHackGoldRoom, previous_observation=None, action=0, current_obsetrvation=None):
            reward = env.time_penalty
            state = env.curr_state
            if 'Your purse feels lighter' in state['message']:
                reward -= env.collected_gold
            elif env.get_agent_coords() in env.get_gold_coords():
                reward += env.gold_score
                env.collected_gold += env.gold_score
            return reward
        
        reward_manager.add_custom_reward_fn(my_reward_function)
        reward_manager.add_coordinate_event(
            coordinates=(stair_coords[0], h - stair_coords[1] - 1),
            reward=stair_score,
            repeatable=False,
            terminal_required=True,
            terminal_sufficient=True
        )

        super().__init__(
            *args, des_file=level_generator.get_des(), reward_manager=reward_manager,
            actions=ACTIONS, allow_all_yn_questions=False, allow_all_modes=False,
            character='rog-hum-cha-mal', max_episode_steps=max_episode_steps,
            observation_keys=('blstats', 'chars', 'pixel', 'message')
        )

    register(
        id="MiniHack-MyTask-Custom-v0",
        entry_point="gold_room_env:MiniHackGoldRoom",
    )


    def myreset(self):
        self.curr_state = MiniHackGoldRoom.stateRapr(self.reset())
        if self.get_agent_coords() in self.get_gold_coords():
            reward = self.gold_score
            self.collected_gold += self.gold_score
        else:
            reward = 0
        return self.curr_state, reward


    def mystep(self, action:int):
        state, reward, done, info = self.step(action)
        self.curr_state = MiniHackGoldRoom.stateRapr(state)
        return self.curr_state, reward, done, info


    def _agent_idxs(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return x[0], y[0]

    def get_agent_coords(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return y[0], self.w - x[0] - 1
    
    def get_leprechaun_coords(self):
        x, y = np.where(self.curr_state['map'] == LEPRECHAUN_CHAR)
        return y[0], self.w - x[0] - 1
    
    def get_gold_coords(self):
        return self.gold_coordss
    
    def get_stair_coords(self):
        return self.stair_coords


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


    def _add_leprechaun(self, level_generator, x, y):
        col_idx = x
        row_idx = self.h - y - 1
        if x < 0:
            raise IndexError('invalid argument x')
        elif x >= self.w:
            raise IndexError('x out of bound')
        if y < 0:
            raise IndexError('invalid argument y')
        elif y >= self.h:
            raise IndexError('y out of bound')
        level_generator.add_monster(name='leprechaun', place=(col_idx, row_idx), args=['awake', 'hostile'])
        return x, y


    def _add_gold(self, level_generator, x, y):
        col_idx = x
        row_idx = self.h - y - 1
        if x < 0:
            raise IndexError('invalid argument x')
        elif x >= self.w:
            raise IndexError('x out of bound')
        if y < 0:
            raise IndexError('invalid argument y')
        elif y >= self.h:
            raise IndexError('y out of bound')
        level_generator.add_gold(amount=1, place=(col_idx, row_idx))
        return x, y


    def _set_start_position(self, level_generator, x=-1, y=-1):
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
        level_generator.set_start_pos((col_idx, row_idx))
        x = col_idx
        y = self.h - row_idx -1
        return x, y


    def _set_stair_position(self, level_generator, x=-1, y=-1):
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
        level_generator.add_goal_pos((col_idx, row_idx))
        x = col_idx
        y = self.h - row_idx -1
        return x, y