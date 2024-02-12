#from pyswip import Prolog
from minihack import LevelGenerator, RewardManager, MiniHack
from minihack.envs import register
import random
import numpy as np
from nle import nethack
import enum
import warnings
from nle.nethack import Command, CompassCardinalDirection, CompassIntercardinalDirection
import gym
from typing import Any
from utils import action_to_move


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

class MiniHackGoldRoom(MiniHack):

    def __init__(
        self, *args, width=2, height=2, seeds=None, gold_score=1,
        stair_score=1, time_penalty=0, agent_coord=(-1, -1),
        stair_coord=(-1, -1), gold_coords=None, leprechaun_coords=None,
        n_golds=0, n_leps=0, max_episode_steps=100
    ):
        
        if max_episode_steps < 0:
            raise RuntimeError(f'Invalid argument max_episode_steps = {max_episode_steps}')

        if width < 0:
            raise RuntimeError(f'Invalid argument width = {w}')

        if height < 0:
            raise RuntimeError(f'Invalid argument height = {h}')

        if n_golds < 0:
            raise RuntimeError(f'Invalid argument n_stairs = {n_golds}')

        if n_leps < 0:
            raise RuntimeError(f'Invalid argument n_stairs = {n_leps}')
        
        if gold_coords != None:
            seen = set()
            duplicate = False
            for item in gold_coords:
                if item in seen:
                    duplicate = True
                    break
                seen.add(item)
            if duplicate:
                warnings.warn('Gold already in this position, not added', Warning)


        level_generator = LevelGenerator(w=width, h=height)
        self.width = width
        self.height = height
        self.collected_gold = 0
        self.instant = -1
        self.gold_score = gold_score
        self.stair_score = stair_score
        self.time_penalty = time_penalty
        self.max_episode_steps = max_episode_steps
        
        coords = [(x, y) for x in range(width) for y in range(height)]
        if gold_coords == None:
            gold_coords = random.sample(coords, n_golds)

        if leprechaun_coords == None:
            leprechaun_coords = random.sample(coords, n_leps)

        self.start_coord = self._set_start_position(level_generator=level_generator, x=agent_coord[0], y=agent_coord[1])
        self.agent_coord = self.start_coord
        self.stair_coord = self._set_stair_position(level_generator=level_generator, x=stair_coord[0], y=stair_coord[1])
        self.gold_coords = [self._add_gold(level_generator=level_generator, x=x, y=y) for x, y in gold_coords]
        self.leprechaun_coords = [self._add_leprechaun(level_generator=level_generator, x=x, y=y) for x, y in leprechaun_coords]
        
        reward_manager = RewardManager()

        def my_reward_function(env:MiniHackGoldRoom, previous_observation: Any, action: int, current_obsetrvation: Any) -> float:
            reward = env.time_penalty
            state = env.curr_state
            env.agent_coord = tuple(np.array(env.agent_coord) + action_to_move(action=action))

            if 'Your purse feels lighter' in state['message']:
                reward -= env.collected_gold #TODO
            
            elif env.agent_coord in env.gold_coords:
                reward += env.gold_score
                env.collected_gold += env.gold_score
                env.gold_coords.remove(env.agent_coord)
            
            if env.agent_coord == env.stair_coord:
                reward += env.stair_score

            return reward
        
        reward_manager.add_custom_reward_fn(my_reward_function)
        reward_manager.add_coordinate_event(
            coordinates=(stair_coord[0], height - stair_coord[1] - 1),
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


    def copy(self):
        return gym.make(
            'MiniHack-MyTask-Custom-v0',
            width=self.width, height=self.height,
            max_episode_steps=self.max_episode_steps,
            gold_score=self.gold_score,
            stair_score=self.stair_score,
            time_penalty=self.time_penalty,
            agent_coord=self.agent_coord,
            stair_coord=self.stair_coord,
            gold_coords=self.gold_coords,
            leprechaun_coords=self.leprechaun_coords
            )

    def myreset(self):
        if self.agent_coord in self.gold_coords:
            reward = self.gold_score
            self.collected_gold += self.gold_score
            self.gold_coords.remove(self.agent_coord)
        else:
            reward = 0
        self.instant += 1
        self.curr_state = self._state_rapr(minihack_state=self.reset())
        print(f'my coords: {self.agent_coord}\nits coords: {self.get_agent_coords()}')
        return self.curr_state, reward


    def mystep(self, action: int):
        state, reward, done, info = self.step(action)
        self.instant += 1
        self.curr_state = self._state_rapr(minihack_state=state)
        return self.curr_state, reward, done#, info


    def _agent_idxs(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return x[0], y[0]
    
    def get_agent_idxs(self):
        return np.where(self.curr_state['map'] == AGENT_CHAR)
    
    def get_stair_idxs(self):
        x, y = self.stair_coord
        return y - self.height + 1, x

    def get_agent_coords(self):
        x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
        return y[0], self.height - x[0] - 1
    #    return self.agent_coords
    
    def get_leprechaun_coords(self): #TBR
        rows, cols = np.where(self.curr_state['map'] == LEPRECHAUN_CHAR)
        coords = [(col, self.height - row - 1) for row, col in zip(rows, cols)]
        return coords
    
    #def get_gold_coords(self):
    #    #rows, cols = np.where(self.curr_state['map'] == GOLD_CHAR)
    #    #coords = [(col, self.height - row - 1) for row, col in zip(rows, cols)]
    #    #return coords
    #    return self.gold_coordss
    
    #def get_stair_coords(self):
    #    return self.stair_coords

    def _state_rapr(self, minihack_state):
        stats = minihack_state['blstats']
        #stats_dict = {
        #    'health': stats[HEALTH],
        #    'gold': stats[GOLD],
        #    'time': stats[TIME],
        #    'carrying_capacity': stats[CARRYING_CAPACITY]
        #}
        non_empty_rows = ~np.all(minihack_state['chars'] == 32, axis=1)
        non_empty_cols = ~np.all(minihack_state['chars'] == 32, axis=0)
        matrix_map = minihack_state['chars'][non_empty_rows][:, non_empty_cols]
        mystate = {
            #'health': stats[HEALTH],
            'gold': self.collected_gold,#stats[GOLD],
            'time': self.instant,#stats[TIME],
            'carrying_capacity': stats[CARRYING_CAPACITY],
            'agent_coord': self.agent_coord,
            'stair_coord': self.stair_coord,
            'gold_coords': self.gold_coords,
            'map': matrix_map,
            'pixel': minihack_state['pixel'],
            'message': bytes(minihack_state['message']).decode('utf-8').rstrip('\x00')
        }
        return mystate


    def _add_leprechaun(self, level_generator, x, y):
        col_idx = x
        row_idx = self.height - y - 1
        if x < 0:
            raise IndexError('invalid argument x')
        elif x >= self.width:
            raise IndexError('x out of bound')
        if y < 0:
            raise IndexError('invalid argument y')
        elif y >= self.height:
            raise IndexError('y out of bound')
        level_generator.add_monster(name='leprechaun', place=(int(col_idx), int(row_idx)), args=['awake', 'hostile'])#TBR
        return x, y


    def _add_gold(self, level_generator, x, y):
        col_idx = x
        row_idx = self.height - y - 1
        if x < 0:
            raise IndexError('invalid argument x')
        elif x >= self.width:
            raise IndexError('x out of bound')
        if y < 0:
            raise IndexError('invalid argument y')
        elif y >= self.height:
            raise IndexError('y out of bound')
        level_generator.add_gold(amount=1, place=(int(col_idx), int(row_idx))) #TBR
        return x, y


    def _set_start_position(self, level_generator, x=-1, y=-1):
        col_idx = x
        row_idx = self.height - y - 1
        if x < 0:
            col_idx = random.randint(0, self.width-1)
        elif x >= self.width:
            raise IndexError('x out of bound')
        if y < 0:
            row_idx = random.randint(0, self.height-1)
        elif y >= self.height:
            raise IndexError('y out of bound')
        level_generator.set_start_pos((int(col_idx), int(row_idx)))
        x = col_idx
        y = self.height - row_idx - 1
        return x, y


    def _set_stair_position(self, level_generator, x=-1, y=-1):
        col_idx = x
        row_idx = self.height - y - 1
        if x < 0:
            col_idx = random.randint(0, self.width-1)
        elif x >= self.width:
            raise IndexError('x out of bound')
        if y < 0:
            row_idx = random.randint(0, self.height-1)
        elif y >= self.height:
            raise IndexError('y out of bound')
        level_generator.add_goal_pos((int(col_idx), int(row_idx)))
        x = col_idx
        y = self.height - row_idx - 1
        return x, y