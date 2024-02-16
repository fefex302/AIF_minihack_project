from minihack import LevelGenerator, RewardManager, MiniHack
from minihack.envs import register
import random
import numpy as np
from nle import nethack
import enum
import warnings
from nle.nethack import Command, CompassCardinalDirection, CompassIntercardinalDirection
import gym
from typing import Any, List, Tuple
from utils import action_to_move, DIAGONAL_ACTIONS

Actions = enum.IntEnum(
    "Actions",
    {
        **CompassCardinalDirection.__members__,
        **CompassIntercardinalDirection.__members__,
    },
)

ACTIONS = tuple(Actions)


#STATS = ['x', 'y', 'strength_percentage', 'strength', 'dexterity', 'constitution', 'intelligence',
#    'wisdom', 'charisma', 'score', 'hitpoints (health)', 'max_hitpoints', 'depth', 'gold',
#    'energy', 'max_energy', 'armor_class', 'monster_level', 'experience_level',
#    'experience_points', 'time', 'hunger_state', 'carrying_capacity',
#    'dungeon_number', 'level_number']

#X_COORD = 0
#Y_COORD = 1
#STRENGTH_PERCENTAGE = 2
#STRENGTH = 3
#HEALTH = 10
#GOLD = 13
#TIME = 15
#CARRYING_CAPACITY = 17

GOLD_CHAR = 36
LEPRECHAUN_CHAR = 108
AGENT_CHAR = 64
STAIR_CHAR = 62

class MiniHackGoldRoom(MiniHack):

    # Constructor ------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        *args,
        width: int = 2,
        height: int = 2,
        gold_score: float = 1.0,
        stair_score: float = 1.0,
        time_penalty: float = -1.0,
        agent_coord: Tuple[int, int] = None,
        stair_coord: Tuple[int, int] = None,
        gold_coords: List[Tuple[int, int]] = None,
        leprechaun_coords: List[Tuple[int, int]] = None,
        n_golds: int = 0,
        n_leps: int = 0,
        max_episode_steps: int = 100
    ):
        # Argument checks ---------------------------------------------------------------------

        # Type checks

        if not isinstance(width, int):
            raise TypeError(f'width parameter must be of type int, not {type(width)}')
        
        if not isinstance(height, int):
            raise TypeError(f'height parameter must be of type int, not {type(height)}')
        
        if not isinstance(gold_score, (int, float)):
            raise TypeError(f'gold_score parameter must be of type int or float, not {type(gold_score)}')
        
        if not isinstance(stair_score, (int, float)):
            raise TypeError(f'stair_score parameter must be of type int or float, not {type(stair_score)}')

        if not isinstance(time_penalty, (int, float)):
            raise TypeError(f'time_penalty parameter must be of type int or float, not {type(time_penalty)}')

        if not isinstance(agent_coord, Tuple) and agent_coord != None:
            raise TypeError(f'agent_coord parameter must be of type Tuple, not {type(agent_coord)}')
        
        if not isinstance(stair_coord, Tuple) and stair_coord != None:
            raise TypeError(f'stair_coord parameter must be of type Tuple, not {type(stair_coord)}')
        
        if not isinstance(gold_coords, List) and gold_coords != None:
            raise TypeError(f'gold_coords parameter must be of type List, not {type(gold_coords)}')
        
        if gold_coords != None:
            for item in gold_coords:
                if not isinstance(item, Tuple):
                    raise TypeError(f'gold_coords parameter must be of type List[Tuple], not {type(gold_coords)}')

            if not isinstance(leprechaun_coords, List) and leprechaun_coords != None:
                raise TypeError(f'leprechaun_coords parameter must be of type List, not {type(leprechaun_coords)}')

        if leprechaun_coords != None:
            for item in leprechaun_coords:
                if not isinstance(item, Tuple):
                    raise TypeError(f'leprechaun_coords parameter must be of type List[Tuple], not {type(leprechaun_coords)}')

        if not isinstance(n_golds, int):
            raise TypeError(f'n_golds parameter must be of type int, not {type(n_golds)}')
        
        if not isinstance(n_leps, int):
            raise TypeError(f'n_leps parameter must be of type int, not {type(n_leps)}')

        # Value checks

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
        
        if agent_coord != None:
            x, y = agent_coord
            if x not in range(width):
                    raise RuntimeError(f'Agent coordinate out of bound [0, {width}): x = {x}')
            if y not in range(height):
                raise RuntimeError(f'Agent coordinate out of bound [0, {height}): y = {y}')
            if agent_coord == stair_coord:
                raise RuntimeError(f'Overlapping start and end coordinates')
        
        if stair_coord != None:
            x, y = stair_coord
            if x not in range(width):
                    raise RuntimeError(f'Stair coordinate out of bound [0, {width}): x = {x}')
            if y not in range(height):
                raise RuntimeError(f'Stair coordinate out of bound [0, {height}): y = {y}')
        
        if gold_coords != None:
            seen = set()
            duplicate = False
            for (x, y) in gold_coords:
                if (x, y) in seen:
                    duplicate = True
                if x not in range(width):
                    raise RuntimeError(f'Gold coordinate out of bound [0, {width}): x = {x}')
                if y not in range(height):
                    raise RuntimeError(f'Gold coordinate out of bound [0, {height}): y = {y}')
                seen.add((x, y))
            if duplicate:
                warnings.warn(f'Multiple gold items in the same position: only one added', Warning)
        
        if leprechaun_coords != None:
            for (x, y) in leprechaun_coords:
                if x not in range(width):
                    raise RuntimeError(f'Leprechaun coordinate out of bound [0, {width}): x = {x}')
                if y not in range(height):
                    raise RuntimeError(f'Leprechaun coordinate out of bound [0, {height}): y = {y}')

        # Set class variables -----------------------------------------------------------------

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
            self.gold_coords = random.sample(population=coords, k=n_golds)
        else:
            self.gold_coords = gold_coords

        if leprechaun_coords == None:
            self.leprechaun_coords = random.sample(population=coords, k=n_leps)
        else:
            self.leprechaun_coords = leprechaun_coords
        
        if agent_coord == None:
            if stair_coord == None:
                self.agent_coord, self.stair_coord = tuple(random.sample(population=coords, k=2))
            else:
                self.stair_coord = stair_coord
                coords.remove(self.stair_coord)
                self.agent_coord = random.sample(population=coords, k=1)[0]
        else:
            self.agent_coord = agent_coord
            if stair_coord == None:
                coords.remove(self.agent_coord)
                self.stair_coord = random.sample(population=coords, k=1)[0]
            else:
                self.stair_coord = stair_coord
        
        self.matrix_map = None
        self.pixel = None
        self.message = None
        self.gold_picked = False

        # Generate the des file using the level generator -------------------------------------

        level_generator = LevelGenerator(w=self.width, h=self.height)
        level_generator.set_start_pos(coord=self._coords_to_idxs(self.agent_coord))
        level_generator.add_goal_pos(place=self._coords_to_idxs(self.stair_coord))
        for gold_coord in self.gold_coords:
            level_generator.add_gold(amount=1, place=self._coords_to_idxs(gold_coord))
        for leprechaun_coord in self.leprechaun_coords:
            level_generator.add_monster(name='leprechaun', place=self._coords_to_idxs(leprechaun_coord), args=['awake', 'hostile'])
        
        des_file = level_generator.get_des()

        # Define the reward manager -----------------------------------------------------------

        reward_manager = RewardManager()

        def my_reward_function(env:MiniHackGoldRoom, previous_observation: Any, action: int, current_obsetrvation: Any) -> float:
            reward = env.time_penalty * np.linalg.norm(action_to_move(action))

            if 'Your purse feels lighter' in env.message:
                reward -= env.collected_gold
            
            if env.gold_picked:
                reward += env.gold_score
            
            if env.agent_coord == env.stair_coord:
                reward += env.stair_score

            return reward
        
        reward_manager.add_custom_reward_fn(my_reward_function)
        reward_manager.add_coordinate_event(
            coordinates=(self.stair_coord[0], self.height - self.stair_coord[1] - 1),
            reward=self.stair_score,
            repeatable=False,
            terminal_required=True,
            terminal_sufficient=True
        )

        # Call the constructor of the superclass ----------------------------------------------

        super().__init__(
            *args, des_file=des_file, reward_manager=reward_manager,
            actions=ACTIONS, allow_all_yn_questions=False, allow_all_modes=False,
            character='rog-hum-cha-mal', max_episode_steps=max_episode_steps,
            observation_keys=('blstats', 'chars', 'pixel', 'message')
        )

    register(
        id="MiniHack-MyTask-Custom-v0",
        entry_point="gold_room_env:MiniHackGoldRoom",
    )

    # Copy method ----------------------------------------------------------------------------------------------------
    def copy(self):
        return gym.make(
            'MiniHack-MyTask-Custom-v0',
            width=self.width,
            height=self.height,
            max_episode_steps=self.max_episode_steps,
            gold_score=self.gold_score,
            stair_score=self.stair_score,
            time_penalty=self.time_penalty,
            agent_coord=self.agent_coord,
            stair_coord=self.stair_coord,
            gold_coords=self.gold_coords,
            leprechaun_coords=self.leprechaun_coords
            )

    # Reset the environment  -------------------------------------------------------------------------------------------
    def myreset(self) -> Tuple[dict, float]:
        minihack_state = self.reset()
        non_empty_rows = ~np.all(minihack_state['chars'] == 32, axis=1)
        non_empty_cols = ~np.all(minihack_state['chars'] == 32, axis=0)
        self.matrix_map = minihack_state['chars'][non_empty_rows][:, non_empty_cols]
        self.agent_coord = self._get_agent_coord()
        self.gold_coords = self.get_gold_coords()
        #self.stair_coord = self._get_stair_coord()
        self.pixel = minihack_state['pixel']
        self.message = bytes(minihack_state['message']).decode('utf-8').rstrip('\x00')
        self.instant += 1

        if self.agent_coord in self.gold_coords:
            reward = self.gold_score
            self.collected_gold += self.gold_score
            self.gold_coords.remove(self.agent_coord)
        else:
            reward = 0.0
        
        return self.state(), reward

    # Perform a step in the environment --------------------------------------------------------------------------------
    def mystep(self, action: int) -> Tuple[dict, float, bool]:
        self.instant += 1
        self.agent_coord = tuple(np.array(self.agent_coord) + action_to_move(action=action))
        if self.agent_coord in self.gold_coords:#TBR TODO: lep
            self.collected_gold += self.gold_score
            self.gold_coords.remove(self.agent_coord)
            self.gold_picked = True

        minihack_state, reward, done, info = self.step(action)

        self.gold_picked = False

        non_empty_rows = ~np.all(minihack_state['chars'] == 32, axis=1)
        non_empty_cols = ~np.all(minihack_state['chars'] == 32, axis=0)
        self.matrix_map = minihack_state['chars'][non_empty_rows][:, non_empty_cols]
        self.pixel = minihack_state['pixel']
        self.message = bytes(minihack_state['message']).decode('utf-8').rstrip('\x00')
    
        return self.state(), reward, done#, info


    #def _agent_idxs(self):
    #    x, y = np.where(self.curr_state['map'] == AGENT_CHAR)
    #    return x[0], y[0]
    #
    #def _get_agent_idxs(self):
    #    return np.where(self.curr_state['map'] == AGENT_CHAR)
    #
    #def _get_stair_idxs(self):
    #    x, y = self.stair_coord
    #    return y - self.height + 1, x
    
    def _get_agent_coord(self):
        x, y = np.where(self.matrix_map == AGENT_CHAR)
        return y[0], self.height - x[0] - 1
    
    def get_leprechaun_coords(self):
        rows, cols = np.where(self.matrix_map == LEPRECHAUN_CHAR)
        coords = [(col, self.height - row - 1) for row, col in zip(rows, cols)]
        return coords
    
    def get_gold_coords(self):
        rows, cols = np.where(self.matrix_map == GOLD_CHAR)
        coords = [(col, self.height - row - 1) for row, col in zip(rows, cols)]
        return coords
    
    #def _get_stair_coord(self):
    #    x, y = np.where(self.matrix_map == STAIR_CHAR)
    #    print((x, y))
    #    return int(y[0]), int(self.height - x[0] - 1)

    def state(self):
        #stats = minihack_state['blstats']
        #stats_dict = {
        #    'health': stats[HEALTH],
        #    'gold': stats[GOLD],
        #    'time': stats[TIME],
        #    'carrying_capacity': stats[CARRYING_CAPACITY]
        #}
        env_state = {
            #'health': stats[HEALTH],
            'gold': self.collected_gold,#stats[GOLD],
            'time': self.instant,#stats[TIME],
            #'carrying_capacity': stats[CARRYING_CAPACITY],
            'agent_coord': self.agent_coord,
            'stair_coord': self.stair_coord,
            'gold_coords': self.gold_coords,
            'map': self.matrix_map,
            'pixel': self.pixel,
            'message': self.message
        }
        return env_state


    #def _add_leprechaun(self, level_generator, coord):
    #    x, y = coord
    #    col_idx = x
    #    row_idx = self.height - y - 1
    #    if x < 0:
    #        raise IndexError('invalid argument x')
    #    elif x >= self.width:
    #        raise IndexError('x out of bound')
    #    if y < 0:
    #        raise IndexError('invalid argument y')
    #    elif y >= self.height:
    #        raise IndexError('y out of bound')
    #    level_generator.add_monster(name='leprechaun', place=(int(col_idx), int(row_idx)), args=['awake', 'hostile'])#TBR
    #    return x, y
#
#
    #def _add_gold(self, level_generator, coord):
    #    x, y = coord
    #    col_idx = x
    #    row_idx = self.height - y - 1
    #    if x < 0:
    #        raise IndexError('invalid argument x')
    #    elif x >= self.width:
    #        raise IndexError('x out of bound')
    #    if y < 0:
    #        raise IndexError('invalid argument y')
    #    elif y >= self.height:
    #        raise IndexError('y out of bound')
    #    level_generator.add_gold(amount=1, place=(int(col_idx), int(row_idx))) #TBR
    #    return x, y

    def _coords_to_idxs(self, coord):
        x, y = coord
        col_idx = x
        row_idx = self.height - y - 1
        return int(col_idx), int(row_idx)

    #def _set_start_position(self, level_generator, x=-1, y=-1):
    #    col_idx = x
    #    row_idx = self.height - y - 1
    #    if x < 0:
    #        col_idx = random.randint(0, self.width-1)
    #    elif x >= self.width:
    #        raise IndexError('x out of bound')
    #    if y < 0:
    #        row_idx = random.randint(0, self.height-1)
    #    elif y >= self.height:
    #        raise IndexError('y out of bound')
    #    level_generator.set_start_pos((int(col_idx), int(row_idx)))
    #    x = col_idx
    #    y = self.height - row_idx - 1
    #    return x, y
#
#
    #def _set_stair_position(self, level_generator, x=-1, y=-1):
    #    col_idx = x
    #    row_idx = self.height - y - 1
    #    
    #    if x < 0:
    #        col_idx = random.randint(0, self.width-1)
#
    #    elif x >= self.width:
    #        raise IndexError('x out of bound')
#
    #    if y < 0:
    #        row_idx = random.randint(0, self.height-1)
    #        while row_idx == self.height - self.start_coord[1] - 1:
    #            row_idx = random.randint(0, self.height-1)
#
    #    elif y >= self.height:
    #        raise IndexError('y out of bound')
#
    #    level_generator.add_goal_pos((int(col_idx), int(row_idx)))
    #    x = col_idx
    #    y = self.height - row_idx - 1
    #    return x, y