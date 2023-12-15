import numpy as np
import math

from typing import Tuple, List

def get_player_location(game_map: np.ndarray, symbol : str = "@") -> Tuple[int, int]:
    # x, y = np.where(game_map == ord(symbol))
    # return (x[0], y[0])
    y, x = np.where(game_map == ord(symbol))
    return (y[0], x[0])

def get_target_location(game_map: np.ndarray, symbol : str = ">") -> Tuple[int, int]:
    # x, y = np.where(game_map == ord(symbol))
    # return (x[0], y[0])
    y, x = np.where(game_map == ord(symbol))
    return (y[0], x[0])

def is_wall(position_element: int) -> bool:
    obstacles = "|- "
    return chr(position_element) in obstacles

def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int]) -> List[Tuple[int, int]]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position    
    # North
    if y - 1 > 0 and not is_wall(game_map[x, y-1]):
        valid.append((x, y-1)) 
    # East
    if x + 1 < x_limit and not is_wall(game_map[x+1, y]):
        valid.append((x+1, y)) 
    # South
    if y + 1 < y_limit and not is_wall(game_map[x, y+1]):
        valid.append((x, y+1)) 
    # West
    if x - 1 > 0 and not is_wall(game_map[x-1, y]):
        valid.append((x-1, y))
    # # NE
    # if y - 1 > 0 and not is_wall(game_map[x, y-1]) and x + 1 < x_limit and not is_wall(game_map[x+1, y]) and not is_wall(game_map[x+1,y-1]):
    #     valid.append(x+1,y-1)
    # # NO
    # if y - 1 > 0 and not is_wall(game_map[x, y-1]) and x - 1 > 0 and not is_wall(game_map[x-1, y]) and not is_wall(game_map[x-1,y-1]):
    #     valid.append(x-1,y-1)
    # # SE
    # if y + 1 < y_limit and not is_wall(game_map[x, y+1]) and x + 1 < x_limit and not is_wall(game_map[x+1, y]) and not is_wall(game_map[x+1,y+1]):
    #     valid.append(x+1,y+1)
    # # SO
    # if y + 1 < y_limit and not is_wall(game_map[x, y+1]) and x - 1 > 0 and not is_wall(game_map[x-1, y]) and not is_wall(game_map[x-1,y+1]):
    #     valid.append(x-1,y+1)
    

    return valid

def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> List[int]:
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3
    }
    actions = []
    # le coordinate x_s e y_s rappresentano le coordinate della posizione del personaggio (che ovviamente si muove quindi cambiano)
    # x_s, y_s = start
    y_s, x_s = start
    for (y, x) in path:
        if x_s == x:
            if y_s > y:
                actions.append(action_map["N"]) #W 
            else: actions.append(action_map["S"]) #E
        elif y_s == y:
            if x_s > x:
                actions.append(action_map["W"])  #N
            else: actions.append(action_map["E"]) #S
        else:
            raise Exception("x and y can't change at the same time. oblique moves not allowed!")
        x_s = x
        y_s = y
    
    return actions

def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)