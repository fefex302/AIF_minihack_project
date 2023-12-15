import numpy as np

from collections import deque
from queue import PriorityQueue
from utils import get_valid_moves
from typing import Tuple, List

def build_path(parent: dict, target: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = ...
    while target is not None:
        ...
        target = ...
    ...
    return path

def bfs(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    # Create a queue for BFS and mark the start node as visited
    queue = ...
    visited = ...
    ...

    # Create a dictionary to keep track of the parent node for each node in the path
    parent = {start: None}

    while queue:
        # Dequeue a vertex from the queue
        current = ...

        # Check if the target node has been reached
        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        # Visit all adjacent neighbors of the dequeued vertex
        for neighbor in get_valid_moves(game_map, current):
            ...

    print("Target node not found!")
    return None

# ---------------------------------------------

def a_star(game_map: np.ndarray, start: Tuple[int, int], target: Tuple[int, int], h: callable) -> List[Tuple[int, int]]:
    # initialize open and close list
    open_list = PriorityQueue()
    close_list = []
    # additional dict which maintains the nodes in the open list for an easier access and check
    support_list = {}

    starting_state_g = ...
    starting_state_h = ...
    starting_state_f = ...

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent = {start: None}

    while not open_list.empty():
        # get the node with lowest f
        ... = open_list.get()
        # add the node to the close list
        close_list.append(current)

        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        for neighbor in get_valid_moves(game_map, current):
            # check if neighbor in close list, if so continue
            ...

            # compute neighbor g, h and f values
            neighbor_g = ...
            neighbor_h = ...
            neighbor_f = ...
            parent[neighbor] = current
            neighbor_entry = (neighbor_f, (neighbor, neighbor_g))
            # if neighbor in open_list
            if neighbor in support_list.keys():
                # if neighbor_g is greater or equal to the one in the open list, continue
                ...

            # add neighbor to open list and update support_list
            ...
            support_list[...] = ...

    print("Target node not found!")
    return None