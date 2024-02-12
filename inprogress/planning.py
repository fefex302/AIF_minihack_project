import time
import matplotlib.pyplot as plt
import random
import numpy as np
from queue import PriorityQueue
from gold_room_env import MiniHackGoldRoom
from utils import action_to_string, action_to_move, move_to_action
from typing import Callable, Tuple, List
import gym


N_ARR = np.array([0, 1])
S_ARR = np.array([0, -1])
W_ARR = np.array([-1, 0])
E_ARR = np.array([1, 0])
NE_ARR = N_ARR + E_ARR
SE_ARR = S_ARR + E_ARR
SW_ARR = S_ARR + W_ARR
NW_ARR = N_ARR + W_ARR
MOVES = [N_ARR, E_ARR, S_ARR, W_ARR, NE_ARR, SE_ARR, SW_ARR, NW_ARR]

N = 0
E = 1
S = 2
W = 3
NE = 3 + N + E
SE = 2 + S + E
SW = 1 + S + W
NW = 4 + N + W
ACTIONS = [N, E, S, W, NE, SE, SW, NW]

ACTION_NAMES = ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW']


class Plan():
    def __init__(self):
        self.path = []
        self.action_sequence = []
    
    def add_reverse(self, action: List[int], coords: Tuple[int, int]) -> None:
        self.action_sequence = action + self.action_sequence
        moves = [action_to_move(a) for a in action]
        self.path.insert(0, coords)

        actual_coords = np.array(coords)
        for move in reversed(moves[1:]):
            actual_coords -= move
            self.path.insert(0, tuple(actual_coords))
    
    def show(self) -> None:
        print(f'Path: {self.path}')
        print(f'Actions: {[action_to_string(action) for action in self.action_sequence]}')

    
class State:
    def __init__(
        self,
        agent_coords: Tuple[int, int],
        gold_coords: List[Tuple[int, int]], 
        stair_coords: Tuple[int, int]
        ):

        self.agent_coords = agent_coords
        self.gold_coords = gold_coords
        self.stair_coords = stair_coords

    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return self.agent_coords == other.agent_coords and self.gold_coords == other.gold_coords and self.stair_coords == other.stair_coords
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.agent_coords, tuple(self.gold_coords), self.stair_coords))


class Node:
    def __init__(
        self,
        state: State,
        g_value: float = None,
        priority: float = None,
        parent: 'Node' = None,
        action: List[int] = []
        ):

        self.state = state
        self.g_value = g_value
        self.priority = priority
        self.parent = parent
        self.action = action

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __lt__(self, other) -> bool:
        return self.priority > other.priority
    
    def __hash__(self):
        return hash(self.state)


class GFunction:
    def __init__(
        self,
        gold_score: float,
        time_penalty: float,
        stair_score: float,
        prev_state: State = None,
        prev_g: float = 0.0,
        g_function: Callable[[float, float, float, float, State, State], float] = None
        ):

        self.gold_score = gold_score
        self.time_penalty = time_penalty
        self.stair_score = stair_score
        self.prev_state = prev_state
        self.prev_g = prev_g
        self.g_function = g_function
    
    def __call__(self, state: State) -> float:
        return self.g_function(
            gold_score=self.gold_score,
            time_penalty=self.time_penalty,
            stair_score=self.stair_score,
            prev_g=self.prev_g,
            prev_state=self.prev_state,
            state=state
            )


class HFunction:
    def __init__(
        self,
        gold_score: float,
        time_penalty: float,
        stair_score: float,
        h_function: Callable[[float, float, float, State], float] = None
        ):

        self.gold_score = gold_score
        self.time_penalty = time_penalty
        self.stair_score = stair_score
        self.h_function = h_function
    
    def __call__(self, state: State) -> float:
        return self.h_function(
            gold_score=self.gold_score, 
            time_penalty=self.time_penalty, 
            stair_score=self.stair_score, 
            state=state
            )


class AllowedMovesFunction:
    def __init__(
        self,
        width: int,
        height: int,
        to_avoid: Tuple[int, int] = None,
        function: Callable[[int, int, State], List[np.ndarray[int]]] = None
        ):

        self.width = width
        self.height = height
        self.function = function
        self.to_avoid = to_avoid
    
    def __call__(self, state: State, to_avoid: Tuple[int, int] = None) -> List[np.ndarray[int]]:
        return self.function(
            width=self.width,
            height=self.height,
            to_avoid=self.to_avoid,
            state=state
        )


def h(gold_score: float, time_penalty: float, stair_score: float, state: State) -> float:
    agent_stair_dist = np.linalg.norm(np.array(state.agent_coords) - np.array(state.stair_coords))
    actual_golds = [g for g in state.gold_coords if g != state.agent_coords and g != state.stair_coords]
    gold_in_stair = (state.stair_coords in state.gold_coords)
    n_golds = len(actual_golds)
    if agent_stair_dist == 0:
        return 0
    strategy1_score = time_penalty * agent_stair_dist + stair_score + gold_score * gold_in_stair
    if n_golds == 0:
        return strategy1_score
    agent_gold_dists = [np.linalg.norm(np.array(state.agent_coords) - np.array(gold_coords)) for gold_coords in actual_golds]
    gold_stair_dists = [np.linalg.norm(np.array(state.stair_coords) - np.array(gold_coords)) for gold_coords in actual_golds]
    path_lenghts = [ag + gs for ag, gs in zip(agent_gold_dists, gold_stair_dists)]
    min_path_len = min(path_lenghts)
    strategy2_score = time_penalty * (min(path_lenghts)) + gold_score * n_golds + stair_score + gold_score * gold_in_stair
    return max([strategy1_score, strategy2_score])


def g(gold_score: float, time_penalty: float, stair_score: float, prev_g: float, prev_state: State, state: State) -> float:
    return prev_g + \
        gold_score * (state.agent_coords in state.gold_coords) + \
        stair_score * (state.agent_coords == state.stair_coords) + \
        time_penalty * np.linalg.norm(np.array(state.agent_coords) - np.array(prev_state.agent_coords))


def allowed_moves(width: int, height: int, state: State, to_avoid: Tuple[int, int] = None) -> List[np.ndarray[int]]:
    x, y = state.agent_coords
    n = 0b1000
    s = 0b0100
    w = 0b0010
    e = 0b0001
    b = 0b0000
    nw = n | w
    ne = n | e
    sw = s | w
    se = s | e
    moves = []
    if x-1 >= 0:
        b |= w
        moves.append(W_ARR)
    if x+1 < height:
        b |= e
        moves.append(E_ARR)
    if y-1 >= 0:
        b |= s
        moves.append(S_ARR)
    if y+1 < width:
        b |= n
        moves.append(N_ARR)
    if b & (n | w) == nw:
        moves.append(NW_ARR)
    if b & (n | e) == ne:
        moves.append(NE_ARR)
    if b & (s | w) == sw:
        moves.append(SW_ARR)
    if b & (s | e) == se:
        moves.append(SE_ARR)
    
    moves = [m for m in moves if tuple(state.agent_coords + m) != to_avoid]

    return moves


def allowed_supermoves(state: State, width: int = None, height: int = None, to_avoid: Tuple[int, int] = None) -> List[np.ndarray[int]]:
    del width, height, to_avoid
    return [np.array(state.stair_coords) - np.array(state.agent_coords)] + [np.array(g_coords) - np.array(state.agent_coords) for g_coords in state.gold_coords if state.agent_coords != g_coords]


def is_composite(move: np.ndarray[int]) -> bool:
    for m in MOVES:
        if np.array_equal(move, m):
            return False
    return True


def a_star(
    env: MiniHackGoldRoom,
    allowed_moves_fun: Callable[[int, int, State], List[np.ndarray[int]]],
    g_function: Callable[[float, float, float, float, State], float],
    h_function: Callable[[float, float, float, State], float],
    to_avoid: Tuple[int, int] = None
    ) -> Tuple[Plan, int]:

    g = GFunction(
        gold_score=env.gold_score,
        time_penalty=env.time_penalty,
        stair_score=env.stair_score,
        g_function=g_function
        )
    
    h = HFunction(
        gold_score=env.gold_score,
        time_penalty=env.time_penalty,
        stair_score=env.stair_score,
        h_function=h_function
    )

    moves_from_state = AllowedMovesFunction(
        width=env.width,
        height=env.height,
        to_avoid=to_avoid,
        function=allowed_moves_fun
    )

    _, init_g = env.myreset()
    
    init_state = State(
        agent_coords=env.agent_coord,
        gold_coords=env.gold_coords,
        stair_coords=env.stair_coord
    )

    init_node = Node(
        state = init_state,
        g_value = init_g,
        priority = init_g + h(init_state),
        parent = None,
        action=[]
    )

    expanded_nodes = set()
    nodes_queue = PriorityQueue()
    nodes_queue.put(init_node)

    additional_expanded_nodes = 0

    while not nodes_queue.empty():
        node = nodes_queue.get()
        expanded_nodes.add(node)
        stair_reached = (node.state.agent_coords == node.state.stair_coords)
        if stair_reached:
            final_node = node
            break
        g.prev_g = node.g_value
        g.prev_state = node.state
        moves = moves_from_state(node.state)
        reachable_points = [tuple(np.array(node.state.agent_coords) + move) for move in moves]
        actual_golds = [gold_coords for gold_coords in node.state.gold_coords if gold_coords != node.state.agent_coords]

        for move, point in zip(moves, reachable_points):
            if is_composite(move):
                
                reachable_state = State(
                    agent_coords=tuple(point),
                    gold_coords=actual_golds,
                    stair_coords=env.stair_coord
                )

                reachable_node = Node(
                    state = reachable_state,
                    g_value = g(reachable_state),
                    priority = g(reachable_state) + h(reachable_state),
                    parent = node
                )
                
                if reachable_node not in expanded_nodes:

                    env2 = gym.make(
                        'MiniHack-MyTask-Custom-v0',
                        width=env.width,
                        height=env.height,
                        n_leps=0,
                        max_episode_steps=env.max_episode_steps,
                        stair_score=env.gold_score,
                        stair_coord=tuple(point),
                        agent_coord=node.state.agent_coords,
                        time_penalty=env.time_penalty
                        )

                    in_stair = (env2.stair_coord == env.stair_coord)
                    if in_stair:
                        to_avoid = None
                    else:
                        to_avoid = node.state.stair_coords

                    subplan, n_expanded_nodes = a_star(
                        env=env2,
                        allowed_moves_fun=allowed_moves,
                        g_function=g_function,
                        h_function=h_function,
                        to_avoid=to_avoid
                        )

                    additional_expanded_nodes += n_expanded_nodes

                    intersection = [gold for gold in actual_golds if gold in subplan.path]

                    reachable_state_golds = [gold for gold in actual_golds if gold not in intersection or gold == tuple(point)]

                    path_score = node.g_value + env.gold_score * len(intersection) + env.stair_score * in_stair

                    for a in subplan.action_sequence:
                        path_score += (np.linalg.norm(action_to_move(a)) * env.time_penalty)

                    reachable_state = State(
                        agent_coords=tuple(point),
                        gold_coords=reachable_state_golds,
                        stair_coords=env.stair_coord
                    )   

                    reachable_node = Node(
                        state = reachable_state,
                        g_value = path_score,
                        priority = path_score + h(reachable_state),
                        parent = node,
                        action = subplan.action_sequence
                    )

                    if reachable_node not in expanded_nodes:
                        nodes_queue.put(reachable_node)

            else:
                reachable_state = State(
                    agent_coords=tuple(point),
                    gold_coords=actual_golds,
                    stair_coords=env.stair_coord
                )

                reachable_node = Node(
                    state = reachable_state,
                    g_value = g(reachable_state),
                    priority = g(reachable_state) + h(reachable_state),
                    parent = node,
                    action = [move_to_action(move)]
                )

                if reachable_node not in expanded_nodes:
                        nodes_queue.put(reachable_node)

    plan = Plan()
    node = final_node
    while node != None:
        plan.add_reverse(action=node.action, coords=node.state.agent_coords)
        node = node.parent

    return plan, len(expanded_nodes) + additional_expanded_nodes


#def hill_climbing(
    #env: MiniHackGoldRoom,
    #allowed_moves_fun: Callable[[int, int, State], List[np.ndarray[int]]],
    #value_function: Callable[Node, float]
    #) -> Tuple[Plan, int]:

    #moves_from_state = AllowedMovesFunction(
    #    width=env.width,
    #    height=env.height,
    #    function=allowed_moves_fun
    #)
#
    #_, init_g = env.myreset()
    #
    #init_state = State(
    #    agent_coords=env.agent_coord,
    #    gold_coords=env.gold_coords,
    #    stair_coords=env.stair_coord
    #)
#
    #init_node = Node(
    #    state = init_state,
    #    g_value = init_g,
    #    parent = None,
    #    action=[]
    #)
#
    #node_value = value_function(init_node)
#
    #stair_reached = (init_node.state.agent_coords == init_node.state.stair_coords)

    #while not stair_reached:
    #for _ in range(0, 10):
    #    state, reward, stair_reached, info = env.mystep(action=N)
    #    print(state['time'])
    #    print(state['gold'])



def random_search(
    env: MiniHackGoldRoom,
    allowed_moves_fun: Callable[[int, int, State], List[np.ndarray[int]]],
    max_steps=10
    ) -> Tuple[List[dict], List[float], bool]:

    moves_from_state = AllowedMovesFunction(
        width=env.width,
        height=env.height,
        function=allowed_moves_fun
    )

    _, init_reward = env.myreset()

    states = [env.curr_state]
    rewards = [init_reward]
    stair_reached = (env.agent_coord == env.stair_coord)

    for _ in range(0, max_steps):
        mystate = State(
            agent_coords=env.agent_coord,
            gold_coords=env.gold_coords,
            stair_coords=env.stair_coord
        )
        actions = [move_to_action(move) for move in moves_from_state(mystate)]
        print([action_to_string(a) for a in actions])
        state, reward, stair_reached = env.mystep(
            action=random.sample(
                population=actions,
                k=1)[0]
            )
        states.append(state)
        rewards.append(reward)
        if stair_reached:
            break

    return states, rewards, stair_reached


def apply(env: MiniHackGoldRoom, plan: Plan) -> Tuple[List[dict], List[float], bool]:
    if env.agent_coord in env.gold_coords:
        reward = env.gold_score
    else:
        reward = 0
    states = [env.curr_state]
    rewards = [reward]
    done = (env.agent_coord == env.stair_coord)
    for action in plan.action_sequence:
        state, reward, done = env.mystep(action=action)
        states.append(state)
        rewards.append(reward)
    return states, rewards, done