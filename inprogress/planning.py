import random
import numpy as np
from queue import PriorityQueue
from gold_room_env import MiniHackGoldRoom
from utils import action_to_string, action_to_move, move_to_action, allowed_moves, is_composite, AllowedMovesFunction, AllowedSimpleMovesFunction, ALLOWED_SIMPLE_MOVES
from typing import Callable, Tuple, List
import gym

class Plan:
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
    
    def show(self, env: MiniHackGoldRoom) -> None:
        print(f'Path: {self.path}')
        print(f'Actions: {[action_to_string(action) for action in self.action_sequence]}')
        total_score = sum([np.linalg.norm(action_to_move(action)) * env.time_penalty for action in self.action_sequence]) + env.gold_score * len([coord for coord in env.gold_coords if coord in self.path])
        print(f'Total score: {round(total_score, 3)}')


    
class State:
    def __init__(
        self,
        agent_coord: Tuple[int, int],
        stair_coord: Tuple[int, int],
        gold_coords: List[Tuple[int, int]]
        ):

        self.agent_coord = agent_coord
        self.gold_coords = gold_coords
        self.stair_coord = stair_coord

    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return self.agent_coord == other.agent_coord and self.gold_coords == other.gold_coords and self.stair_coord == other.stair_coord
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.agent_coord, tuple(self.gold_coords), self.stair_coord))
    
    def to_dict(self) -> dict:
        return {
            'agent_coord': self.agent_coord,
            'stair_coord': self.stair_coord,
            'gold_coords': self.gold_coords
            }


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
        gold_score: float = None,
        time_penalty: float = None,
        stair_score: float = None,
        prev_state: State = None,
        prev_g: float = 0.0,
        f: Callable[State, float] = None
        ):

        self.gold_score = gold_score
        self.time_penalty = time_penalty
        self.stair_score = stair_score
        self.prev_state = prev_state
        self.prev_g = prev_g
        self.f = f
    
    def __call__(self, state: State) -> float:
        if self.f != None:
            return self.f(state=state)
        return self._g(state=state)
    
    def _g(self, state: State) -> float:
        return self.prev_g + \
            self.gold_score * (state.agent_coord in state.gold_coords) + \
            self.stair_score * (state.agent_coord == state.stair_coord) + \
            self.time_penalty * np.linalg.norm(np.array(state.agent_coord) - np.array(self.prev_state.agent_coord))
    
    def set_for_env(self, env: MiniHackGoldRoom):
        self.gold_score = env.gold_score
        self.time_penalty = env.time_penalty
        self.stair_score = env.stair_score



class HFunction:
    def __init__(
        self,
        gold_score: float = None,
        time_penalty: float = None,
        stair_score: float = None,
        f: Callable[State, float] = None
    ):

        self.gold_score = gold_score
        self.time_penalty = time_penalty
        self.stair_score = stair_score
        self.f = f
    
    def __call__(self, state: State) -> float:
        if self.f != None:
            return self.f(state=state)
        return self._h(state=state)

    def _h(self, state: State) -> float:
        agent_stair_dist = np.linalg.norm(np.array(state.agent_coord) - np.array(state.stair_coord))
        actual_golds = [g for g in state.gold_coords if g != state.agent_coord and g != state.stair_coord]
        gold_in_stair = (state.stair_coord in state.gold_coords)
        n_golds = len(actual_golds)
        if agent_stair_dist == 0:
            return 0
        strategy1_score = self.time_penalty * agent_stair_dist + self.stair_score + self.gold_score * gold_in_stair
        if n_golds == 0:
            return strategy1_score
        agent_gold_dists = [np.linalg.norm(np.array(state.agent_coord) - np.array(gold_coords)) for gold_coords in actual_golds]
        gold_stair_dists = [np.linalg.norm(np.array(state.stair_coord) - np.array(gold_coords)) for gold_coords in actual_golds]
        path_lenghts = [ag + gs for ag, gs in zip(agent_gold_dists, gold_stair_dists)]
        min_path_len = min(path_lenghts)
        strategy2_score = self.time_penalty * (min(path_lenghts)) + self.gold_score * n_golds + self.stair_score + self.gold_score * gold_in_stair
        return max([strategy1_score, strategy2_score])

    def set_for_env(self, env: MiniHackGoldRoom):
        self.gold_score = env.gold_score
        self.time_penalty = env.time_penalty
        self.stair_score = env.stair_score


def a_star_search(
    env: MiniHackGoldRoom,
    g: GFunction = None,
    h: HFunction = None,
    allowed_moves_function: AllowedMovesFunction = ALLOWED_SIMPLE_MOVES
) -> Tuple[Plan, int]:

    if not isinstance(allowed_moves_function, AllowedMovesFunction):
        raise ValueError('Parameter allowed_moves_function must be of type AllowedMovesFunction')
    
    if isinstance(allowed_moves_function, AllowedSimpleMovesFunction):
        allowed_moves_function.width = env.width
        allowed_moves_function.height = env.height
    
    if g == None:
        g = GFunction()
    if h == None:
        h = HFunction()
    
    _, init_g = env.myreset()

    g.set_for_env(env=env)
    h.set_for_env(env=env)

    init_state = State(
        agent_coord=env.agent_coord,
        gold_coords=env.gold_coords,
        stair_coord=env.stair_coord
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

    support_dict = {}
    support_dict[init_node] = init_g

    additional_expanded_nodes = 0

    while not nodes_queue.empty():
        node = nodes_queue.get()
        expanded_nodes.add(node)
        stair_reached = (node.state.agent_coord == node.state.stair_coord)
        if stair_reached:
            final_node = node
            break
        g.prev_g = node.g_value
        g.prev_state = node.state
        moves = allowed_moves_function(state=node.state.to_dict())
        reachable_points = [tuple(np.array(node.state.agent_coord) + move) for move in moves]
        actual_golds = [gold_coords for gold_coords in node.state.gold_coords if gold_coords != node.state.agent_coord]

        for move, point in zip(moves, reachable_points):
            if is_composite(move):
                
                reachable_state = State(
                    agent_coord=tuple(point),
                    gold_coords=actual_golds,
                    stair_coord=env.stair_coord
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
                        agent_coord=node.state.agent_coord,
                        time_penalty=env.time_penalty
                        )

                    in_stair = (env2.stair_coord == env.stair_coord)
                    if in_stair:
                        to_avoid = None
                    else:
                        to_avoid = env.stair_coord
                    
                    sub_allowed_moves_function = AllowedSimpleMovesFunction(
                        width=env2.width,
                        height=env2.height,
                        to_avoid=to_avoid
                    )

                    subplan, n_expanded_nodes = a_star_search(
                        env=env2,
                        allowed_moves_function=sub_allowed_moves_function
                        )

                    additional_expanded_nodes += n_expanded_nodes

                    intersection = [gold for gold in actual_golds if gold in subplan.path]

                    reachable_state_golds = [gold for gold in actual_golds if gold not in intersection or gold == tuple(point)]

                    path_score = node.g_value + env.gold_score * len(intersection) + env.stair_score * in_stair

                    for a in subplan.action_sequence:
                        path_score += (np.linalg.norm(action_to_move(a)) * env.time_penalty)

                    reachable_state = State(
                        agent_coord=tuple(point),
                        gold_coords=reachable_state_golds,
                        stair_coord=env.stair_coord
                    )   

                    reachable_node = Node(
                        state = reachable_state,
                        g_value = path_score,
                        parent = node,
                        action = subplan.action_sequence
                    )

                    if reachable_node not in expanded_nodes:
                        if reachable_node not in support_dict.keys() or (reachable_node in support_dict.keys() and reachable_node.g_value > support_dict[reachable_node]):
                            reachable_node.priority = reachable_node.g_value + h(reachable_state)
                            nodes_queue.put(reachable_node)
                            support_dict[reachable_node] = reachable_node.g_value

            else:
                reachable_state = State(
                    agent_coord=tuple(point),
                    gold_coords=actual_golds,
                    stair_coord=env.stair_coord
                )

                reachable_node = Node(
                    state = reachable_state,
                    g_value = g(reachable_state),
                    parent = node,
                    action = [move_to_action(move)]
                )

                if reachable_node not in expanded_nodes:
                    if reachable_node not in support_dict.keys() or (reachable_node in support_dict.keys() and reachable_node.g_value > support_dict[reachable_node]):
                            reachable_node.priority = reachable_node.g_value + h(reachable_state)
                            nodes_queue.put(reachable_node)
                            support_dict[reachable_node] = reachable_node.g_value

    plan = Plan()
    node = final_node
    while node != None:
        plan.add_reverse(action=node.action, coords=node.state.agent_coord)
        node = node.parent

    return plan, len(expanded_nodes) + additional_expanded_nodes


def uniform_cost_search(
    env: MiniHackGoldRoom,
    allowed_moves_function: AllowedMovesFunction = ALLOWED_SIMPLE_MOVES
) -> Tuple[Plan, int]:

    
    return a_star_search(env=env, h=HFunction(f=lambda state: 0), allowed_moves_function=allowed_moves_function)


def greedy_search(
    env: MiniHackGoldRoom,
    allowed_moves_function: AllowedMovesFunction = ALLOWED_SIMPLE_MOVES
) -> Tuple[Plan, int]:

    return a_star_search(env=env, g=GFunction(f=lambda state: 0))


def random_search(
    env: MiniHackGoldRoom,
    allowed_moves_function: AllowedMovesFunction, # TODO: adapt to CompositeMoves
    max_steps=10
    ) -> Tuple[List[dict], List[float], bool]:

    if isinstance(allowed_moves_function, AllowedSimpleMovesFunction):
        allowed_moves_function.width = env.width
        allowed_moves_function.height = env.height

    _, init_reward = env.myreset()

    states = [env.state()]
    rewards = [init_reward]
    stair_reached = (env.agent_coord == env.stair_coord)

    for _ in range(0, max_steps):
        mystate = State(
            agent_coords=env.agent_coord,
            gold_coords=env.gold_coords,
            stair_coord=env.stair_coord
        )
        actions = [move_to_action(move) for move in allowed_moves_function(mystate.to_dict())]
        state, reward, stair_reached = env.mystep(action=random.sample(population=actions, k=1)[0])
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
    states = [env.state()]
    rewards = [reward]
    done = (env.agent_coord == env.stair_coord)
    for action in plan.action_sequence:
        state, reward, done = env.mystep(action=action)
        states.append(state)
        rewards.append(reward)
    return states, rewards, done