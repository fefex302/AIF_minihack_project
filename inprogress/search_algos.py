import planning as pln
import numpy as np

from gold_room_env import MiniHackGoldRoom


def bfs_search(env: MiniHackGoldRoom):
    _,_ = env.myreset()
    init_state = pln.State(env.agent_coord,
                            env.stair_coord,
                            env.gold_coords)
    init_node = pln.Node(state=init_state)

    exp_nodes = set()
    queue_nodes = [init_node]

    while queue_nodes != []:
        node = queue_nodes.pop(0)
        exp_nodes.add(node)

        if node.state.agent_coords == node.state.stair_coords:
            final_node = node
            break

        moves = pln.allowed_moves(env.width,
                                  env.height,
                                  node.state)
        
        reachable_points = [tuple(np.array(node.state.agent_coords)+ move) for move in moves]
        cur_gold = [gold_coords for gold_coords in node.state.gold_coords if gold_coords != node.state.agent_coords]

        for point,move in zip(reachable_points,moves):
            new_state = pln.State(agent_coords=point,
                                  stair_coords=node.state.stair_coords,
                                  gold_coords= cur_gold)
            new_node = pln.Node(state=new_state,
                                parent=node,
                                action=[pln.move_to_action(move)])
            
            if new_node not in exp_nodes:
                queue_nodes.append(new_node)

    plan = pln.Plan()
    node = final_node
    while node != None:
        plan.add_reverse(action=node.action, coords=node.state.agent_coords)
        node = node.parent

    return plan, len(exp_nodes)


def dfs_search(env: MiniHackGoldRoom):
    _,_ = env.myreset()
    init_state = pln.State(env.agent_coord,
                            env.stair_coord,
                            env.gold_coords)
    init_node = pln.Node(state=init_state)

    exp_nodes = set()
    queue_nodes = [init_node]

    while queue_nodes != []:
        node = queue_nodes.pop()
        exp_nodes.add(node)

        if node.state.agent_coords == node.state.stair_coords:
            print("end")
            final_node = node
            break

        moves = pln.allowed_moves(env.width,
                                  env.height,
                                  node.state)
        
        reachable_points = [tuple(np.array(node.state.agent_coords)+ move) for move in moves]
        cur_gold = [gold_coords for gold_coords in node.state.gold_coords if gold_coords != node.state.agent_coords]

        for point,move in zip(reachable_points,moves):
            new_state = pln.State(agent_coords=point,
                                  stair_coords=node.state.stair_coords,
                                  gold_coords= cur_gold)
            new_node = pln.Node(state=new_state,
                                parent=node,
                                action=[pln.move_to_action(move)])
            
            if new_node not in exp_nodes:
                queue_nodes.append(new_node)

    plan = pln.Plan()
    node = final_node
    while node != None:
        plan.add_reverse(action=node.action, coords=node.state.agent_coords)
        node = node.parent

    return plan, len(exp_nodes)