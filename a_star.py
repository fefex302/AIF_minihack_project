import math
from pyswip import Prolog



#we start with expanding the cells around the player that are navigable, then we define the
#cost value for each cell not only based on the distance with respect to the goal, but we need to ask the knowledge base
#if that cell is 'safe' enough to be naviagable
def node_expansion(reward_pos: tuple,player_pos: tuple, map_bounds: tuple, kb: Prolog):
    
    #the actual cost of the path, it starts with inf
    cost = float('inf')

    #current best node
    best_node = player_pos

    #cells around the player
    radius = 1

    for x in range(player_pos[0] - radius, player_pos[0] + radius + 1):
        for y in range(player_pos[1] - radius, player_pos[1] + radius + 1):
            #check if the given position is a valid position
            #if is_valid((x,y),map_bounds):
            #compute the cost and compare it to the previous one
            if (x,y) != player_pos:
                tmp = compute_cost((x,y),reward_pos,kb)
                # print(f'node: {x},{y}')
                # print(f'cost: {tmp}')
                if tmp  < cost:
                    cost = tmp
                    best_node = (x,y)
    print(best_node)
    return best_node




def is_valid(position: tuple, map_bounds: tuple):

    if 0 <= position[0] < map_bounds[0] and 0 <= position[1] < map_bounds[1]:
        return True
    else:
        return False

def look_around(kb: Prolog, obs: dict):

    kb.retractall("position(_,_,_,_)")
    for i in range(21):
        for j in range(79):
            if not (obs['screen_descriptions'][i][j] == 0).all():
                print('searching')
                obj = bytes(obs['screen_descriptions'][i][j]).decode('utf-8').rstrip('\x00')
                if 'apple' in obj:
                    if bool(list(kb.query(f'position(apple,commestible,_,_)'))) == False:
                        print('apple stored')
                        kb.asserta(f'position(apple, commestible, {j}, {i})')

def euclidean_distance(first: tuple,second: tuple):
    
    distance = math.sqrt(math.pow(first[0]-second[0],2) + math.pow(first[1]-second[1],2))
    return distance


#function that will compute the cost (euclidian distance + kb inference)
def compute_cost(position: tuple, reward_pos: tuple, kb: Prolog):

    return euclidean_distance(position,reward_pos)



def direction_id(player_position: tuple, next_position: tuple):
    #  elif 'northeast' in action: action_id = 4
    # elif 'southeast' in action: action_id = 5
    # elif 'southwest' in action: action_id = 6
    # elif 'northwest' in action: action_id = 7
    # elif 'north' in action: action_id = 0
    # elif 'east' in action: action_id = 1
    # elif 'south' in action: action_id = 2
    # elif 'west' in action: action_id = 3
    action_id = 0

    diff_x = player_position[0] - next_position[0]
    diff_y = player_position[1] - next_position[1]


    #northeast
    if diff_x == -1 and diff_y == 1: action_id = 4
    #north
    elif diff_x == 0 and diff_y == 1: action_id = 0
    #northwest
    elif diff_x == 1 and diff_y == 1: action_id = 7
    #west
    elif diff_x == 1 and diff_y == 0: action_id = 3
    #east
    elif diff_x == -1 and diff_y == 0: action_id = 1
    #southwest
    elif diff_x == 1 and diff_y == -1: action_id = 6
    #south
    elif diff_x == 0 and diff_y == -1: action_id = 2
    #southeast
    elif diff_x == -1 and diff_y == -1: action_id = 5
    # #current position
    # #elif diff_x == 0 and diff_y == 0: action_id = 


    return action_id



#get the direction id given reward position and player position
def get_direction(reward_pos: tuple,player_pos: tuple, map_bounds: tuple, kb: Prolog):

    node = node_expansion(reward_pos,player_pos,map_bounds,kb)

    return direction_id(player_pos,node)

    