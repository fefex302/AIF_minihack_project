import matplotlib.pyplot as plt
import IPython.display as display
import time
from pyswip import Prolog
from minihack import LevelGenerator
from minihack import RewardManager

def create_level(width: int, height: int, monster: str = 'kobold', trap: str = 'falling rock', weapon: str = 'tsurugi'):
    lvl = LevelGenerator(w=width, h=height)
    lvl.add_monster(name=monster)
    lvl.add_trap(name=trap)
    lvl.add_object(name='apple', symbol='%')
    lvl.add_object(name=weapon, symbol=')')

    return lvl.get_des()

def define_reward(monster: str = 'kobold'):
    reward_manager = RewardManager()

    reward_manager.add_eat_event(name='apple', reward=2, terminal_sufficient=True, terminal_required=True)
    reward_manager.add_kill_event(name=monster, reward=1, terminal_required=False)

    return reward_manager

def perform_action(action, env):
    if action == 'eat': 
        action_id = 29
        # print(f'Action performed: {repr(env.actions[action_id])}')
        obs, _, _, _ = env.step(action_id)
        # Example message:
        # What do you want to eat?[g or *]
        message = bytes(obs['message']).decode('utf-8').rstrip('\x00')
        food_char = message.split('[')[1][0] # Because of the way the message in NetHack works
        action_id = env.actions.index(ord(food_char))

    elif action == 'pick': action_id = 49
    elif action == 'wield': action_id = 78

    # Movement/Attack/Run/Get_To_Weapon actions
    # in the end, they all are movement in a direction
    elif 'northeast' in action: action_id = 4
    elif 'southeast' in action: action_id = 5
    elif 'southwest' in action: action_id = 6
    elif 'northwest' in action: action_id = 7
    elif 'north' in action: action_id = 0
    elif 'east' in action: action_id = 1
    elif 'south' in action: action_id = 2
    elif 'west' in action: action_id = 3

    # print(f'Action performed: {repr(env.actions[action_id])}')
    obs, reward, done, info = env.step(action_id)
    return obs, reward, done, info

def process_state(obs: dict, kb: Prolog, monster: str, weapon: str):
    kb.retractall("position(_,_,_,_)")

    # viene analizzata cella per cella (i,j) tutta la mappa
    for i in range(21):
        for j in range(79):
            # si controlla se nella cella (i,j) è presente qualcosa 
            if not (obs['screen_descriptions'][i][j] == 0).all():
                # se è presente qualcosa se ne estrapola il contenuto
                obj = bytes(obs['screen_descriptions'][i][j]).decode('utf-8').rstrip('\x00')
                if 'apple' in obj:
                    kb.asserta(f'position(comestible, apple, {i}, {j})')
                elif monster == obj:
                    kb.asserta(f'position(enemy, {monster.replace(" ", "")}, {i}, {j})')
                elif 'corpse' in obj:
                    kb.asserta(f'position(trap, _, {i}, {j})')
                elif 'sword' in obj:
                    kb.asserta(f'position(weapon, {weapon}, {i}, {j})')

    kb.retractall("wields_weapon(_,_)")
    kb.retractall("has(agent,_,_)")  
    # si itera attraverso gli oggetti nell'inventario dell'agente  
    for obj in obs['inv_strs']:
        # si decodifica l'oggetto dall'inventario
        obj = bytes(obj).decode('utf-8').rstrip('\x00')
        if 'weapon in hand' in obj:
            # the actual name of the weapon is in position 2
            wp = obj.split()[2]
            kb.asserta(f'wields_weapon(agent, {wp})')
        if 'apple' in obj:
            kb.asserta('has(agent, comestible, apple)')

    kb.retractall("position(agent,_,_,_)")
    kb.retractall("health(_)")
    # Asserisce la nuova posizione dell'agente nella knowledge base
    kb.asserta(f"position(agent, _, {obs['blstats'][1]}, {obs['blstats'][0]})")
    # Asserisce la nuova salute dell'agente nella knowledge base
    kb.asserta(f"health({int(obs['blstats'][10]/obs['blstats'][11]*100)})")

    message = bytes(obs['message']).decode('utf-8').rstrip('\x00')
    if 'You see here' in message:
        if 'apple' in message:
            kb.asserta('stepping_on(agent, comestible, apple)')
        if 'sword' in message:
            kb.asserta(f'stepping_on(agent, weapon, {weapon})')

    for m in message.split('.'):
        if 'picks' in m:
            if 'apple' in m:
                print('The enemy took your apple!')

# indexes for showing the image are hard-coded
def show_match(states: list):
    image = plt.imshow(states[0][115:275, 480:750])
    for state in states[1:]:
        time.sleep(0.25)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        image.set_data(state[115:275, 480:750])
    time.sleep(0.25)
    display.display(plt.gcf())
    display.clear_output(wait=True)