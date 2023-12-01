import gym
import time
import matplotlib.pyplot as plt
from pyswip import Prolog
#from utils import create_level, define_reward
import IPython.display as display
from minihack import LevelGenerator
from minihack import RewardManager
import minihack
import random


#creation of a leve
def create_level(width: int, height: int, num_cloud: int = 4, monster: str = 'kobold', trap: str = 'falling rock', weapon: str = 'tsurugi'):

    lvl = LevelGenerator(w=width, h=height)
    lvl.add_object(name='apple', symbol='%')

    #we put a number of cloud equals to the parameter 'num_cloud' which is 4 by default
    for i in range(num_cloud):
        randx = random.randint(0, 9)
        randy = random.randint(0, 9)
        lvl.add_terrain((randx,randy),'C')

    #lvl.add_monster(name=monster)
    #lvl.add_monster(name='minotaur',args=('hostile','asleep'),place=(5,5))
    #lvl.add_stair_down(place=(5,5))

    return lvl.get_des()
