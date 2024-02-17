from gold_room_env import MiniHackGoldRoom
from utils import AllowedSimpleMovesFunction
from planning import HFunction

def hill_climbing(env: MiniHackGoldRoom):
    value = HFunction(gold_score=env.gold_score, time_penalty=env.time_penalty, stair_score=env.stair_score)
    allowed_moves = AllowedSimpleMovesFunction(width=env.width, height=env.height)
    state, reward = env.myreset()
    done = False
    while not done:
        moves = allowed_moves(state=state)
        
    
    
# leprechauns moves also while you are moving