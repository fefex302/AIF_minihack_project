from gold_room_env import MiniHackGoldRoom
from utils import AllowedSimpleMovesFunction

def hill_climbing(env: MiniHackGoldRoom):
    allowed_moves = AllowedSimpleMovesFunction(width=env.width, height=env.height)
    state, reward = env.myreset()
    moves = allowed_moves(state=state)
    moves = [move for move in moves if state['agent_coord']]
# leprechauns moves also while you are moving