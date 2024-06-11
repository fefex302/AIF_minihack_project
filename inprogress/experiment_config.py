from utils import ALLOWED_SIMPLE_MOVES, ALLOWED_COMPOSITE_MOVES
from planning import a_star_search, weighted_a_star_search, uniform_cost_search, greedy_search
from online_search import online_greedy_search, weighted_online_greedy_search, online_random_greedy_search, simulated_annealing

CONFIG_PLANNING = {
    'widths': list(range(2, 9, 1)),
    'heights': list(range(2, 9, 1)),
    'n_golds': list(range(1, 9, 2)),
    'n_leps': [0],
    'gold_scores': [100],
    'stair_scores': [0],
    'time_penalties': [-1, -5, -10, -15],
    'max_steps': 1000,
    'n_episodes': 3,
    'algorithms': [a_star_search, weighted_a_star_search, uniform_cost_search, greedy_search],
    'alg_paramss': [
            [
                {'allowed_moves_function': ALLOWED_SIMPLE_MOVES},
                {'allowed_moves_function': ALLOWED_COMPOSITE_MOVES}
            ],
            [
                {'w': 0.5, 'allowed_moves_function': ALLOWED_SIMPLE_MOVES},
                {'w': 2, 'allowed_moves_function': ALLOWED_SIMPLE_MOVES},
                {'w': 0.5, 'allowed_moves_function': ALLOWED_COMPOSITE_MOVES},
                {'w': 2, 'allowed_moves_function': ALLOWED_COMPOSITE_MOVES}
            ],
            [
                {'allowed_moves_function': ALLOWED_SIMPLE_MOVES},
                {'allowed_moves_function': ALLOWED_COMPOSITE_MOVES}
            ],
            [
                {'allowed_moves_function': ALLOWED_SIMPLE_MOVES},
                {'allowed_moves_function': ALLOWED_COMPOSITE_MOVES}
            ]
        ]
}

CONFIG_ONLINE = {
    'widths': list(range(2, 9, 1)),
    'heights': list(range(2, 9, 1)),
    'n_golds': list(range(1, 9, 2)),
    'n_leps': list(range(1, 20, 5)),
    'gold_scores': [100],
    'stair_scores': [0],
    'time_penalties': [-1, -5, -10, -15],
    'max_steps': 1000,
    'n_episodes': 3,
    'algorithms': [online_greedy_search, weighted_online_greedy_search, online_random_greedy_search, simulated_annealing],
    'alg_paramss': [
            [{}],
            [{'w': 0.5}, {'w': 2}],
            [{'prob_rand_move': 0.5, 'decay': 0.5}, {'prob_rand_move': 0.3, 'decay': 0.5}, {'prob_rand_move': 0.3, 'decay': 0}],
            [{}]
        ]
}

CONFIG_ONLINE2 = {
    'widths': list(range(2, 9, 1)),
    'heights': list(range(2, 9, 1)),
    'n_golds': list(range(1, 9, 2)),
    'n_leps': list(range(0, 10, 2)),
    'gold_scores': [100],
    'stair_scores': [0],
    'time_penalties': [-1, -5, -10, -15],
    'max_steps': 1000,
    'n_episodes': 3,
    'algorithms': [online_greedy_search, weighted_online_greedy_search, online_random_greedy_search, simulated_annealing],
    'alg_paramss': [
            [{}],
            [{'w': 0.5}, {'w': 2}],
            [{'prob_rand_move': 0.5, 'decay': 0.5}, {'prob_rand_move': 0.3, 'decay': 0.5}, {'prob_rand_move': 0.3, 'decay': 0}],
            [{}]
        ]
}