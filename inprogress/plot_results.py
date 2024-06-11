from planning import a_star_search, weighted_a_star_search, uniform_cost_search, greedy_search
import matplotlib.pyplot as plt
from online_search import online_greedy_search, weighted_online_greedy_search, online_random_greedy_search, simulated_annealing
from utils import get_plot

def plot_w_all(plans):
    algorithms = [
        {'name': weighted_a_star_search.__name__, 'params': [['w', '0.5'], ['allowed_moves_function', 'simple_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '2'], ['allowed_moves_function', 'simple_moves']]},
        {'name': uniform_cost_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': greedy_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': a_star_search.__name__, 'params': [['allowed_moves_function', 'simple_moves']]}
    ]
    x_vars = ['width', 'width', 'width']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('n_golds', 5)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=3)
    plt.show()

def plot_tp_all(plans):
    algorithms = [
        {'name': weighted_a_star_search.__name__, 'params': [['w', '0.5'], ['allowed_moves_function', 'simple_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '2'], ['allowed_moves_function', 'simple_moves']]},
        {'name': uniform_cost_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': greedy_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': a_star_search.__name__, 'params': [['allowed_moves_function', 'simple_moves']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=3)
    plt.show()

def plot_w_as(plans):

    algorithms = [
        {'name': a_star_search.__name__, 'params': [['allowed_moves_function', 'simple_moves']]},
        {'name': a_star_search.__name__, 'params': [['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['width', 'width', 'width']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('n_golds', 5)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_tp_as(plans):

    algorithms = [
        {'name': a_star_search.__name__, 'params': [['allowed_moves_function', 'simple_moves']]},
        {'name': a_star_search.__name__, 'params': [['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_w_was(plans):

    algorithms = [
        {'name': weighted_a_star_search.__name__, 'params': [['w', '0.5'], ['allowed_moves_function', 'simple_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '0.5'], ['allowed_moves_function', 'composite_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '2'], ['allowed_moves_function', 'simple_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '2'], ['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['width', 'width', 'width']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('n_golds', 5)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_tp_was(plans):

    algorithms = [
        {'name': weighted_a_star_search.__name__, 'params': [['w', '0.5'], ['allowed_moves_function', 'simple_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '0.5'], ['allowed_moves_function', 'composite_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '2'], ['allowed_moves_function', 'simple_moves']]},
        {'name': weighted_a_star_search.__name__, 'params': [['w', '2'], ['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_w_uc(plans):

    algorithms = [
        {'name': uniform_cost_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': uniform_cost_search.__name__, 'params': [['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['width', 'width', 'width']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('n_golds', 5)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_tp_uc(plans):

    algorithms = [
        {'name': uniform_cost_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': uniform_cost_search.__name__, 'params': [['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_w_g(plans):

    algorithms = [
        {'name': greedy_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': greedy_search.__name__, 'params': [['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['width', 'width', 'width']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('n_golds', 5)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_tp_g(plans):

    algorithms = [
        {'name': greedy_search.__name__, 'params': [ ['allowed_moves_function', 'simple_moves']]},
        {'name': greedy_search.__name__, 'params': [['allowed_moves_function', 'composite_moves']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['score', 'path_len', 'expanded_nodes']
    fs = [(lambda x: x), (lambda x: x), (lambda x: x)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 2))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = plans,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=2)
    plt.show()

def plot_lep_all(episodes):
    algorithms = [
        {'name': online_greedy_search.__name__, 'params': []},
        {'name': weighted_online_greedy_search.__name__, 'params': [['w', '0.5']]},
        {'name': weighted_online_greedy_search.__name__, 'params': [['w', '2']]},
        {'name': simulated_annealing.__name__, 'params': []},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0.5']]},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0']]}
    ]
    x_vars = ['n_leps', 'n_leps', 'n_leps', 'n_leps']
    y_vars = ['rewards', 'steps', 'gold_gains', 'gold_thefts']
    fs = [(lambda x: sum(x)), (lambda x: x), (lambda x: len([item for item in x if item !=0])), (lambda x: len([item for item in x if item !=0]))]

    fig, axs = plt.subplots(1, 4, figsize=(15, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = episodes,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=6)
    plt.show()


def plot_lep_part(episodes):
    algorithms = [
        {'name': online_greedy_search.__name__, 'params': []},
        {'name': weighted_online_greedy_search.__name__, 'params': [['w', '0.5']]},
        {'name': simulated_annealing.__name__, 'params': []},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0.5']]},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0']]}
    ]
    x_vars = ['n_leps', 'n_leps', 'n_leps', 'n_leps']
    y_vars = ['rewards', 'steps', 'gold_gains', 'gold_thefts']
    fs = [(lambda x: sum(x)), (lambda x: x), (lambda x: len([item for item in x if item !=0])), (lambda x: len([item for item in x if item !=0]))]

    fig, axs = plt.subplots(1, 4, figsize=(15.3, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = episodes,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=6)
    plt.show()

def plot_tp_all2(episodes):
    algorithms = [
        {'name': online_greedy_search.__name__, 'params': []},
        {'name': weighted_online_greedy_search.__name__, 'params': [['w', '0.5']]},
        {'name': weighted_online_greedy_search.__name__, 'params': [['w', '2']]},
        {'name': simulated_annealing.__name__, 'params': []},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0.5']]},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['rewards', 'steps', 'gold_gains', 'gold_thefts']
    fs = [(lambda x: sum(x)), (lambda x: x), (lambda x: len([item for item in x if item !=0])), (lambda x: len([item for item in x if item !=0]))]

    fig, axs = plt.subplots(1, 4, figsize=(15.3, 3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = episodes,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=6)
    plt.show()


def plot_tp_part(episodes):
    algorithms = [
        {'name': online_greedy_search.__name__, 'params': []},
        {'name': weighted_online_greedy_search.__name__, 'params': [['w', '0.5']]},
        {'name': simulated_annealing.__name__, 'params': []},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0.5']]},
        {'name': online_random_greedy_search.__name__, 'params': [['prob_rand_move', '0.3'], ['decay', '0']]}
    ]
    x_vars = ['time_penalty', 'time_penalty', 'time_penalty', 'time_penalty']
    y_vars = ['rewards', 'steps', 'gold_gains', 'gold_thefts']
    fs = [(lambda x: sum(x)), (lambda x: x), (lambda x: len([item for item in x if item !=0])), (lambda x: len([item for item in x if item !=0]))]

    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3))

    i = 0
    for ax in axs:
        handles, labels = get_plot(
        episodes = episodes,
        ax = ax,
        fixed = [('width', 8)],
        algorithms = algorithms,
        x_variable = x_vars[i],
        y_variable = y_vars[i],
        f = fs[i]
        )
        i += 1

    fig.legend(handles, labels, loc='upper center', fontsize='8', ncol=6)
    plt.show()