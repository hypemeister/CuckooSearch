from os import error
import numpy as np
import itertools
from numpy.core.numeric import full

import fitness as fit
import sortingAlgorithms as sort
import feasibility as feas


def local_search(route_set, route_set_index, population_container, problem_information_container,  data_demand, data_tt, total_demand, rng, neighborhood_size):
    "multi-objective local search, returns routeset, routeset fitness and a population container "

    neighborhood_size = 20

    population, population_fitness, population_fronts, population_front_sizes = population_container
    full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, route_set_size, number_of_nodes = problem_information_container
    route_set_fitness = population_fitness[route_set_index]

    "Have a return here for a quick but ugly way of not having a local search"
    # return population_fitness[route_set_index], route_set, population_container

    possible_combination_indices = list(itertools.product(
        range(route_set_size), range(max_route_size)))

    i = 0
    while i < neighborhood_size:

        rng_comb_index = rng.integers(len(possible_combination_indices))
        possible_combination_indices[rng_comb_index], possible_combination_indices[
            -1] = possible_combination_indices[-1], possible_combination_indices[rng_comb_index]
        comb_index = possible_combination_indices.pop()

        route_index = comb_index[0]
        node_index = comb_index[1]
        node = route_set[route_index][node_index]

        if node == -1:
            "Another way would be to use np.argwhere(route_set < 0) to get possible combinations. More elegant but not sure if it is more efficient on these kinds of routesets"
            pass
        else:
            next_node_index = node_index + 1
            prev_node_index = node_index - 1

            next_node_is_last = False
            if next_node_index == max_route_size:
                "Terminal last last node has been chosen"
                next_node_is_last = True

            elif route_set[route_index][next_node_index] == -1:
                "Terminal last node has been chosen (not the last possible)"
                next_node_is_last = True

            "Find possible_neighbors with this if statement"
            if node_index == 0 or next_node_is_last == True:
                if next_node_is_last == True:
                    next_node_index = prev_node_index

                "Terminal node has been chosen"
                next_node = route_set[route_index][next_node_index]

                "21/4 - 2022: Removed the constraint of no same node multiple times in a route. "
                "The previous code looked like this: "
                # Önskar att jag kände till detta trick tidigare i kodningen, snabbt enkelt sätt att unpacka ett dictionary till en lista
                """
                neighbors_next = np.array([*full_graph[next_node]])
               
                "Check which neighbors are not already in the route"
                mask = np.isin(element=neighbors_next,
                               test_elements=route_set[route_index], invert=True)
                

                "Possible neighbors - must also check terminal nodes!!"
                neighbors_next_not_in_route = set(neighbors_next[mask])
                """
                neighbors_next_not_in_route_list = np.array(
                    [*full_graph[next_node]])
                rng.shuffle(neighbors_next_not_in_route_list)
                neighbors_next_not_in_route = set(
                    neighbors_next_not_in_route_list)

                possible_neighbors = neighbors_next_not_in_route.intersection(
                    terminal_nodes)

            else:
                "Any other node in the route, inbetween two other nodes"
                next_node = route_set[route_index][next_node_index]
                prev_node = route_set[route_index][prev_node_index]
                "21/4 - 2022: Removed the constraint of no same node multiple times in a route. "
                "The new code added this rng shuffle: "
                neighbors_next = np.array([*full_graph[next_node]])
                rng.shuffle(neighbors_next)
                neighbors_prev = np.array([*full_graph[prev_node]])
                neighbors_tot = np.intersect1d(neighbors_next, neighbors_prev)

                """
                "Check which neighbors are not already in the route"
                mask = np.isin(element=neighbors_tot,
                               test_elements=route_set[route_index], invert=True)

                possible_neighbors = neighbors_tot[mask]
                """
                possible_neighbors = neighbors_tot

            for possible_neighbor in possible_neighbors:

                new_route_set = route_set.copy()
                new_route_set[route_index][node_index] = possible_neighbor
                feasible = feas.feasability_and_repair(
                    new_route_set, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=False)

                if feasible:
                    new_fitness = fit.multi_fitness(
                        full_graph, new_route_set, data_demand, data_tt, total_demand, number_of_nodes)

                    is_better, new_population_container, new_route_set_index = sort.is_better(route_set_fitness, new_route_set, new_fitness, population.copy(
                    ), population_fitness.copy(), population_fronts.copy(), population_front_sizes.copy(), route_set_index)
                    if is_better:

                        population_container = new_population_container
                        # print("old index", route_set_index, "new index", new_route_set_index, "We move up ", route_set_index-new_route_set_index, "places")
                        "Comment away the line of code below if you want only depth of one local search"
                        new_fitness, new_route_set, population_container = local_search(
                            new_route_set, new_route_set_index, population_container, problem_information_container, data_demand, data_tt, total_demand, rng, neighborhood_size)

                        return new_fitness, new_route_set, population_container

        i += 1
    return population_fitness[route_set_index], route_set, population_container


def perform_exchange_part_of_route_mode_1(route_set, good_route_set, rng, terminal_nodes, max_route_size, route_set_size, number_of_parts_to_exchange=1):
    "Wrapper for exchange_part_of_route_mode_1"
    "Performs the method 1 from K&B 2014"

    success = False
    possible_combinations = list(
        itertools.product(range(route_set_size), repeat=2))
    temp_route_set = route_set.copy()
    max_number_of_tries = len(possible_combinations)/2
    success_exchanges = 0
    i = 0
    while not success and i < max_number_of_tries and len(possible_combinations) > 0:
        rng_comb_index = rng.integers(len(possible_combinations))

        "Fast way to remove a random element from possible indices list, since pop() removes last item from list and is faster than removing from middle of list"
        "It will lead to an unordered list however"
        possible_combinations[rng_comb_index], possible_combinations[-1] = possible_combinations[-1], possible_combinations[rng_comb_index]
        comb_index = possible_combinations.pop()
        route_1_index = comb_index[0]
        route_2_index = comb_index[1]

        route_1 = route_set[route_1_index].copy()
        route_2 = good_route_set[route_2_index].copy()

        success_move, new_route_1 = exchange_part_of_route_move_1(
            route_1, route_2, terminal_nodes, max_route_size)

        if success_move == True:
            temp_route_set[route_1_index] = new_route_1
            success_exchanges += 1
        if success_exchanges == number_of_parts_to_exchange:
            success = True
            route_set = temp_route_set
        i += 1

    return success, route_set


def exchange_part_of_route_move_1(route_1, route_2, terminal_nodes,  max_route_size):
    "This move will get a part of a route_2 (from a good routeset) and try to place it in route 1 "

    "Could probably be improved by first finding matching values, then looping from there but the logic is quite complicated given how the numpy functions work"
    for i, node_i in enumerate(np.nditer(route_1)):
        if node_i == -1:
            break
        for j, node_j in enumerate(np.nditer(route_2)):
            if j + 1 == max_route_size:
                "Break if j is the last node"
                break

            if node_j == -1 or route_2[j+1] == -1:
                "Break if node j is -1, or if node j is the last node, since that would lead only to adding on nothing from route 2, to route 1"
                break

            if node_i == node_j:
                "21/4 - 2022: Removed the constraint of no same node multiple times in a route. "
                "The previous code looked like this: "
                """"
                if np.setxor1d(route_1[i:], route_2[j:]).size == False:
                    "This means that these two rest routes are the same"
                    break

                mask = np.isin(
                    element=route_1[0:i+1], test_elements=route_2[j+1:])
                if True not in mask:
                    "We have no cycles"
                """
                if True:

                    if i <= j:
                        "The size is not a problem - and we have a new route"
                        nmb_of_no_nodes = j-i
                        new_route_1 = np.concatenate(
                            (route_1[0:i], route_2[j:], [-1]*nmb_of_no_nodes))

                        return True, new_route_1
                    else:
                        "The route size is a problem, we go backwards in route 2 and delete nodes until it fits"
                        k = 0
                        nmb_of_no_nodes = 0
                        while True:
                            size = i + max_route_size - j - k
                            if size <= max_route_size:

                                if k >= max_route_size-j-2:
                                    "If k becomes large enough, so that there is no rest route left, we must break"
                                    break

                                new_last_node = route_2[-k+1]
                                if new_last_node == -1 or new_last_node in terminal_nodes:
                                    "We have a new route"
                                    new_route_1 = np.concatenate(
                                        (route_1[0:i], route_2[j:-k], [-1]*nmb_of_no_nodes))
                                    break

                                else:
                                    k += 1
                                    nmb_of_no_nodes += 1
                            else:
                                k += 1
                else:
                    pass
                    "Then we have a cycle so we could try the other part of route 2?, not implemented atm"
    return False, route_1


def method_2(route_set, good_route_set, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=1):
    "This is the method 2 from K&B 2014 - it tries to replace entire routes instead of route parts as in 'exchange_part_of_route_move_1'"

    possible_combinations = list(
        itertools.product(range(route_set_size), repeat=2))

    "The loop will try all possible combinations, i.e., the max_number_of_tries limit is not enforced"
    "Uncomment i+=1 and i-=1 if you want a  max number of tries. If computation-time performance is very importance for example"
    success = False
    temp_route_set = route_set.copy()
    max_number_of_tries = len(possible_combinations)/2
    success_exchanges = 0
    i = 0
    while not success and i < max_number_of_tries and len(possible_combinations) > 0:
        rng_comb_index = rng.integers(len(possible_combinations))
        possible_combinations[rng_comb_index], possible_combinations[-1] = possible_combinations[-1], possible_combinations[rng_comb_index]
        comb_index = possible_combinations.pop()
        route_1_index = comb_index[0]
        route_2_index = comb_index[1]

        route_2 = good_route_set[route_2_index].copy()
        temp_temp_route_set = temp_route_set.copy()

        temp_temp_route_set[route_1_index] = route_2
        feasible = feas.feasability_and_repair(temp_temp_route_set, route_set_size, full_graph,
                                               full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
        if feasible:
            temp_route_set = temp_temp_route_set
            success_exchanges += 1
            # i -= 1
        # i += 1
        if success_exchanges == number_of_routes_to_exchange:
            return True, temp_route_set
    return False, route_set
