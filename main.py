from os import error
import numpy as np
from numpy.core.numeric import full
import pandas as pd
import json
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
import cProfile
import pstats

from heapq import heappush, heappop
from dijkstarAlgorithm import find_path, single_source_shortest_paths, extract_shortest_path_from_predecessor_list
from dijkstarGraph import Graph


import readData as readdata
import createInitialPop as createpop
import createInitialPopulationSvensson as createpop2
import feasibility as feas
import perturbations as perturb
import sortingAlgorithms as sort
import fitness as fit
#from parameterClasses import NetworkParameterClass, DataClass, CuckooParameterClass


def create_graph(data_tt, number_of_nodes):
    "03052021 this is the create graph function for non-symmetric demand and travel times matrices"
    "The full_graph_shortest_paths dictionary contains all of the shortest paths for the full graph, with respect to edges traversed (cost for each edge is counted as 1)"
    "On the form: {node i: {node j: (edges traversed, path), node k: (edges traversed, path),...} ...}"

    "I hade a similar function before that only could create symmetric graphs, it should save some computation time but insignificant since it is only run once"

    graph = Graph()
    for i, row in enumerate(data_tt):
        for j, link in enumerate(row):
            if j != i:
                if link != np.inf:
                    edge = graph.add_edge(i, j, edge=(link, 0))
    #print("\n Graph \n", json.dumps(graph.get_data(), indent=4, default=str))

    full_graph_shortest_paths = dict()
    for i in range(number_of_nodes):
        full_graph_shortest_paths[i] = dict()

        pred = single_source_shortest_paths(graph, i, cost_func=fit.cost_func2)

        for j in range(number_of_nodes):
            temp_info = extract_shortest_path_from_predecessor_list(pred, j)
            path = temp_info.nodes
            cost = temp_info.total_cost

            full_graph_shortest_paths[i][j] = (cost, path)

    #print(json.dumps(full_graph_shortest_paths, indent=3, default=str))
    return graph, full_graph_shortest_paths


def levy(mu, rng):
    "A approximation of Levy flights distribution. Returns a value between 0 and 1"
    "The following code is copied from (Aziz Ouaarab, Discrete Cuckoo Search for Combinatorial Optimization,2020) and 'translated' from java into python"
    t = rng.random() * 9 + 1
    t = np.power((1/t), mu)
    return t


def smartCS(population, population_fitness, n, mu, p_c, data_demand, total_demand, feasability_container, k_vector, l_vector):
    "This is the single objective version of the smart CS function"
    "Search with a smart portion of the population"
    "Do not include the best one (Notice that indexing starts at 1)"
    "Now we only compare with best solution, can be changed to a random solution in the smart CS portion of the population"

    route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng = feasability_container

    best_sol = population[0].copy()
    i = 1

    number_of_intervals = len(k_vector) + len(l_vector) + 1
    bins = np.linspace(0, 1, number_of_intervals)

    while i < n*p_c:

        best_sol = population[0].copy()

        success_and_feasible = False

        smart_sol = best_sol
        sol = population[i].copy()

        levy_value = levy(mu, rng)
        interval_number = np.digitize(levy_value, bins)

        if interval_number <= len(k_vector):
            k = k_vector[interval_number-1]

            "Do one perform exchange part of route mode 1 calls"
            success, new_sol = perturb.perform_exchange_part_of_route_mode_1(
                sol, smart_sol, rng, terminal_nodes, max_route_size, route_set_size, number_of_parts_to_exchange=k)
            if success:
                feasible = feas.feasability_and_repair(
                    new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
                if feasible:
                    success_and_feasible = True
                    new_fitness = fit.passenger_fitness(
                        full_graph, new_sol, data_demand, total_demand, number_of_nodes)

        else:
            l = l_vector[interval_number - len(k_vector) - 1]
            success, new_sol = perturb.method_2(sol, smart_sol, route_set_size, full_graph, full_graph_shortest_paths,
                                                terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=l)

            if success:
                feasible = feas.feasability_and_repair(
                    new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=True)
                if feasible:
                    success_and_feasible = True
                    new_fitness = fit.passenger_fitness(
                        full_graph, new_sol, data_demand, total_demand, number_of_nodes)

        if success_and_feasible:
            if population_fitness[i] > new_fitness:
                #print("\n Smart Cuckoo number,", i," was improved to ", new_fitness, "from ", population_fitness[i],"\n")
                population[i] = new_sol
                population_fitness[i] = new_fitness
        i += 1

    population, population_fitness = sort.sort_population(
        population, population_fitness)
    return population, population_fitness


def worstCS(population, population_fitness, n, p_a, data_demand, total_demand, feasability_container, number_of_big_pert):
    "This is the single objective version of the worst CS function"

    route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng = feasability_container

    start_index_p_a = int((n-n*p_a))
    "Does method 2 from best_sol and sol i every time, probably better to use a random top 10% sol instead of best sol"
    best_sol = population[0].copy()
    for i in range(start_index_p_a, n):
        success_and_feasible = False

        "Picks sol i from the population"
        sol = population[i].copy()

        smart_sol = best_sol

        success, new_sol = perturb.method_2(smart_sol, sol, route_set_size, full_graph, full_graph_shortest_paths,
                                            terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=number_of_big_pert)

        if success:
            #print("\n Success  for  worse solution..")
            feasible = feas.feasability_and_repair(
                new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=True)
            if feasible:
                #print("\n Success and feasible  for  worse solution..")
                success_and_feasible = True
                new_fitness = fit.passenger_fitness(
                    full_graph, new_sol, data_demand, total_demand, number_of_nodes)

        if success_and_feasible:
            #print(i,"New", new_fitness, "old", population_fitness[i])
            if population_fitness[i] > new_fitness:
                print(
                    "\n Success and feasible and better fitness for  worse solution..\n")
                population[i] = new_sol
                population_fitness[i] = new_fitness

    population, population_fitness = sort.sort_population(
        population, population_fitness)
    return population, population_fitness


def getCukoo(population, population_fitness, mu, data_demand, total_demand, feasability_container, k_vector, l_vector):
    "This is the single objective version of the get cuckoo function"

    route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng = feasability_container

    "Uses the best solution to mix with the random solution k"
    best_sol = population[0].copy()

    "Picks a random solution 'sol_k' from the population"
    sol_k_index = rng.integers(1, len(population))
    sol_k = population[sol_k_index].copy()

    number_of_intervals = len(k_vector) + len(l_vector) + 1
    bins = np.linspace(0, 1, number_of_intervals)

    levy_value = levy(mu, rng)
    interval_number = np.digitize(levy_value, bins)

    success_and_feasible = False

    if interval_number <= len(k_vector):
        k = k_vector[interval_number-1]

        "Do one perform exchange part of route mode 1 call"
        success, new_sol = perturb.perform_exchange_part_of_route_mode_1(
            best_sol, sol_k, rng, terminal_nodes, max_route_size, route_set_size, number_of_parts_to_exchange=k)
        if success:
            feasible = feas.feasability_and_repair(
                new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
            if feasible:
                success_and_feasible = True
                new_fitness = fit.passenger_fitness(
                    full_graph, new_sol, data_demand, total_demand, number_of_nodes)

    else:
        l = l_vector[interval_number - len(k_vector) - 1]
        success, new_sol = perturb.method_2(best_sol, sol_k, route_set_size, full_graph, full_graph_shortest_paths,
                                            terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=l)

        if success:
            feasible = feas.feasability_and_repair(
                new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
            if feasible:
                success_and_feasible = True
                new_fitness = fit.passenger_fitness(
                    full_graph, new_sol, data_demand, total_demand, number_of_nodes)

    if success_and_feasible:
        if population_fitness[sol_k_index] > new_fitness:
            print("BOOM! Get Cuckoo got a better solution with new fitness",
                  new_fitness, "compared to old fitness", population_fitness[sol_k_index])
            population[sol_k_index] = new_sol
            population_fitness[sol_k_index] = new_fitness

            population, population_fitness = sort.sort_population(
                population, population_fitness)
    return population, population_fitness


def multi_smartCS(population, population_fitness, population_fronts, population_front_sizes, n, mu, p_c, rng, route_set_size, data_tt, data_demand, total_demand, feasability_container, k_vector, l_vector):
    "This is the multi-objective version of the smart CS function"
    "Search with a smart portion of the population"
    "It searches from i = 0, i.e., it includes the solutions of the current pareto front"

    route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng = feasability_container
    best_front_size = population_front_sizes[0][0]

    i = 0
    number_of_intervals = len(k_vector) + len(l_vector) + 1
    bins = np.linspace(0, 1, number_of_intervals)

    while i < n*p_c:
        "Picks a 'best_sol' from the current pareto front to compare with"
        best_sol_index = rng.integers(best_front_size)
        best_sol = population[best_sol_index].copy()
        smart_sol = best_sol

        success_and_feasible = False

        "Picks sol i from the population"
        sol = population[i].copy()

        levy_value = levy(mu, rng)
        interval_number = np.digitize(levy_value, bins)

        if interval_number <= len(k_vector):
            k = k_vector[interval_number - 1]
            success, new_sol = perturb.perform_exchange_part_of_route_mode_1(
                sol, smart_sol, rng, terminal_nodes, max_route_size, route_set_size, number_of_parts_to_exchange=k)

            if success:
                feasible = feas.feasability_and_repair(
                    new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
                if feasible:
                    success_and_feasible = True
                    new_fitness = fit.multi_fitness(
                        full_graph, new_sol, data_demand, data_tt, total_demand, number_of_nodes)

        else:
            l = l_vector[interval_number - len(k_vector) - 1]
            success, new_sol = perturb.method_2(sol, smart_sol, route_set_size, full_graph, full_graph_shortest_paths,
                                                terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=l)

            if success:
                feasible = feas.feasability_and_repair(
                    new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=True)
                if feasible:
                    success_and_feasible = True
                    new_fitness = fit.multi_fitness(
                        full_graph, new_sol, data_demand, data_tt, total_demand, number_of_nodes)

        if success_and_feasible:

            "Puts the new sol and its fitness in copies of the population and population_fitness"
            "These copies are used in the local search function"
            new_population = population.copy()
            new_population_fitness = population_fitness.copy()
            new_population[i] = new_sol
            new_population_fitness[i] = new_fitness

            population_container = (new_population, new_population_fitness,
                                    population_fronts.copy(), population_front_sizes.copy())
            problem_information_container = (
                full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, route_set_size, number_of_nodes)
            new_fitness, new_sol,  new_population_container = perturb.local_search(
                new_sol, i,  population_container, problem_information_container, data_demand, data_tt, total_demand, rng, neighborhood_size=20)
            is_better, new_population_container, new_route_set_index = sort.is_better(
                population_fitness[i], new_sol, new_fitness, population, population_fitness, population_fronts, population_front_sizes, i)

            if is_better:
                #print("\n Smart Cuckoo number,", i," was improved to ", new_fitness, "from ", population_fitness[i],"\n")
                population, population_fitness, population_fronts, population_front_sizes = new_population_container

        i += 1
    return population, population_fitness


def multi_getCukoo(population, population_fitness, population_fronts, population_front_sizes, n, mu, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, number_of_nodes, max_route_size, rng, k_vector, l_vector, data_demand, data_tt, total_demand):
    "This is the multi-objective version of get cuckoo function"

    "Picks a 'best_sol' from the current pareto front to compare with"
    best_front_size = population_front_sizes[0][0]
    best_sol_index = rng.integers(best_front_size)
    best_sol = population[best_sol_index].copy()

    "Picks a random solution 'sol_k' from the population"
    sol_k_index = rng.integers(1, len(population))
    sol_k = population[sol_k_index].copy()

    number_of_intervals = len(k_vector)+len(l_vector) + 1
    bins = np.linspace(0, 1, number_of_intervals)

    levy_value = levy(mu, rng)
    interval_number = np.digitize(levy_value, bins)

    success_and_feasible = False

    if interval_number <= len(k_vector):
        k = k_vector[interval_number-1]
        success, new_sol = perturb.perform_exchange_part_of_route_mode_1(
            best_sol, sol_k, rng, terminal_nodes, max_route_size, route_set_size, number_of_parts_to_exchange=k)

        if success:
            feasible = feas.feasability_and_repair(
                new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
            if feasible:
                success_and_feasible = True
                new_fitness = fit.multi_fitness(
                    full_graph, new_sol, data_demand, data_tt, total_demand, number_of_nodes)

    else:
        l = l_vector[interval_number - len(k_vector) - 1]
        success, new_sol = perturb.method_2(best_sol, sol_k, route_set_size, full_graph, full_graph_shortest_paths,
                                            terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=l)

        if success:
            feasible = feas.feasability_and_repair(
                new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, True)
            if feasible:
                success_and_feasible = True
                new_fitness = fit.multi_fitness(
                    full_graph, new_sol, data_demand, data_tt, total_demand, number_of_nodes)

    if success_and_feasible:

        "Puts the new sol and its fitness in copies of the population and population_fitness"
        "These copies are used in the local search function"
        new_population = population.copy()
        new_population_fitness = population_fitness.copy()
        new_population[sol_k_index] = new_sol
        new_population_fitness[sol_k_index] = new_fitness

        population_container = (new_population, new_population_fitness,
                                population_fronts.copy(), population_front_sizes.copy())
        problem_information_container = (
            full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, route_set_size, number_of_nodes)
        new_fitness, new_sol,  new_population_container = perturb.local_search(
            new_sol, sol_k_index,  population_container, problem_information_container, data_demand, data_tt, total_demand, rng, neighborhood_size=20)
        is_better, new_population_container, new_route_set_index = sort.is_better(
            population_fitness[sol_k_index], new_sol, new_fitness, population, population_fitness, population_fronts, population_front_sizes, sol_k_index)

        if is_better:
            #print("BOOM! Get Cuckoo got a better solution with new fitness", new_fitness, "compared to old fitness", population_fitness[sol_k_index])
            population, population_fitness, population_fronts, population_front_sizes = new_population_container

    return population, population_fitness


def multi_worstCS(population, population_fitness, population_fronts, population_front_sizes, n, p_a, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, data_demand, data_tt, total_demand, number_of_big_pert):
    "This is the multi-objective version of the worst CS function"

    best_front_size = population_front_sizes[0][0]
    start_index_p_a = int((n-n*p_a))

    for i in range(start_index_p_a, n):
        success_and_feasible = False

        "Picks a 'best_sol' from the current pareto front to compare with"
        best_sol_index = rng.integers(best_front_size)
        best_sol = population[best_sol_index].copy()
        smart_sol = best_sol

        "Picks sol i from the population"
        sol = population[i].copy()

        success, new_sol = perturb.method_2(smart_sol, sol, route_set_size, full_graph, full_graph_shortest_paths,
                                            terminal_nodes, max_route_size, number_of_nodes, rng, number_of_routes_to_exchange=number_of_big_pert)

        if success:
            feasible = feas.feasability_and_repair(
                new_sol, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=True)
            if feasible:
                success_and_feasible = True
                new_fitness = fit.multi_fitness(
                    full_graph, new_sol, data_demand, data_tt, total_demand, number_of_nodes)

        if success_and_feasible:

            "Puts the new sol and its fitness in copies of the population and population_fitness"
            "These copies are used in the local search function"
            new_population = population.copy()
            new_population_fitness = population_fitness.copy()
            new_population[i] = new_sol
            new_population_fitness[i] = new_fitness

            population_container = (new_population, new_population_fitness,
                                    population_fronts.copy(), population_front_sizes.copy())
            problem_information_container = (
                full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, route_set_size, number_of_nodes)
            new_fitness, new_sol,  new_population_container = perturb.local_search(
                new_sol, i,  population_container, problem_information_container, data_demand, data_tt, total_demand, rng, neighborhood_size=20)
            is_better, new_population_container, new_route_set_index = sort.is_better(
                population_fitness[i], new_sol, new_fitness, population, population_fitness, population_fronts, population_front_sizes, i)

            if is_better:
                #print("\n Success and feasible and better fitness for  worse solution..\n")
                population, population_fitness, population_fronts, population_front_sizes = new_population_container

    return population, population_fitness


def main():
    start_time = timer()
    choice = "M"
    multi_objective = True

    if choice == "M":
        "Parameters for Mandl problem"
        max_route_size = 8
        #max_route_size = 6
        #route_set_size = 6
        route_set_size = 8
        number_of_nodes = 15
        terminal_nodes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

    elif choice == "S":
        "Parameters for Södertälje problem"
        max_route_size = 12
        #route_set_size = 18
        route_set_size = 20
        number_of_nodes = 58

        "från philips matlab fil terminal nodes"
        #terminalNodes = {3,5,9,11,13,16,18,20,21,22,24,26,27,30,34,37,40,43,44,47,52,53,54,57,59,60,61,65}
        "Tar bort 2,6,7,25,31,60,63 - vilket leder till förskjutning av noder -->"
        " 0:<2 -1:<6,-2:<7, -3:<25, -4:<31, -5:<60, -6:<63, -7:<      ,60 tas bort"
        #terminalNodes2 = {2,4,6,11-3,13-3,16-3,18-3,20-3,21-3,22-3,24-3,26-4,27-4,30-4,34-5,37-5,40-5,43-5,44-5, 47-5,52-5,53-5,54-5,57-5,59-5,61-6,65-7}
        #terminalNodes3 = {2,4,6,8,10,13,15,17,18,19,21,22,23,26,29,32,35,38,39, 42,47,48,49,52,54,55,58}
        "Minus 1 för index with python - These have been confirmed to be correct with Philip's matlab code"
        terminal_nodes = {1, 3, 5, 7, 9, 12, 14, 16, 17, 18, 20, 21,
                          22, 25, 28, 31, 34, 37, 38, 41, 46, 47, 48, 51, 53, 54, 57}
    print("Terminal nodes:", terminal_nodes)

    data_tt, data_demand = readdata.read_textfile(choice)
    adj_dict = readdata.create_adj_dictionary(data_tt)
    full_graph, full_graph_shortest_paths = create_graph(
        data_tt, number_of_nodes)
    total_demand = np.sum(np.sum(data_demand))/2

    "-----------------------Parameters for cuckoo search------------------------------"
    "Time penalty for a transfer can be found in 'fitness.py' under passenger fitness"
    "OBS! Don't forget to change the time penalty for the Södertälje network!!"
    "FITNESS = 5 atm"
    "local search depth = recursive atm"

    n = 10
    print("Population size ", n)
    feasability_buffer = 20
    max_generation = 50
    #max_generation = 3000
    #max_generation = 1500
    p_a = 0.5
    p_c = 0.6
    mu = 1

    # default seed used has been 123, others often used are 421, 536, 879
    rng_seed = 536
    rng = np.random.default_rng(rng_seed)

    solution_save_path = f"Results/Mandl/K8/no local/solutions{rng_seed}.csv"
    obj_save_path = f"Results/Mandl/K8/no local/objvalues{rng_seed}.csv"
    save_results = False

    if choice == 'M':
        k_vector = [1, 2]
        l_vector = [1, 4]
        number_of_big_pert = 4

    elif choice == 'S':
        k_vector = [1, 3]
        l_vector = [1, 3, 6, 10]
        number_of_big_pert = 10

    "----------------------------------------------------------------------------------------"

    "Create initial population"
    if choice == 'M':
        "K&B's initialization"
        initial_population = createpop.create_initial_population(
            data_tt, data_demand, n+feasability_buffer, adj_dict, max_route_size, route_set_size, number_of_nodes)
    elif choice == 'S':
        "Svensson's initialization"
        initial_population = createpop2.create_initial_population_2(
            data_tt, data_demand, n+feasability_buffer, full_graph, max_route_size, route_set_size, number_of_nodes, terminal_nodes, adj_dict)

    "Create necessary numpy arrays that are empty"
    if multi_objective:
        population_fitness = np.empty(shape=(n, 2))
    else:
        population_fitness = np.empty(n)

    population_fronts = np.empty(shape=(n, 2))
    pop_lista = []
    for i in range(n):
        lista = np.zeros(shape=(route_set_size, max_route_size))
        pop_lista.append(lista)
    population = np.asarray(pop_lista)

    print("\nInitial population\n", initial_population, "\n")
    "Add the route sets of the initial population that are feasible (repair if possible)"
    i = 0
    j = 0
    for route_set in initial_population:
        feasible = feas.feasability_and_repair(
            route_set, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=True)

        if feasible:
            if multi_objective:
                population_fitness[i] = fit.multi_fitness(
                    full_graph, route_set, data_demand, data_tt, total_demand, number_of_nodes)
            else:
                population_fitness[i] = fit.passenger_fitness(
                    full_graph, route_set, data_demand, total_demand, number_of_nodes)
            population[i] = route_set
            i += 1

        j += 1
        if i == n:
            break

    "Sort the population"
    if multi_objective:
        print("Time for multi sort")
        population, population_fitness, population_fronts, population_front_sizes, sorted_indices = sort.multi_sort_population(
            population, population_fitness)
    else:
        population, population_fitness = sort.sort_population(
            population, population_fitness)

    "Create a container of constants"
    feasability_container = (route_set_size, full_graph, full_graph_shortest_paths,
                             terminal_nodes, max_route_size, number_of_nodes, rng)

    if multi_objective:
        best_front_each_generation = []
        best_sol_each_generation = []
        final_gen = False
        t = 0
        "Run the main loop for max_generation number of times"
        while t < max_generation:
            print("Beginning generation ", t)

            population, population_fitness = multi_smartCS(population, population_fitness, population_fronts, population_front_sizes,
                                                           n, mu, p_c, rng, route_set_size, data_tt, data_demand, total_demand, feasability_container, k_vector, l_vector)

            population, population_fitness = multi_getCukoo(population, population_fitness, population_fronts, population_front_sizes, n, mu, route_set_size,
                                                            full_graph, full_graph_shortest_paths, terminal_nodes, number_of_nodes, max_route_size, rng, k_vector, l_vector, data_demand, data_tt, total_demand)

            population, population_fitness = multi_worstCS(population, population_fitness, population_fronts, population_front_sizes, n, p_a, route_set_size, full_graph,
                                                           full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, data_demand, data_tt, total_demand, number_of_big_pert=number_of_big_pert)

            print("Best solution of generation",
                  t, "is", population_fitness[0])
            print("sol", population[0],"\n", population[1], "\n")
            if t == max_generation - 1:
                final_gen = True

            for i, sol_fitness in enumerate(population_fitness[0:population_front_sizes[0][0]]):
                best_front_each_generation.append(
                    {
                        'Generation': t,
                        'Passenger objective': sol_fitness[0],
                        'Operator objective': sol_fitness[1],
                        'Final generation': final_gen
                    }
                )
                best_sol_each_generation.append(
                    {
                        'Generation': t,
                        'Route set': population[i]
                    }
                )
            t += 1
        print("THE END, population fitness:\n", population_fitness)
        print("\n Population fronts \n", population_fronts)
    else:
        t = 0
        while t < max_generation:

            population, population_fitness = smartCS(
                population, population_fitness, n, mu, p_c, data_demand, total_demand, feasability_container, k_vector, l_vector)

            population, population_fitness = getCukoo(
                population, population_fitness, mu, data_demand, total_demand, feasability_container, k_vector, l_vector)

            population, population_fitness = worstCS(
                population, population_fitness, n, p_a, data_demand, total_demand, feasability_container, number_of_big_pert)

            print("Best solution of generation",
                  t, "is", population_fitness[0])
            t += 1

    "----------------------Save-------------------"
    "No save for single-objective implemented"
    if multi_objective:
        df = pd.DataFrame(best_front_each_generation)
        df_sols = pd.DataFrame(best_sol_each_generation)
        if save_results == True:
            df_sols.to_csv(solution_save_path, index=False)
            df.to_csv(obj_save_path, index=False)

        end_time = timer()
        print("Elapsed time of the algorithm:",
              timedelta(seconds=end_time-start_time))


if __name__ == '__main__':
    #cProfile.run('main()', 'restats')
    #p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats('cumulative').print_stats(20)
    main()
