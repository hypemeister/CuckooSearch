import pandas as pd
import numpy as np


def create_initial_population(data_tt, data_demand, n, adj_dict, max_route_size, route_set_size, number_of_nodes):
    "Create the initial population by using the procedure of (Kechagiopolous and Beligiannis 2014) to create the route sets"
    "My report (Ekelöf, 2021) contains better information on how it works"
    "03052021 change in how to calculate activity level, i.e., before we only summed over the node's row, now we also add its column, since that is its incoming demand"

    rng = np.random.default_rng(123)
    R = route_set_size

    data_demand = pd.DataFrame(data_demand)

    pop_lista = []
    for i in range(n):
        lista = np.zeros(shape=(R, max_route_size))
        pop_lista.append(lista)
    population = np.asarray(pop_lista)

    activity_level = np.zeros(shape=(number_of_nodes, 2))
    for index, row in data_demand.iterrows():
        activity_level[index, 0] = index
        col_sum = data_demand.iloc[:, index].sum()
        activity_level[index, 1] = row.sum() + col_sum

    #print("Activity level ", pd.DataFrame(activity_level))
    for i in range(n):
        route_set = create_initial_routes(
            data_tt, data_demand, activity_level, rng, adj_dict, R, max_route_size, number_of_nodes)
        population[i] = route_set

    data_demand = data_demand.to_numpy()
    return population


def create_initial_routes(data_tt, data_demand, activity_level, rng, adj_dict, R, max_route_size, number_of_nodes):
    "Creates and returns a route set"

    activity_level_sorted = activity_level[(-activity_level[:, 1]).argsort()]

    "Select first K nodes"
    "K = 14 if Mandl's network, this idea is from K&B 2014, since node 15 has zero activity or demand."
    "Should probably be modified for the Södertälje network"
    K = number_of_nodes-1
    initial_node_set = activity_level_sorted[0:K, :]

    first_nodes = np.zeros(R)
    for i in range(R):
        prob_vector = initial_node_set[:, 1] / np.sum(initial_node_set[:, 1])
        rng_index = np.flatnonzero(rng.multinomial(1, prob_vector))[0]

        first_nodes[i] = initial_node_set[rng_index][0]
        initial_node_set = np.delete(initial_node_set, rng_index, 0)

    route_set_numpy = -np.ones(shape=(R, max_route_size), dtype=int)
    route_set_numpy[:, 0] = first_nodes.T
    route_set = pd.DataFrame(route_set_numpy)

    d_bias = np.ones(number_of_nodes)
    for i in range(number_of_nodes):
        "If node i only has one neighbor, set its bias to 0.5 instead of 1"
        if adj_dict[i].size == 1:
            d_bias[i] = 0.5

    for i in range(R):
        prev_node = route_set.iloc[i].to_numpy()[0]
        add_nodes_to_route(prev_node, route_set.iloc[i].to_numpy(
        ), d_bias, activity_level, adj_dict, rng, max_route_size)

        # We dont need the following: route_set.iloc[i] = route since pandas arrays apparently dont need it, it happens inplace somehow
    return route_set.to_numpy()


def add_nodes_to_route(prev_node, route, d_bias, activity_level, adj_dict, rng, max_route_size):
    "Add nodes to a route and returns it when no more nodes can be added"

    adj_nodes = adj_dict[prev_node]

    mask = np.isin(adj_nodes, route, invert=True)
    # Vicinity node set = nodes in adjency and not in route
    vns = adj_nodes[mask]
    if vns.size == 0:
        return route

    prob_vector = np.zeros(vns.size)
    sum_dk_ak = 0
    for i, vns_node in enumerate(vns):
        vns_node = int(vns_node)
        dk_ak = d_bias[vns_node] * activity_level[vns_node][1]
        prob_vector[i] = dk_ak
        sum_dk_ak += dk_ak

    if vns.size == 1:
        new_node = int(vns[0])
    else:
        prob_vector = np.divide(prob_vector, sum_dk_ak)
        rng_index = np.flatnonzero(rng.multinomial(1, prob_vector))[0]
        new_node = int(vns[rng_index])

    with np.nditer(route, op_flags=['readwrite']) as it:
        for i, node in enumerate(it):
            if node == -1:
                route[i] = new_node
                prev_node = new_node
                d_bias[new_node] = d_bias[new_node] * 0.1
                break
    if i == max_route_size-1:
        return route
    else:
        route = add_nodes_to_route(
            prev_node, route, d_bias, activity_level, adj_dict, rng, max_route_size)
