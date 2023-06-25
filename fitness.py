import numpy as np

import dijkstarAlgorithm as dijkstaralgorithm
import dijkstarGraph as dijkstargraph


def multi_fitness(full_graph, route_set, data_demand, data_tt, total_demand, number_of_nodes):

    operator_fitness_value = operator_fitness(full_graph, route_set, data_tt)

    passenger_fitness_value = passenger_fitness(
        full_graph, route_set, data_demand, total_demand, number_of_nodes)

    return (passenger_fitness_value, operator_fitness_value)


def operator_fitness(full_graph, route_set, data_tt):
    "Sum all of travel times for a route set"
    tt = 0
    for route in route_set:
        tt_route = 0
        for j, node in enumerate(route):
            if j == len(route) - 1:
                break
            if route[j+1] == -1:
                break
            node = int(node)
            node_k = int(route[j+1])
            tt_between_nodes = data_tt[node, node_k]
            tt_route += tt_between_nodes
        tt += tt_route

    return tt


def passenger_fitness(full_graph, route_set, data_demand, total_demand, number_of_nodes):
    "03052021, this is the fitness function for non-symmetric demand and travel times matrix"
    "Passenger_fitness = ATT = Average travel time (at this moment)"
    "Borde testa att använda något nätverkspackage nu när det enda som spelar roll är hur man konstruerar nätverket!"

    sub_graph = dijkstargraph.Graph()

    "Use transfer penatly 5 for Mandl, and 7.5 for Södertälje"
    transfer_penalty = 5
    #transfer_penalty = 7.5

    prev_nodes = route_set[:, 0]
    node_tracker = dict()
    parallel_nodes = dict()
    node_id = number_of_nodes

    for i, route in enumerate(route_set):
        prev_node = prev_nodes[i]
        prev_inter_node = None

        for j, node in enumerate(route):
            "For the first node in the route, check if other  inter nodes for that node exists and connect with a penalty edge"
            if prev_inter_node == None:
                inter_node = node_id
                node_tracker[inter_node] = (node, i)
                if prev_node in parallel_nodes:
                    for i_n in parallel_nodes[prev_node]:
                        sub_graph.add_edge(
                            i_n, inter_node, (transfer_penalty, None))
                        sub_graph.add_edge(
                            inter_node, i_n,  (transfer_penalty, None))
                    parallel_nodes[node].add(inter_node)
                else:
                    parallel_nodes[prev_node] = {inter_node}

                prev_inter_node = inter_node
                node_id += 1

            if node == -1:
                break

            if node != prev_node:
                inter_node = node_id
                node_tracker[inter_node] = (node, i)

                # find the cost for the edge, using the full graph
                edge = (full_graph.get_edge(prev_node, node)[0], i)

                "This node has not been added to the graph yet so we add the id of its node"
                if node in parallel_nodes:
                    for parallel_node in parallel_nodes[node]:
                        sub_graph.add_edge(
                            inter_node, parallel_node, (transfer_penalty, None))
                        sub_graph.add_edge(
                            parallel_node, inter_node,  (transfer_penalty, None))
                    parallel_nodes[node].add(inter_node)

                else:
                    parallel_nodes[node] = {inter_node}

                "We add two edges (would have happened automatically if we used undirected graph"
                sub_graph.add_edge(prev_inter_node, inter_node, edge)
                sub_graph.add_edge(inter_node, prev_inter_node, edge)

                "This is why we have undirected, since this edge must be directed"
                node_id += 1
                prev_node = node
                prev_inter_node = inter_node

    shortest_time = dict()
    "Total cost route set is the sum of demand from node i to j multiplied with shortest time from node i to j, divided by the total demand for the network"

    total_cost_route_set = 0

    "Denna del går att förbättra enormt, exempelvis: Om vi kör med single-source-shortest-paths och kollar från i till alla andra noder. Då är det onödigt att ha en massa 'main' noder, eftersom "
    "Vi redan har gått igenom och hittat shortest paths för alla noder! Då räcker det att ta min(parallela noder[j])'s shortest paths'. Det enda vi behöver då är en startnod. Vi blir därför av med 14 noder ur nätverket"

    total_cost_route_set = 0
    for i in range(number_of_nodes):
        if i not in shortest_time:
            shortest_time[i] = {}
        "Lägg directed edges 'åt andra hållet' för startnoden till dess parallela noder istället för att behöva loopa igenom och köra shortest path på varje parallel nod"
        for parallel_node in parallel_nodes[i]:
            sub_graph.add_edge(i, parallel_node, (0, None))
        pred = dijkstaralgorithm.single_source_shortest_paths(
            sub_graph, i, cost_func=cost_func)

        "Går att förbättra detta, gör mycket onödigt arbete"
        for j in range(number_of_nodes):

            cost = np.inf
            for parallel_node in parallel_nodes[j]:
                cost_pn = dijkstaralgorithm.extract_shortest_path_from_predecessor_list(
                    pred, parallel_node).total_cost
                if cost_pn < cost:
                    cost = cost_pn

            total_cost_route_set += cost*data_demand[i][j]
            shortest_time[i][j] = cost

        for parallel_node in parallel_nodes[i]:
            sub_graph.remove_edge(i, parallel_node)

    total_cost_route_set = total_cost_route_set/(total_demand*2)
    return total_cost_route_set


def cost_func(u, v, edge, prev_edge):
    cost, route_id = edge
    return cost


def cost_func2(u, v, edge, prev_edge):
    "This is for shortest path only counting number of edges traversed, so each edge has a cost of 1"
    cost = 1
    return cost
