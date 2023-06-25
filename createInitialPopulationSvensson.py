import numpy as np


def create_initial_population_2(data_tt, data_demand, n, full_graph, max_route_size, route_set_size, number_of_nodes, terminal_nodes, adj_dict):
    "Creates an initial population based on (Svensson, 2019)'s initial population procedure."
    "He in turn based his procedure on Soares et al 2019, which is based on Kilic and Gök, 2014"

    rng = np.random.default_rng(123)
    terminal_nodes_list = sorted(terminal_nodes)
    number_of_routes_in_pallette = 1000
    route_pallette = create_route_pallette(number_of_routes_in_pallette, data_tt, data_demand, full_graph,
                                           max_route_size, route_set_size, number_of_nodes, terminal_nodes, terminal_nodes_list, adj_dict, rng)

    "This loop creates the initial population"
    population_list = []
    for i, route in enumerate(route_pallette):

        added_nodes = []
        route_set = -1 * np.ones(shape=(route_set_size, max_route_size))
        route_set[0] = route

        "Find out how many nodes each route has and add to a list"
        route_without_minus_ones = route[np.where(route >= 0)]
        added_nodes = route_without_minus_ones.tolist()

        added_routes = {i}

        "This loop creates route sets"
        number_of_added_routes = 1
        while number_of_added_routes < route_set_size:
            route_set_without_minus_ones = route_set[np.where(route_set >= 0)]

            max_number_of_new_nodes = 0
            max_routes_list = []
            for j, other_route in enumerate(route_pallette):
                if j not in added_routes:

                    "See how many new nodes are added by the other_route (that are not in the route set)"
                    other_route_without_minus_ones = other_route[np.where(
                        other_route >= 0)]
                    matching_nodes_inverted = np.isin(
                        other_route_without_minus_ones, route_set_without_minus_ones, invert=True)
                    number_of_new_nodes = len(
                        np.flatnonzero(matching_nodes_inverted))
                    number_of_matching_nodes = len(
                        matching_nodes_inverted) - number_of_new_nodes

                    "Make sure that the new route shares at least one node with another route to ensure connectivity"
                    if number_of_matching_nodes >= 1:
                        if number_of_new_nodes > max_number_of_new_nodes:
                            max_number_of_new_nodes = number_of_new_nodes
                            max_routes_list = [j]
                        elif number_of_new_nodes == max_number_of_new_nodes:
                            max_routes_list.append(j)

            "Pick one of the routes that have the most new number of nodes added, and add it to the route set"
            route_to_add_index = max_routes_list[rng.integers(
                len(max_routes_list))]
            route_to_add = route_pallette[route_to_add_index]
            route_set[number_of_added_routes] = route_to_add
            route_mask = np.where(route_to_add >= 0)
            route_nodes = np.isin(
                route_to_add[route_mask], added_nodes, invert=True)

            for node in route_to_add[route_mask][route_nodes]:
                added_nodes.append(node)
            added_routes.add(route_to_add_index)
            number_of_added_routes += 1

        population_list.append(route_set)

        if i == n-1:
            break

    return np.array(population_list)


def create_route_pallette(number_of_routes_in_pallette, data_tt, data_demand, full_graph, max_route_size, route_set_size, number_of_nodes, terminal_nodes, terminal_nodes_list, adj_dict, rng):
    "We multiply by 10000 here so that in the end of the create route pallette function, we don't have have a usage matrix with values too close to zero"
    "Since everything is multiplied by 10000 and divided by bias factor, it should not change the intended outcome"
    "If we don't do this, it will affect the Mandl problem by quite a lot!"

    usage_matrix = 10000*np.divide(data_demand, data_tt, out=-1/10000*np.ones(
        shape=data_demand.shape), where=((data_tt != 0) & (data_tt != np.inf)))
    #usage_matrix = np.divide(data_demand, data_tt, out=-1*np.ones(shape=data_demand.shape), where=((data_tt!=0) & (data_tt!=np.inf)))

    bias = 1.1
    route_size_lower_bound = 2

    "Idea: Add manually some usage to node 8 from node 14, since it won't get picked otherwise? if you want to use for Mandl's network"

    route_pallette = np.empty(
        shape=(number_of_routes_in_pallette, max_route_size))
    j = 0
    for i in range(10*number_of_routes_in_pallette):

        if j == number_of_routes_in_pallette:
            break

        route_size = rng.integers(route_size_lower_bound, max_route_size+1)
        success, route = create_route(usage_matrix, route_size, max_route_size,
                                      full_graph, bias, terminal_nodes, terminal_nodes_list, adj_dict, rng)
        if success:
            route_pallette[j] = route

            j += 1

    return route_pallette


def create_route(usage_matrix, route_size, max_route_size, full_graph, bias, terminal_nodes, terminal_nodes_list, adj_dict, rng):
    success = False
    np_route = -1 * np.ones(max_route_size)

    "Special case if route_size is 2 or 3"
    if route_size == 2:
        "If route size = 2 we can only have two terminal nodes in the route!"
        "Find the link between two terminal nodes with highest usage"

        row_idx = np.array(terminal_nodes_list)
        col_idx = np.array(terminal_nodes_list)
        usage_matrix_terminals = usage_matrix[row_idx[:, None], col_idx]

        highest_usage = np.amax(usage_matrix_terminals)

        if highest_usage >= 0:
            highest_usage_index = np.where(
                usage_matrix_terminals == highest_usage)

            terminal_i = terminal_nodes_list[highest_usage_index[0][0]]
            terminal_j = terminal_nodes_list[highest_usage_index[1][0]]

            np_route = np.concatenate(
                ([terminal_i, terminal_j], [-1]*(max_route_size-2)))

            usage_matrix[terminal_i][terminal_j] = usage_matrix[terminal_i][terminal_j] / bias
            success = True

        else:
            "Infeasible"
            pass
    elif route_size == 3:
        "If route size = 3, we first find the highest usage between a terminal and any other node, then append the other terminal node"

        usage_matrix_term_to_any = usage_matrix[terminal_nodes_list]

        highest_usage = np.amax(usage_matrix_term_to_any)
        highest_usage_index = np.where(
            usage_matrix_term_to_any == highest_usage)

        if highest_usage >= 0:
            terminal_i = terminal_nodes_list[highest_usage_index[0][0]]
            node_j = highest_usage_index[1][0]
            usage_matrix[terminal_i][node_j] = usage_matrix[terminal_i][node_j] / bias

            mask = np.isin(adj_dict[node_j], [terminal_i, node_j], invert=True)
            mask_term = np.isin(adj_dict[node_j], terminal_nodes_list)

            "VNS_terminal = Vicinity node set (terminal) = nodes in adjency and not in route, AND a terminal"
            mask_ultimate = np.all([mask, mask_term], axis=0)
            vns_term = adj_dict[node_j][mask_ultimate]

            if len(vns_term) > 0:
                " len(vns) > 0 will always be true for södertälje and mandl, unless there are no more adj nodes not in route, but that is already checked by highest usage >= 0"

                max_vn = -1
                max_usage = -1
                for vn in vns_term:

                    vn = int(vn)
                    vn_usage = usage_matrix[node_j][vn]
                    if vn_usage >= max_usage:
                        max_usage = vn_usage
                        max_vn = vn
                if max_usage >= 0:
                    usage_matrix[node_j][max_vn] = usage_matrix[node_j][max_vn]/bias
                    np_route = np.concatenate(
                        ([terminal_i, node_j, max_vn], [-1]*(max_route_size-3)))

                    success = True

        else:
            "Infeasible"
            pass
    else:
        highest_usage = np.amax(usage_matrix)
        route = []

        if highest_usage >= 0:

            "Here we add the first two nodes"
            highest_usage_index = np.where(usage_matrix == highest_usage)

            node_i = highest_usage_index[0][0]
            node_j = highest_usage_index[1][0]
            route.append(node_i)
            route.append(node_j)

            usage_matrix[node_i][node_j] = usage_matrix[node_i][node_j]/bias

            "Here we add the remaining 'middle' nodes"
            while len(route) < route_size - 2:
                "Add any node to the route"
                "ugly code.."

                first_node = route[0]
                last_node = route[-1]
                mask_first = np.isin(adj_dict[first_node], route, invert=True)
                mask_last = np.isin(adj_dict[last_node], route, invert=True)
                vns_first = adj_dict[first_node][mask_first]
                vns_last = adj_dict[last_node][mask_last]

                choose_first_node = True
                max_vn = -1
                max_usage = -1
                for vn in vns_first:
                    vn = int(vn)
                    vn_usage = usage_matrix[first_node][vn]
                    if vn_usage >= max_usage:
                        max_usage = vn_usage
                        max_vn = vn
                for vn in vns_last:
                    vn = int(vn)
                    vn_usage = usage_matrix[last_node][vn]
                    if vn_usage >= max_usage:
                        choose_first_node = False
                        max_usage = vn_usage
                        max_vn = vn
                if max_usage >= 0:
                    if choose_first_node:
                        usage_matrix[first_node][max_vn] = usage_matrix[first_node][max_vn]/bias
                        route.insert(0, max_vn)
                    else:
                        usage_matrix[last_node][max_vn] = usage_matrix[last_node][max_vn]/bias
                        route.append(max_vn)
                else:
                    success = False
                    return success, np_route

            "Here we add two terminals"
            nodes_to_add_terminal_to = [0, -1]
            for i in nodes_to_add_terminal_to:
                first_node = route[i]
                mask_first = np.isin(adj_dict[first_node], route, invert=True)
                mask_first_term = np.isin(
                    adj_dict[first_node], terminal_nodes_list)
                mask_ult_first = np.all([mask_first, mask_first_term], axis=0)
                vns_first_term = adj_dict[first_node][mask_ult_first]
                max_vn = -1
                max_usage = -1
                for vn in vns_first_term:
                    vn = int(vn)
                    vn_usage = usage_matrix[first_node][vn]
                    if vn_usage >= max_usage:
                        max_usage = vn_usage
                        max_vn = vn
                if max_usage >= 0:
                    usage_matrix[first_node][max_vn] = usage_matrix[first_node][max_vn]/bias
                    if i == 0:
                        route.insert(0, max_vn)
                    else:
                        route.append(max_vn)
                else:
                    success = False
                    return success, np_route

            success = True
            np_route = np.concatenate(
                (route, [-1]*(max_route_size-len(route))))

    return success, np_route
