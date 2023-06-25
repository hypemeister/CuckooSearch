import numpy as np


def feasability_and_repair(route_set, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, repair=False):
    "Returns True if the route set is feasible, otherwise False. If repair = True, then this function will also try to repair the route set"
    "It repairs the route set the same way as Soares Mumford etc 2019"

    nodes_in_network = np.array(list(range(number_of_nodes)))
    missing_nodes_mask = np.isin(
        element=nodes_in_network, test_elements=route_set, invert=True)
    missing_nodes = set(nodes_in_network[missing_nodes_mask])
    missing_terminal_nodes = missing_nodes.intersection(terminal_nodes)
    missing_nodes_container = (missing_nodes, missing_terminal_nodes)

    if len(missing_nodes) > 0:
        "It is missing nodes, so it is infeasible"

        if repair == True:
            if len(missing_terminal_nodes) > 0:
                "X is the step counter"
                "X_max should be decided some other way I believe"
                X = 1
                minus_ones = np.argwhere(route_set < 0)
                if len(minus_ones[:, 1]) == 0:
                    "Every route in the route set is of full length already"
                    return False
                else:
                    X_max = max_route_size - min(minus_ones[:, 1])
                    #X_max = 2
                    success_phase_1, missing_nodes = repair_phase_1(
                        route_set, route_set_size, full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, X, X_max, missing_nodes_container)
                    if success_phase_1 == False:
                        return False

            success_phase_2 = False
            if len(missing_nodes) > 0:
                success_phase_2 = repair_phase_2(
                    route_set, route_set_size, full_graph, full_graph_shortest_paths,  max_route_size, number_of_nodes, rng, missing_nodes)

            if len(missing_nodes) == 0 or success_phase_2:
                "Have to check for secluded routes also"
                no_secluded_routes = check_for_secluded_routes(
                    route_set, route_set_size, max_route_size)
                if no_secluded_routes:
                    return True
                else:
                    return False
            else:
                return False
    else:
        "It is not missing nodes. We don't check for terminal nodes since the initial population functions do that, and the perturbations also do that."
        "However, it may need to be implemented if one were to use 'creatInitialPop' with the K&B 2014 function for the Södertälje network or any problem case where the set of terminal nodes is a subset of all the nodes "
        no_secluded_routes = check_for_secluded_routes(
            route_set, route_set_size, max_route_size)
        if no_secluded_routes:
            return True
        else:
            return False


def repair_phase_1(route_set, route_set_size,  full_graph, full_graph_shortest_paths, terminal_nodes, max_route_size, number_of_nodes, rng, X, X_max, missing_nodes_container):
    "repair phase 1 from soares mumford etc 2019"
    "går att optimera genom att break loopsen när de tex hittat första terminal loopen inom x steg"

    success = False
    route_rng_indices = list(range(route_set_size))
    missing_nodes, missing_terminal_nodes = missing_nodes_container
    go_to_phase_B = False

    while len(route_rng_indices) > 0:
        go_to_phase_B = False

        "Select a random route from the route_set (without replacement)"
        temp_rng_index = rng.integers(len(route_rng_indices))
        route_rng_indices[temp_rng_index], route_rng_indices[-1] = route_rng_indices[-1], route_rng_indices[temp_rng_index]
        route_index = route_rng_indices.pop()

        selected_route = route_set[route_index]
        for i, node in enumerate(selected_route[::-1]):
            if node != -1:
                node_last_index = len(selected_route) - i - 1
                node_last = node
                break

        node_first = selected_route[0]
        selected_route_size = node_last_index + 1

        if selected_route_size + X <= max_route_size:
            "We can fit another node in this route"
            within_X_steps_dict = dict()

            for missing_terminal_node in missing_terminal_nodes:
                "nl = node last"
                nl_edges_traversed, path = full_graph_shortest_paths[node_last][missing_terminal_node]

                if nl_edges_traversed <= X:
                    "Then we can reach a missing terminal node, within X steps"

                    # We only select the last one in the dict, which means that if there are multiple terminal nodes within X steps, only one (the last) will be saved
                    within_X_steps_dict[node_last] = (nl_edges_traversed, path)
                    go_to_phase_B = True

            if go_to_phase_B == False:
                for missing_terminal_node in missing_terminal_nodes:
                    "nf = node first"
                    nf_edges_traversed, path = full_graph_shortest_paths[
                        node_first][missing_terminal_node]

                    if nf_edges_traversed <= X:
                        "Then we can reach a missing terminal node, within X steps (from first node)"

                        temp_route = np.flip(
                            selected_route[:node_last_index+1])
                        selected_route = np.concatenate(
                            (temp_route, selected_route[node_last_index+1:]))

                        node_last = node_first
                        node_first = selected_route[0]
                        within_X_steps_dict[node_last] = (
                            nf_edges_traversed, path)
                        go_to_phase_B = True

            if go_to_phase_B == True:
                "Go to phase B in repair (Connecting terminal to route)"
                selected_route_original = selected_route
                node_last_index_original = node_last_index
                selected_route_size_original = selected_route_size
                X_original = X
                added_missing_nodes = set()

                while True:
                    if X == 1:
                        steps, path = within_X_steps_dict[node_last]
                        if steps == 0:
                            "BANDAID fix? then it tries to add itself which obviously wont work"
                            pass
                        else:
                            "PATH MUST BE 2 element list"
                            nmb_of_no_nodes = max_route_size - selected_route_size - 1
                            selected_route = np.concatenate(
                                (selected_route[:node_last_index], path, [-1]*nmb_of_no_nodes))

                            # Finns nog bättre sätt än det här
                            added_missing_nodes.add(path[-1])

                        missing_terminal_nodes.difference_update(
                            added_missing_nodes)
                        missing_nodes.difference_update(added_missing_nodes)

                        route_set[route_index] = selected_route
                        "This one is weird, but we add the route back into route list so it can be chosen again, to avoid some situations when another terminal can be added within one step (X=1)"
                        "but the route index list is empty so X is automatically increased to two."
                        "It happens because we can see that we can reach a terminal node WITHIN X steps, which means that we can reach it in LESS than X steps which produces error"
                        route_rng_indices.append(route_index)

                        "WE CANNOT ADD X = original X here. By not doing this, it allows us to get X = 1 over and over again"
                        break
                    else:
                        nodes_k = []
                        "Correspond to first shifted square in phase B"
                        for missing_node in missing_nodes:
                            if missing_node in full_graph.get_data()[node_last]:
                                for missing_terminal_node in missing_terminal_nodes:
                                    nl_edges_traversed, path = full_graph_shortest_paths[
                                        missing_node][missing_terminal_node]
                                    if nl_edges_traversed <= X-1:
                                        nodes_k.append(missing_node)

                                        if missing_node not in within_X_steps_dict:
                                            within_X_steps_dict[missing_node] = (
                                                nl_edges_traversed, path)
                                        break
                        "Correspond to second shifted square in phase B"
                        if len(nodes_k) == 0:
                            for neighbor_node in full_graph.get_data()[node_last]:
                                if neighbor_node not in selected_route:
                                    for missing_terminal_node in missing_terminal_nodes:
                                        nl_edges_traversed, path = full_graph_shortest_paths[
                                            neighbor_node][missing_terminal_node]
                                        if nl_edges_traversed <= X-1:
                                            nodes_k.append(neighbor_node)
                                            if missing_node not in within_X_steps_dict:
                                                within_X_steps_dict[neighbor_node] = (
                                                    nl_edges_traversed, path)
                                            break
                        if len(nodes_k) == 0:
                            "If no possible nodes still.. We restore route and X, and leave phase B"
                            selected_route = selected_route_original
                            selected_route_index = selected_route_original
                            selected_route_size = selected_route_original
                            temp_added_missing_terminal_nodes = added_missing_nodes.intersection(
                                terminal_nodes)
                            missing_terminal_nodes.update(
                                temp_added_missing_terminal_nodes)
                            missing_nodes.update(added_missing_nodes)
                            X = X_original
                            break
                        else:
                            rng_index_2 = rng.integers(len(nodes_k))
                            node_k = nodes_k[rng_index_2]
                            node_last = node_k

                            nmb_of_no_nodes = max_route_size - selected_route_size - 1
                            selected_route = np.concatenate(
                                (selected_route[:node_last_index+1], [node_last], [-1]*nmb_of_no_nodes))

                            if node_last in missing_nodes:
                                "Only add a node to 'added missing nodes set', if it is missing"
                                added_missing_nodes.add(node_last)

                            missing_terminal_nodes.difference_update(
                                added_missing_nodes)
                            missing_nodes.difference_update(
                                added_missing_nodes)
                            node_last_index += 1
                            selected_route_size += 1
                            X -= 1
            if len(missing_terminal_nodes) == 0:
                "No missing terminal nodes anymore, can move on from repair phase 1"
                success = True
                return success, missing_nodes

        else:
            pass

    X += 1
    if X <= X_max and success == False:

        missing_nodes_container = (missing_nodes, missing_terminal_nodes)
        success, missing_nodes = repair_phase_1(route_set, route_set_size, full_graph, full_graph_shortest_paths,
                                                terminal_nodes, max_route_size, number_of_nodes, rng, X, X_max, missing_nodes_container)

    return success, missing_nodes


def repair_phase_2(route_set, route_set_size, full_graph, full_graph_shortest_paths, max_route_size, number_of_nodes, rng, missing_nodes):
    "repair phase 2 from soares mumford etc 2019"
    if len(missing_nodes) == 0:
        return True
    else:
        success = False
    missing_node = missing_nodes.pop()
    route_rng_indices = list(range(route_set_size))

    "Blue phase A in soares mumford etc 2019"
    while len(route_rng_indices) > 0:

        temp_rng_index = rng.integers(len(route_rng_indices))
        route_rng_indices[temp_rng_index], route_rng_indices[-1] = route_rng_indices[-1], route_rng_indices[temp_rng_index]
        route_index = route_rng_indices.pop()
        selected_route = route_set[route_index]

        if selected_route[-1] == -1:
            "Then we know that len(selected_route) + 1 <= max_route_size"

            A_list = []
            A_list_to_pick_from = []
            for index, route_node in enumerate(selected_route):
                "Check if any neighboring nodes in the route"
                if route_node in full_graph.get_data()[missing_node]:
                    A_list.append(route_node)
                    A_list_to_pick_from.append((route_node, index))
            if len(A_list) >= 2:
                first_check = False
                "Check if there are at least two neighboring nodes in the route"
                while len(A_list_to_pick_from) > 0:
                    node_a, node_a_index = A_list_to_pick_from.pop()

                    if node_a_index != 0:
                        "Check so that node a is not first, or last"
                        if node_a_index + 1 != max_route_size:
                            if selected_route[node_a_index + 1] != -1:
                                first_check = True
                    if first_check == True:
                        node_b = selected_route[node_a_index-1]
                        if node_b in A_list:
                            "Checking 'in' with a list, not very fast for large A-lists"
                            "Insert missing node between node_b and node_a"

                            selected_route = np.concatenate((selected_route[:node_a_index], [
                                                            missing_node], selected_route[node_a_index:-1]))
                            route_set[route_index] = selected_route

                            "Call repair function again"
                            success = repair_phase_2(
                                route_set, route_set_size, full_graph, full_graph_shortest_paths, max_route_size, number_of_nodes, rng, missing_nodes)
                            return success
                        else:
                            node_c = selected_route[node_a_index+1]
                            if node_c in A_list:
                                selected_route = np.concatenate(
                                    (selected_route[:node_a_index+1], [missing_node], selected_route[node_a_index+1:-1]))
                                route_set[route_index] = selected_route

                                "Call repair function again"
                                success = repair_phase_2(
                                    route_set, route_set_size, full_graph, full_graph_shortest_paths, max_route_size, number_of_nodes, rng, missing_nodes)
                                return success
            else:
                "Try to select another route"
                pass

        else:
            "not enough room to add another node to this route"
            pass

    "When arriving here in the function, it means that the missing node can't be added and the route set is infeasible"
    "Have to re-add the missing node to missing_nodes, since that is how the rest of the feasiblity function works"
    missing_nodes.add(missing_node)
    return success


def check_for_secluded_routes(route_set, route_set_size, max_route_size):
    "Returns true if there are no secluded routes"

    list_of_sets = []
    for i in range(route_set_size):

        new_set = {i}

        # Never matching elements to get indexing rate, it takes the place of the route being tested against
        never_matching_elements = -10*np.ones(shape=(1, max_route_size))

        elements = np.concatenate(
            (route_set[0:i], never_matching_elements, route_set[i+1:]))

        for ind, minus_one_node in enumerate(route_set[i][::-1]):
            if minus_one_node != -1:
                last_non_zero_index = max_route_size-ind-1
                break

        mask = np.isin(elements, route_set[i][0:last_non_zero_index + 1])

        b = np.nonzero(mask)

        for j in b[0]:
            new_set.add(j)

        if i == 0:
            list_of_sets = [new_set]

        else:
            list_of_new_set_macthes = []
            for k, main_set in enumerate(list_of_sets):
                if not main_set.isdisjoint(new_set):

                    list_of_new_set_macthes.append(k)

            if len(list_of_new_set_macthes) == 0:
                "no matches"
                list_of_sets.append(new_set)

            else:
                new_set_list = []

                for other_set in [list_of_sets[i] for i in list_of_new_set_macthes]:
                    new_set = new_set.union(other_set)

                new_set_list = [new_set]

                for m, settet in enumerate(list_of_sets):
                    if m not in list_of_new_set_macthes:
                        new_set_list.append(settet)

                list_of_sets = new_set_list

    if len(list_of_sets) > 1:
        return False
    else:
        return True
