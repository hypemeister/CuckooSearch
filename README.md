# How it works
Firstly, one has to choose between Mandl's network or the Södertälje network. This is done by either "choice="M"" or "choice="S"".

One can choose not to solve the multi-objective version by setting "multi_objective = False". 

The parameters for each problem case can then be set, e.g., max route size, number of routes in a route set (route_set_size), and which nodes are terminal nodes. 

The data for each problem case is then read by the readdata functions. The "full_graph" is also created, which is a graph with nodes and edges with travel times for the full network. 

The parameters for the MODCS algorithm is then set, e.g., population size (n), number of generations to be run (max_generation), p_a, etc. The "feasibility_buffer" is used to create some extra route sets in the initialization if some of them are infeasible. If one wants to change the transfer penalty or the depth of the local search algorithm, these can be found in "fitness.py" or "perturbations.py".

The seed number for the random generator is then set, as well as if one wants to save the results with "save_results". The k_vector and l_vector are then set, as well as "number_of_big_pert" which is the number of routes to be exchanged in the worst CS procedure. More details can be found in my thesis. 

The initial population is then created with either "createpop" or "createpop2" functions, the former being K&B 2014's initialization and the latter being Svensson's initialization. All of the route sets in the initial population are then evaluated with the fitness functions. The population is then sorted. 

Lastly, the main loop is run and each loop represents one generation. Each loop, "multi_smartCS", "multi_getCuckoo" and "multi_worstCS" are being called. These three functions will make changes to the route sets and improve the population. 


# Some functions
## Perturbations
### Local Search (for the multi-objective case)
The local_search function returns a routeset, its fitness and a population container. It is not exhaustive since it will use the first improvement it finds. It is recursive so it will try to find improvements until no further improvements can be found. 
If the local search only should have a depth of one, i.e., it finds an improvement and is done, one can simply comment away the recursive function call. 
If there should be no local search, one can simply return the given routeset, its fitness and the population container in the beginning of the function. 
The local search will try 20 (neighborhood_size) switches of nodes to find an improvement. 

### Perform exchange part of route mode 1
Used perform the exchange part of route function. That function will take a part of a route in a routeset and try to replace with a part from a route from another routeset. 

### Method 2
Will try to replace entire routes in a routeset with routes from another routeset. 

## Sorting Algorithms
### Is better
Sees which of two multi-objective solutions are better than the other. If no such distinction can be made, it returns False.
Otherwise, if the solution is better than the one it compares to, it returns True, new population container and new solution index for the its place in the new population. 

### Sort population 
The single-objective sort function 

### Multi sort population
The multi-objective sort function. It returns the newly sorted population and its corresponding population fitness. It also returns the population fronts, the sizes of the different population fronts, and a list of the sorted indices. 
The list of the sorted indices is needed for the is_better function. The population front array is composed by the front numbers and the crowding distance values, e.g., population_front[i] = (front_k (that sol i belongs to), crowding distance (of solution i))

## Dijkstar Graph and Algorithm
It is a python package for a graph structure and functions to find the shortest path between two nodes, using Dijkstra's algorithm. It was downloaded in order to be able to use all of the functions. 

# Improvements to be made

## Use a global config file to store information and parameters 
Instead of having to use a billion different input arguments in the function, using a global config file for problem constants and parameters is much easier. Would be important, since it is easy to forget the transfer penalty in the fitness-file when changing between Mandl's and Södertälje. 

## Create a depth parameter for local search, and a parameter for if local search should be done

## Add a terminal nodes check for the feasibility function or the initial population function of K&B 2014
At this moment, the feasibility function do not check that both end-nodes are terminal nodes. This is not a problem, since the initial-population function of Svensson do, and the exchange part of route mode 1 perturbation and local search do as well. However, if one were to use the initial-population function of K&B 2014 for the Södertälje problem, no check for terminal nodes in routes will happen and it will have an effect. However, after enough generations, the problem will might 'fix' itself, since the perturbations and will slowly enforce the terminal nodes constraint. 

## Initialization for the Södertälje problem
It is of course very unnecessary to run the initialization algorithm to generate an initial population for the Södertälje problem every time since it is deterministic and thus will return the same initial population every time. It would be much better to just use the "importPhilipsNetworks.py" script, in misc functions, to import as many of the initial networks as one would want! However, at the moment, it only has 50 initial networks stored and only with route set size = 18!


# Thoughts
## About the perturbation functions
### Exchange part of route move 1
If there is a cycle, one could do try to the other part of route 2 that haven't been tested. This has not been implemented. 
