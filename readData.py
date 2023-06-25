import numpy as np
import pandas as pd 

def read_textfile(choice):

    if choice == "M":
        "Mandl's nätverk"
        #file_path_coords = "Data/mumford 2016/Instances/MandlCoords.txt"
        file_path_tt = "Data/mumford 2016/Instances/MandlTravelTimes.txt"
        file_path_demand = "Data/mumford 2016/Instances/MandlDemand.txt"
    
    elif choice == "S": 
        "Södertälje"
        file_path_tt = "Data/Sodertalje/SödertäljeTravelTimes.txt" 
        file_path_demand = "Data/Sodertalje/SödertäljeDemand.txt"
    
    
    #data_coords = pd.read_csv(file_path_coords, delim_whitespace=True, header=0, names=["x","y"])
    data_tt = pd.read_csv(file_path_tt,delim_whitespace=True,header=None)
    data_demand = pd.read_csv(file_path_demand,delim_whitespace=True, header=None)


    #pd.set_option('display.max_row', 50)
    #pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_row', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("Travel times\n", data_tt,'\n\n Demand \n', data_demand)

    return data_tt.to_numpy(), data_demand.to_numpy()

def create_adj_dictionary(data_tt):
    #Create adjacent node set dictionary for all of the nodes
    data_tt2 = pd.DataFrame(data_tt)
    adj_dict = {}
    for i, row in enumerate(data_tt2.itertuples(index=False)):
        adj_array = np.array([])
        
        for j, node in enumerate(row):
            if i != j and node != np.inf:    
                adj_array = np.append(adj_array, j)     

        adj_dict[i] = adj_array    
    
    return adj_dict


