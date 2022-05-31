import numpy as np
import pandas as pd
import copy

def parse_coordinates_from_name(name:str):
    if "<" in name:
        name = name.split("<")[0]
    return tuple([int(i) for i in name.split(":")[1].replace("]","").split(",")])

def parse_coordinates_from_names(names:list):
    return [parse_coordinates_from_name(name) for name in names]

def parse_shape_from_name(name):
    relevant = name.split("<")[1].replace(">","").split(":")
    dims = int(relevant[0])
    shape = [int(i) for i in relevant[1].split(",")]

    return dims, shape

def get_chunks(column_names:list):
    unique_names_ranges = []
    unique_indice_names = []
    index = 0
    for name in column_names:
        if "<" in name:
            unique_names_ranges.append(name.split("[")[0])
            unique_indice_names.append(index)
        index+=1

    return unique_names_ranges, unique_indice_names

def create_matrices(dataframe:pd.DataFrame):
    columns = dataframe.columns[1:]
    numeric_data = dataframe[columns].values
    size = len(dataframe)

    names, name_ranges = get_chunks(columns)

    dict_to_matrices = {}
    for i in range(len(name_ranges)):
        print(names[i])
        matrix = create_matrix(size, numeric_data, columns, name_ranges, i)
        dict_to_matrices[names[i]] = matrix

    trial_names = dataframe[dataframe.columns[0]].values
    return dict_to_matrices, trial_names

def create_matrix(number_examples, numeric_data, columns, name_ranges:list, current_index:int):
    dims, shape = parse_shape_from_name(columns[name_ranges[current_index]])
    shape.insert(0,number_examples) #add full size to the matrix

    full_input = np.zeros(shape)

    if current_index + 1 < len(name_ranges):
        relevant_coords = parse_coordinates_from_names(columns[name_ranges[current_index]:name_ranges[current_index+1]]) #where to put an element in
    else:
        relevant_coords = parse_coordinates_from_names(columns[name_ranges[current_index]:]) #where to put an element in

    for i in range(len(numeric_data)):
        for j in range(len(relevant_coords)):
            index = relevant_coords[j]
            full_input[i][index] = numeric_data[i][j]

    return full_input

if __name__ == '__main__':

    temporary = pd.read_csv("TrainAB_Sample.csv",sep=",")

    complex_multidimensional_dict, row_names = create_matrices(temporary )

    for key in complex_multidimensional_dict:
        print(complex_multidimensional_dict[key].shape)

    temporary = pd.read_csv("random_5x5_25_gen.tsv",sep="\t")

    simple_multidimensional_dict, row_names = create_matrices(temporary )

    for key in simple_multidimensional_dict:
        print(simple_multidimensional_dict[key].shape)

