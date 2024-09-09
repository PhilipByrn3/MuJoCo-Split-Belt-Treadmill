import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


def convert_dataframe(dataframe):
    array = dataframe.to_numpy()
    belt_diff_axis = []
    steady_velocity_avg = []
    i = 1

    print(array)
    for sub_array in array:
        for datum in sub_array:
            
            if i%2 != 0:
                belt_diff_axis.append(datum)
            elif i%2 == 0:
                steady_velocity_avg.append(datum)
            i+=1
    return belt_diff_axis, steady_velocity_avg

def generate_graph(belt_diff_axis, steady_velocity_avg):
    figure, ax = plt.subplots()
    ax.plot(belt_diff_axis, steady_velocity_avg)
    plt.show()
    return
