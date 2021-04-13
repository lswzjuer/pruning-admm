import numpy as np
from numpy import linalg as LA

def pruning_bad_connect(weight, prun_rate):
    shape = weight.shape  
    mask = np.ones_like(weight)
    norm_val = np.zeros((shape[3],shape[2]))
    zeros_3x3 = np.zeros((3,3)) 

    for filter_index in range(shape[3]):
        for channel_index in range(shape[2]):
            norm_val[filter_index,channel_index] = LA.norm(weight[:,:,channel_index,filter_index])

    pcen = np.percentile(norm_val,prun_rate)
    above_threshold = (norm_val>pcen)

    for filter_index in range(shape[3]):
        for channel_index in range(shape[2]):
            if above_threshold[filter_index, channel_index] == False:
                mask[:,:,channel_index,filter_index] = zeros_3x3

    return above_threshold, mask, weight*mask