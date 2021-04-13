import numpy as np
from numpy import linalg as LA

class GenPatterns():
  def __init__(self, pattern_num): 
    self.pattern_num = pattern_num
    self.best_patterns = []

    self.pattern_freq = {}
    for i in range(56):
        self.pattern_freq[i] = 0
    
    pattern0 = np.array([[1,1,1],
                         [0,1,0],
                         [0,0,0]])

    pattern1 = np.array([[1,1,0],
                         [1,1,0],
                         [0,0,0]])

    pattern2 = np.array([[1,1,0],
                         [0,1,1],
                         [0,0,0]])

    pattern3 = np.array([[1,1,0],
                         [0,1,0],
                         [1,0,0]])

    pattern4 = np.array([[1,1,0],
                         [0,1,0],
                         [0,1,0]])

    pattern5 = np.array([[1,1,0],
                         [0,1,0],
                         [0,0,1]])

    pattern6 = np.array([[1,0,1],
                         [1,1,0],
                         [0,0,0]])
    
    pattern7 = np.array([[1,0,1],
                         [0,1,1],
                         [0,0,0]])

    pattern8 = np.array([[1,0,1],
                         [0,1,0],
                         [1,0,0]])

    pattern9 = np.array([[1,0,1],
                         [0,1,0],
                         [0,1,0]])

    pattern10 = np.array([[1,0,1],
                          [0,1,0],
                          [0,0,1]])

    pattern11 = np.array([[1,0,0],
                          [1,1,1],
                          [0,0,0]])

    pattern12 = np.array([[1,0,0],
                          [1,1,0],
                          [1,0,0]])

    pattern13 = np.array([[1,0,0],
                          [1,1,0],
                          [0,1,0]])

    pattern14 = np.array([[1,0,0],
                          [1,1,0],
                          [0,0,1]])

    pattern15 = np.array([[1,0,0],
                          [0,1,1],
                          [1,0,0]])

    pattern16 = np.array([[1,0,0],
                          [0,1,1],
                          [0,1,0]])

    pattern17 = np.array([[1,0,0],
                          [0,1,1],
                          [0,0,1]])


    pattern18 = np.array([[1,0,0],
                          [0,1,0],
                          [1,1,0]])

    pattern19 = np.array([[1,0,0],
                          [0,1,0],
                          [1,0,1]])

    pattern20 = np.array([[1,0,0],
                          [0,1,0],
                          [0,1,1]])

    pattern21 = np.array([[0,1,1],
                          [1,1,0],
                          [0,0,0]])

    pattern22 = np.array([[0,1,1],
                          [0,1,1],
                          [0,0,0]])

    pattern23 = np.array([[0,1,1],
                          [0,1,0],
                          [1,0,0]])

    pattern24 = np.array([[0,1,1],
                          [0,1,0],
                          [0,1,0]])

    pattern25 = np.array([[0,1,1],
                          [0,1,0],
                          [0,0,1]])

    pattern26 = np.array([[0,1,0],
                          [1,1,1],
                          [0,0,0]])

    pattern27 = np.array([[0,1,0],
                          [1,1,0],
                          [1,0,0]])

    pattern28 = np.array([[0,1,0],
                          [1,1,0],
                          [0,1,0]])
    
    pattern29 = np.array([[0,1,0],
                          [1,1,0],
                          [0,0,1]])

    pattern30 = np.array([[0,1,0],
                          [0,1,1],
                          [1,0,0]])

    pattern31 = np.array([[0,1,0],
                          [0,1,1],
                          [0,1,0]])

    pattern32 = np.array([[0,1,0],
                          [0,1,1],
                          [0,0,1]])

    pattern33 = np.array([[0,1,0],
                          [0,1,0],
                          [1,1,0]])

    pattern34 = np.array([[0,1,0],
                          [0,1,0],
                          [1,0,1]])

    pattern35 = np.array([[0,1,0],
                          [0,1,0],
                          [0,1,1]])

    pattern36 = np.array([[0,0,1],
                          [1,1,1],
                          [0,0,0]])

    pattern37 = np.array([[0,0,1],
                          [1,1,0],
                          [1,0,0]])

    pattern38 = np.array([[0,0,1],
                          [1,1,0],
                          [0,1,0]])

    pattern39 = np.array([[0,0,1],
                          [1,1,0],
                          [0,0,1]])

    pattern40 = np.array([[0,0,1],
                          [0,1,1],
                          [1,0,0]])

    pattern41 = np.array([[0,0,1],
                          [0,1,1],
                          [0,1,0]])

    pattern42 = np.array([[0,0,1],
                          [0,1,1],
                          [0,0,1]])

    pattern43 = np.array([[0,0,1],
                          [0,1,0],
                          [1,1,0]])

    pattern44 = np.array([[0,0,1],
                          [0,1,0],
                          [1,0,1]])

    pattern45 = np.array([[0,0,1],
                          [0,1,0],
                          [0,1,1]])

    pattern46 = np.array([[0,0,0],
                          [1,1,1],
                          [1,0,0]])
    
    pattern47 = np.array([[0,0,0],
                          [1,1,1],
                          [0,1,0]])

    pattern48 = np.array([[0,0,0],
                          [1,1,1],
                          [0,0,1]])

    pattern49 = np.array([[0,0,0],
                          [1,1,0],
                          [1,1,0]])

    pattern50 = np.array([[0,0,0],
                          [1,1,0],
                          [1,0,1]])

    pattern51 = np.array([[0,0,0],
                          [1,1,0],
                          [0,1,1]])

    pattern52 = np.array([[0,0,0],
                          [0,1,1],
                          [1,1,0]])

    pattern53 = np.array([[0,0,0],
                          [0,1,1],
                          [1,0,1]])

    pattern54 = np.array([[0,0,0],
                          [0,1,1],
                          [0,1,1]])

    pattern55 = np.array([[0,0,0],
                          [0,1,0],
                          [1,1,1]])

    self.patterns = [pattern0,  pattern1,  pattern2,  pattern3,  pattern4,  pattern5,  pattern6,  pattern7,  pattern8,  pattern9, 
                     pattern10, pattern11, pattern12, pattern13, pattern14, pattern15, pattern16, pattern17, pattern18, pattern19, 
                     pattern20, pattern21, pattern22, pattern23, pattern24, pattern25, pattern26, pattern27, pattern28, pattern29, 
                     pattern30, pattern31, pattern32, pattern33, pattern34, pattern35, pattern36, pattern37, pattern38, pattern39,
                     pattern40, pattern41, pattern42, pattern43, pattern44, pattern45, pattern46, pattern47, pattern48, pattern49,
                     pattern50, pattern51, pattern52, pattern53, pattern54, pattern55]


  def init_patterns(self, weight):
    shape = weight.shape  
    for filter_index in range(shape[3]):
        for channel_index in range(shape[2]):
            best_pattern_index = 0
            largest_norm = 0
            for n in range(56):
                norm_val = LA.norm(weight[:,:,channel_index,filter_index] * self.patterns[n])
                if norm_val>largest_norm:
                    largest_norm = norm_val
                    best_pattern_index = n
            self.pattern_freq[best_pattern_index] += 1

  def gen_patterns(self):
    self.pattern_freq = sorted(self.pattern_freq.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(self.pattern_num):
        self.best_patterns.append(self.patterns[self.pattern_freq[i][0]])
        #print(self.pattern_freq[i][0])

  def get_best_pattern(self, weight):
    shape = weight.shape  
    pattern_arr = np.zeros((shape[3],shape[2]),dtype=np.int)
    mask = np.zeros_like(weight)
    for filter_index in range(shape[3]):
        for channel_index in range(shape[2]):
            best_pattern_index = 0
            largest_norm = 0
            for n in range(self.pattern_num):
                norm_val = LA.norm(weight[:,:,channel_index,filter_index] * self.best_patterns[n])
                if norm_val>largest_norm:
                    largest_norm = norm_val
                    best_pattern_index = n
            #print("%d %d" % (self.pattern_num, best_pattern_index))
            pattern_arr[filter_index,channel_index] = best_pattern_index
            mask[:,:,channel_index,filter_index] = self.best_patterns[best_pattern_index]
    return pattern_arr, mask, weight * mask
    
def get_patterns(pattern_num):
    return GenPatterns(pattern_num)

        
    


