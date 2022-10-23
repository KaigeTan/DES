# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:52:44 2022

@author: KaigeT
"""
import numpy as np

def norm_state(S):
    max_state = np.array([16, 16, 16, 16, 16, 9, 9, 9, 4, 2, 2, 4, 1, 1])       # maximal number of train model state
    # max_state = np.array([16, 16, 16, 16, 16, 9, 9, 9, 4, 2, 2, 4])       # maximal number of train model state
    S_norm_arr = np.array(S)/max_state
    S_norm = S_norm_arr.tolist()
    return S_norm