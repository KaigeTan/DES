# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:52:44 2022

@author: KaigeT
"""
import numpy as np

def norm_state(S):
    max_state = np.array([3, 7, 3, 5, 3, 1, 255])
    S_norm_arr = np.array(S)/max_state
    S_norm = S_norm_arr.tolist()
    return S_norm