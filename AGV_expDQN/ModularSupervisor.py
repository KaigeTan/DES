# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:36:54 2022

@author: Jonny
"""

from AvailableEvents import Enb
import numpy as np


def Permit(obs,AGV_1,AGV_2,AGV_3,AGV_4,AGV_5,SUP_IPSR,SUP_ZWSR):
        
    Enable_P1 = Enb(obs[0], AGV_1)   #define Enb function
    Enable_P2 = Enb(obs[1], AGV_2)
    Enable_P3 = Enb(obs[2], AGV_3)
    Enable_P4 = Enb(obs[3], AGV_4)
    Enable_P5 = Enb(obs[4], AGV_5)
    
    Enable_P = np.union1d(Enable_P1, Enable_P2)
    Enable_P = np.union1d(Enable_P, Enable_P3)
    Enable_P = np.union1d(Enable_P, Enable_P4)
    Enable_P = np.union1d(Enable_P, Enable_P5)
    
    
    Enable_B1SUP = Enb(obs[5], SUP_IPSR)
    Enable_B2SUP = Enb(obs[6], SUP_ZWSR)    
    Enable = np.intersect1d(Enable_B1SUP, Enable_B2SUP)
    
    Enable_P_S = np.intersect1d(Enable_P, Enable)
    
    return(Enable_P_S,Enable_P)