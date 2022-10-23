
import numpy as np




def Permit(obs,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12):
    
    Enable_P1 = A1[0][obs[0]]
    Enable_P2 = A2[0][obs[1]]
    Enable_P3 = A3[0][obs[2]]
    Enable_P4 = A4[0][obs[3]]
    
    Enable_P = np.union1d(Enable_P1, Enable_P2)
    Enable_P = np.union1d(Enable_P, Enable_P3)
    Enable_P = np.union1d(Enable_P, Enable_P4)    # Plant permits
    
    Enable_P5 = A5[0][obs[4]]
    Enable_P6 = A6[0][obs[5]]
    Enable_P7 = A7[0][obs[6]]
    Enable_P8 = A8[0][obs[7]]
    Enable_P9 = A9[0][obs[8]]
    Enable_P10 = A10[0][obs[9]]
    Enable_P11 = A11[0][obs[10]]
    Enable_P12 = A12[0][obs[11]]  #modular supervisor permits
    
    Enable_S = np.intersect1d(Enable_P5, Enable_P6)
    Enable_S = np.intersect1d(Enable_S, Enable_P7)
    Enable_S = np.intersect1d(Enable_S, Enable_P8)
    Enable_S = np.intersect1d(Enable_S, Enable_P9)
    Enable_S = np.intersect1d(Enable_S, Enable_P10)
    Enable_S = np.intersect1d(Enable_S, Enable_P11)
    Enable_S = np.intersect1d(Enable_S, Enable_P12)
    
    Enable_P_S = np.intersect1d(Enable_S, Enable_P)
    
    
    
    return(Enable_P_S,Enable_P)
    
    