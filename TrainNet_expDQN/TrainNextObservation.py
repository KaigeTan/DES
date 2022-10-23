import numpy as np



def Next(State,action,st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4):

    X1 = State[0]   
    X2 = State[1]
    X3 = State[2]
    X4 = State[3] 
    X5 = State[4]
    X6 = State[5]
    X7 = State[6]
    X8 = State[7]  
    X9 = State[8]
    X10 = State[9]
    X11 = State[10] 
    X12 = State[11]
    X13 = State[12]
    X14 = State[13]
    # num_S1 = State[14] 
    # num_S2 = State[15]
    # num_S3 = State[16]
    # num_S4 = State[17]
    
    
    if action <= 16:       # 10 ~ 17 -> 9~16 Python
        X1_ = np.where(st1[X1, :, action] == 1)
        X1_ = X1_[0][0]
    else:
        X1_ = X1
    
    if action > 16 and action <= 26:
        X2_ = np.where(st2[X2, :, action] == 1)
        X2_ = X2_[0][0]
    else:
        X2_ = X2
       
    if action > 26 and action <= 36:
        X3_ = np.where(st3[X3, :, action] == 1)
        X3_ = X3_[0][0]
    else:
       X3_ = X3
        
    if action > 36 and action <= 46:  
        X4_ = np.where(st4[X4, :, action] == 1)
        X4_ = X4_[0][0]
    else:
        X4_ = X4
        
    X5_ = np.where(rs1[X5, :, action] == 1)   
    X5_ = X5_[0][0]
    
    X6_ = np.where(rs2[X6, :, action] == 1)   
    X6_ = X6_[0][0]
    
    X7_ = np.where(rs3[X7, :, action] == 1)   
    X7_ = X7_[0][0]
    
    X8_ = np.where(rs4[X8, :, action] == 1)   
    X8_ = X8_[0][0]
    
    X9_ = np.where(rt1[X9, :, action] == 1)   
    X9_ = X9_[0][0]
    
    X10_ = np.where(rt2[X10, :, action] == 1)   
    X10_ = X10_[0][0]
    
    X11_ = np.where(rt3[X11, :, action] == 1)   
    X11_ = X11_[0][0]
    
    X12_ = np.where(rt4[X12, :, action] == 1)   
    X12_ = X12_[0][0]
    
    X14_ = X14 + 1 if action == 15 else X14
    X13_ = X13 + 1 if action == 41 else X13

            
    # if action in [10, 14]:
    #     num_S1 += 1
    # elif action in [11, 15]:
    #     num_S1 -= 1
    # elif action in [20, 24]:
    #     num_S2 += 1
    # elif action in [21, 25]:
    #     num_S2 -= 1
    # elif action in [30, 34]:
    #     num_S3 += 1
    # elif action in [31, 35]:
    #     num_S3 -= 1    
    # elif action in [40, 44]:
    #     num_S4 += 1
    # elif action in [41, 45]:
    #     num_S4 -= 1
    State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_, X8_, X9_, X10_, X11_, X12_, X13_, X14_]
    # State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_,X8_, X9_, X10_, X11_, X12_]
    # State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_,X8_, X9_, X10_, X11_, X12_, X13_, X14_, num_S1, num_S2, num_S3, num_S4]
    return(State_)
