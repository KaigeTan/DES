import numpy as np
from random import choice
from TrainNextObservation import Next
from TrainModularSupervisor import Permit



def StepFun(obs, pattern_index, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4,\
            A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,E_c, E_u, B, num_train, train_left, train_right):
    # %% Determine the available event set at the current state
    [Enable_P_S, Enable_P] = Permit(obs, A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12)
    # # only select pattern from 0 - 136
    # pattern_index = pattern_index
    #the control pattern with the selected pattern index
    control_events = B[pattern_index]       # B only contain 0 and 1, 2^N X N (N -- num of properties)
    events_index = np.argwhere(control_events == 1) # the index of all possible actions
    
    # pattern_control_c collects all possible controllable action index in B[pattern_index]
    if len(events_index) == 0:
        pattern_control_c = events_index
    else:
        pattern_control_c =[]
        for index in events_index:  
            pattern_control_c.append(index)
        pattern_control_c = np.unique(pattern_control_c)
    
    pattern_control_c_event =[]
    for i in pattern_control_c:
        pattern_control_c_event.append(E_c[i])
    
    
   
    # pattern contain all selected control actions and uncontrol actions
    pattern = list(set(pattern_control_c_event).intersection(set(Enable_P_S)))
    pattern = list(set(pattern).union(set(E_u)))
    pattern = list(set(pattern).intersection(set(Enable_P)))
    
    # remove action that enter train if already > 4 trains in the system
    # remove action that enter train if exceed the number of trains on each side
    if (10 in pattern and num_train > 4) or train_left == 0:
        pattern.remove(10)
    if 44 in pattern and num_train > 4 or train_right == 0:
        pattern.remove(44)
    
    # %% iterate to the next state
    # reward definition
    def reward_cal(x):
        return 2/(1+np.exp(-2*(x+1))) - 0.762
    # Calculate the running cost, if 32 is a possible event, give 50 reward
    isDone = 0
    reward = 0
    stop_ind = 0
    all_S_ = []
    
    if len(pattern) != 0:                                          
        action = choice(pattern) - 1        # random selection of action, the value of action: from E_c and E_u
        obs_ = Next(obs, action, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4)    # iterate to the next state
        [Enable_P_S_, N] = Permit(obs_, A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12) # if the available event set for the next observation
        # iterate all available actions in the pattern and calculate the all_S_
        for act_idx in pattern:
            act_idx -= 1        # pattern follows MATLAB naming, from 1; act_idx calls python array, from 0
            S_temp_ = Next(obs,act_idx,st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4)
            all_S_.append(S_temp_)
    else:
        Enable_P_S_ = []
        all_S_ = [obs]
        obs_ = obs
        stop_ind = 1
        action = -1
    # if no possible actions in the next state/intersection is empty set, terminate the episode
    if len(Enable_P_S_) == 0:
         isDone = 1
         reward = -30
    else:
        X13 = obs[-2]
        X14 = obs[-1]
        ratio_inf = 1.5
        # here we calculate the average value of all possible actions
        for i_action in pattern:
            i_reward = 0
            
            # 15 ---> state14, 41 ---> state13
            if i_action == 41:
                i_reward = ratio_inf*20*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*20
            elif i_action == 15:
                i_reward = 20*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*20
            elif i_action == 42:
                i_reward = ratio_inf*10*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*20
            elif i_action == 16:
                i_reward = 10*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*20
            elif i_action == 40:
                i_reward = ratio_inf*5*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*10
            elif i_action == 14:
                i_reward = 5*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*10
            elif i_action == 32:
                i_reward = ratio_inf*2.5*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*5
            elif i_action == 26:
                i_reward = 2.5*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*5
            elif i_action == 30:
                i_reward = ratio_inf*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*2.5
            elif i_action == 24:
                i_reward = reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*2.5
            elif i_action == 22:
                i_reward = ratio_inf*0.5*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*1
            elif i_action == 36:
                i_reward = 0.5*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*1
            elif i_action == 20:
                i_reward = ratio_inf*0.25*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*0.5
            elif i_action == 34:
                i_reward = 0.25*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*0.5
            elif i_action == 12:
                i_reward = ratio_inf*0.1*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*0.25
            elif i_action == 46:
                i_reward = 0.1*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*0.25
            elif i_action == 10:
                i_reward = ratio_inf*0.05*reward_cal(X14 - X13) # np.exp(0.7*(X14 - X13))*0.1
            elif i_action == 44:
                i_reward = 0.05*reward_cal(X13 - X14) # np.exp(0.7*(X13 - X14))*0.1
            
            reward += i_reward
        reward /= len(pattern)
        
    IfAppearGoodEvent = 1 if action in [15, 41] else 0

    return(obs_, all_S_, reward, isDone, IfAppearGoodEvent, stop_ind, action)
    
    