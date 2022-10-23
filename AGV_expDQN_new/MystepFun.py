import numpy as np
from random import choice
from NextObservation import Next
from ModularSupervisor import Permit

# StepFun input: current state, obs: 1 X 7
#                action from RL policy, pattern_index: 0, 1, ... , 99 
# ...
# StepFun output: next state observation, obs_: 1 X 7
#                 reward: 1 X 1

def StepFun(obs, pattern_index, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR, E_c, B, E):
                    
        # %% Determine the available event set at the current state
        [Enable_P_S, Enable_P] = Permit(obs, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR)
        #the control pattern with the selected pattern index
        control_events = B[pattern_index]       # B only contain 0 and 1, 2^N X N (N -- num of properties)
        events_index = np.argwhere(control_events == 1) # the index of all possible actions
        # control actions are: 0 2 4 6 ... 18
        # pattern_control_c collects all possible action index
        if len(events_index) == 0:
            pattern_control_c = events_index
        else:
            pattern_control_c =[]
            for index in events_index:  
                pattern_control_c.append(index)
            pattern_control_c = np.unique(pattern_control_c)
        # pattern_control_c_event collects all possible action (e.g., 2 4 10)
        pattern_control_c_event =[]
        for i in pattern_control_c:
            pattern_control_c_event.append(E_c[i])
            
        #Modified   
        # Compute the last event 
        events_Last = list(set(E).difference(set(pattern_control_c)))
        # Remove the infesible events
        pattern = list(set(events_Last).intersection(set(Enable_P_S)))
        
        
        #Old version
        # # pattern contain all selected control actions and uncontrol actions
        # pattern = list(set(pattern_control_c_event).intersection(set(Enable_P_S)))
        # pattern = list(set(pattern).union(set(E_u)))
        # pattern = list(set(pattern).intersection(set(Enable_P)))
        

        # %% iterate to the next state
        # Calculate the running cost, if 32 is a possible event, give 50 reward
        isDone = 0
        reward = 0.1*len(pattern)
        IfAppear32 = 0
        stop_ind = 0
        all_S_ = []
        all_Enb_ = []
        
        # new version
        if len(pattern) != 0:
            action = choice(pattern)        # random selection of action, the value of action: from E_c and E_u
            obs_ = Next(obs, action, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR)    # iterate to the next state
            [Enable_P_S_, N] = Permit(obs_, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR) # if the available event set for the next observation
            # iterate all available actions in the pattern and calculate the all_S_
            for act_idx in pattern:
                S_temp_ = Next(obs, act_idx, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR)
                all_S_.append(S_temp_)
                [Enable_next_state, M] = Permit(S_temp_, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR)
                all_Enb_.append(Enable_next_state)
                
                
            if [] in all_Enb_:          # Junjun:221013 at Xidian.  If any event in the selected pattern leads to a deadlock, 
                Enable_P_S_ = []                                   # an episode is terminated
                obs_ = obs
                stop_ind = 2
                action = -1
        else:
            Enable_P_S_ = []  
            obs_ = obs
            all_S_ = [obs]
            stop_ind = 1
            action = -1
        # if no possible actions in the next state/intersection is empty set, terminate the episode
        if len(Enable_P_S_) == 0:
             isDone = 1
             reward = -30
            
        else:
            # only give reward if 32 in the pattern action
            if action == 31:
                IfAppear32 = 1
            
            # here we calculate the average value of all possible actions
            for i_action in pattern:
                # next_state = Next(obs, i_action, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR)
                # [Enable_next_state, M] = Permit(next_state, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR)
                i_reward = 0
                # when action == 19, 27, (3, 15)
                # first priority: 31; second priority: 16; third priority: 19， 27;
                if i_action in [31]:
                    i_reward = 10
                # elif i_action in [16]:
                #     i_reward = 5
                # elif i_action in [19, 27]:
                #     i_reward = 1
                
                # if len(Enable_next_state) == 0:    
                #     i_reward = -30                   

                reward += i_reward
            reward /= len(pattern)

        return(obs_, all_S_, reward, isDone, IfAppear32, stop_ind, action)
        
    
    
    