import scipy.io as sio
import numpy as np
from TrainStepFuncPattern import StepFun
from norm_state_fun import norm_state
import random
from RL_brain_class import DeepQNetwork
import tensorflow as tf
import os
from TrainNextObservation import Next


# %% Load models
load_data = sio.loadmat('Train_model_0616.mat')  #Get the model file from MATLAB 
load_data1 = sio.loadmat('B.mat')  #Get the model file from MATLAB 
st1 = load_data['st1']    # st1[i-1][j-1][k-1] <-- st1[i][j][k]
st2 = load_data['st2']
st3 = load_data['st3']
st4 = load_data['st4']

rs1 = load_data['rs1']
rs2 = load_data['rs2']
rs3 = load_data['rs3']
rs4 = load_data['rs4']

rt1 = load_data['rt1']
rt2 = load_data['rt2']
rt3 = load_data['rt3']
rt4 = load_data['rt4']

A1 = load_data['A1']
A2 = load_data['A2']
A3 = load_data['A3']
A4 = load_data['A4']
A5 = load_data['A5']
A6 = load_data['A6']
A7 = load_data['A7']
A8 = load_data['A8']
A9 = load_data['A9']
A10 = load_data['A10']
A11 = load_data['A11']
A12 = load_data['A12']

# B = load_data['B']
B = load_data1['B']

E_c = [11,13,15,17,21,23,25,27,31,33,35,37,41,43,45,47]
E_u = [10,12,14,16,20,22,24,26,30,32,34,36,40,42,44,46]

# hyperparameters defination
random.seed(0)
NUM_ACTION = 137        # only select the action pattern in 137 patterns
# NUM_OBS = 18
NUM_OBS = 14
# NUM_OBS = 12
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 400000
# INIT_OBS = [0,0,0,0,0,0,0,0,0,0,0,0]  # 0~11, 12 states
INIT_OBS = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 0~13, 14 states, last 2 states are binary
# INIT_OBS = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # 0~17, 18 states, last 5-6 states are binary, last 4 states are number of trains in stations 1-4

MAX_EPI_STEP = 200

# build network
tf.compat.v1.reset_default_graph()
RL = DeepQNetwork(NUM_ACTION, 
                  NUM_OBS,
                  learning_rate = 1e-3,
                  reward_decay = 0.98,
                  e_greedy = 0.95,
                  replace_target_iteration = 100,
                  memory_size = MEMORY_SIZE,
                  batch_size = BATCH_SIZE,
                  epsilon_increment = 2.5e-5,
                  epsilon_init = 0.10,
                  output_graph = False)
saver = tf.compat.v1.train.Saver(max_to_keep=None)
# cwd = os.getcwd()
cwd = "C:\\work\\TrainModel\\0628_exp\\"

total_step = 0
reward_history = []
good_event_history = []
episode_step_history = [0]
max_reward_his = -50

# %%train
for num_episode in range(NUM_EPISODE):
    # in each episode, reset initial state and total reward
    S = INIT_OBS
    S_norm = norm_state(S)
    episode_reward = 0
    episode_step = 0
    epi_good_event = 0
    epi_action_list = []
    num_train = 0
    # if num_episode > 200000:
    #     RL.learning_rate -= 1.25e-8
    while True:
        # initialize the Action
        A = RL.choose_action(S_norm, 1)
        # take action and observe
        [S_, all_S_, R, isDone, IfAppearGoodEvent, stop_ind, selected_action] = \
            StepFun(S, A, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, E_c, E_u, B, num_train)
        # normalize the actual next state
        S_norm_ = norm_state(S_)
        # normalize the all next states set
        all_S_norm_ = norm_state(all_S_)
        if selected_action in [10, 44]:
            num_train += 1
        elif selected_action in [15, 41]:
            num_train -= 1
        
        # store transition
        RL.store_exp(S_norm, A, R, all_S_norm_)
        # control the learning starting time and frequency
        if total_step > MEMORY_SIZE and (total_step % 25 == 0):
            RL.learn()
        # update states
        episode_reward += R
        episode_step += 1
        epi_good_event += IfAppearGoodEvent

        S = S_
        S_norm = S_norm_
        if isDone or episode_step > MAX_EPI_STEP:
            if stop_ind == 1:
                stop_reason = 'pattern is impossible'
            elif episode_step > MAX_EPI_STEP:
                stop_reason = 'reach max steps'
            else:
                stop_reason = 'next state is empty'
            if max_reward_his < episode_reward:
                max_reward_his = episode_reward
            print('episode:', num_episode, '\n',
                  'episode reward:', episode_reward, '\n',
                  'episode step:', episode_step, '\n',
                  'good event:', epi_good_event, '\n',
                  'epsilon value:', RL.epsilon, '\n',
                  'action list:', epi_action_list, '\n',
                  'maximal running step:', np.max(episode_step_history), '\n',
                  'total good event:', np.sum(good_event_history), '\n',
                  'maximal episode reward:', max_reward_his, '\n',
                  stop_reason, '\n',
                  '*******************************************'
                  )
            reward_history.append(episode_reward)
            good_event_history.append(epi_good_event)
            episode_step_history.append(episode_step)
            
            # save checkpoint model, if a good model is received
            if episode_reward > 300 or episode_step > 200:
                save_path = cwd + str(num_episode) + '_reward' + str(episode_reward) + 'step' + str(episode_step) + '.ckpt'
                saver.save(RL.sess, save_path)
            
            break
        total_step += 1
        epi_action_list.append(selected_action)

# draw cost curve
RL.plot_cost()
RL.plot_reward(reward_history, 100)
RL.plot_epiEvent(good_event_history)


# %%for testing
# restore checkpoint path
# 399963_reward520.8832714400829step201.ckpt
# 396983_reward478.33483653118174step201.ckpt
ckpt_path = 'C:\\work\\TrainModel\\0621\\396983_reward478.33483653118174step201.ckpt'

tf.reset_default_graph()
num_train_list_left = []
num_train_list_right = []
num_length = []
epi_step = []
epi_diff = []

for i in range(100):
    
    S_init = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # the first action is selected from the pattern 30
    [S_, all_S_, R, isDone, IfAppearGoodEvent, stop_ind, selected_action] = \
        StepFun(S_init, 30, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, E_c, E_u, B, 0)
    
    num_train = 0
    [action_list, pattern_list, state_list_his] = RL.check_action(200, S_, ckpt_path, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, E_c, E_u, B, num_train)
    indices_15 = [i for i, x in enumerate(action_list) if x == 15]
    indices_41 = [i for i, x in enumerate(action_list) if x == 41]
    indices = (indices_41 + indices_15)
    indices.sort()
    diff = 0
    diff_temp = 0
    for i in indices:
        if i in indices_15:
            diff_temp += 1
        else:
            diff_temp -= 1
        if abs(diff_temp) > abs(diff):
            diff = diff_temp    
    print("the length is", len(action_list), "\n")
    print("the number of 15 is", action_list.count(15), "the number of 41 is", action_list.count(41),  "\n")
    # print(np.array(action_list))
    num_length.append(len(action_list))
    num_train_list_left.append(action_list.count(15))
    num_train_list_right.append(action_list.count(41))
    epi_step.append(max(indices_15[-1], indices_41[-1]))
    epi_diff.append(abs(diff))


# %% for testing 
# specify how many trains on each side
ckpt_path = 'C:\\work\\TrainModel\\0621\\396983_reward478.33483653118174step201.ckpt'
tf.reset_default_graph()
num_train_list_left = []
num_train_list_right = []
num_length = []
epi_step = []
epi_diff = []
train_left = 6
train_right = 6

for i in range(100):
    
    S_init = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # the first action is selected from the pattern 30
    [S_, all_S_, R, isDone, IfAppearGoodEvent, stop_ind, selected_action] = \
        StepFun(S_init, 30, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4, \
                A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, E_c, E_u, B, 0, train_left, train_right)
    
    num_train = 0
    [action_list, pattern_list, state_list_his] = RL.check_action_MC(200, train_left, train_right, \
                S_, ckpt_path, st1,st2,st3,st4,rs1,rs2,rs3,rs4,rt1,rt2,rt3,rt4,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12, E_c, E_u, B, num_train)
    indices_15 = [i for i, x in enumerate(action_list) if x == 15]
    indices_41 = [i for i, x in enumerate(action_list) if x == 41]
    indices = (indices_41 + indices_15)
    indices.sort()
    diff = 0
    diff_temp = 0
    for i in indices:
        if i in indices_15:
            diff_temp += 1
        else:
            diff_temp -= 1
        if abs(diff_temp) > abs(diff):
            diff = diff_temp    
    print("the length is", len(action_list), "\n")
    print("the number of 15 is", action_list.count(15), "the number of 41 is", action_list.count(41),  "\n")
    # print(np.array(action_list))
    num_length.append(len(action_list))
    num_train_list_left.append(action_list.count(15))
    num_train_list_right.append(action_list.count(41))
    epi_step.append(max(indices_15[-1], indices_41[-1]))
    epi_diff.append(abs(diff))





