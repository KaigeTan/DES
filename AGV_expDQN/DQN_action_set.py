import scipy.io as sio
import numpy as np
from MystepFun import StepFun
from norm_state_fun import norm_state
import random
from RL_brain_class import DeepQNetwork
import tensorflow as tf
import os

# %% Load models
load_data = sio.loadmat('test3.mat')  #Get the model file from MATLAB 
load_dataB = sio.loadmat('B_2pattern_AGV.mat')  #Get the model file from MATLAB 
AGV_1 = load_data['AGV_1'] 
AGV_2 = load_data['AGV_2'] 
AGV_3 = load_data['AGV_3'] 
AGV_4 = load_data['AGV_4'] 
AGV_5 = load_data['AGV_5'] 
SUP_IPSR = load_data['SUP_IPSR']
SUP_ZWSR = load_data['SUP_ZWSR']
B = load_dataB['B']   #0 ~ 55
E_c = range(0,20,2)  # 0 ~ 18, in MATLAB, it is 1~19, controllable events
E_u = range(1,33,2)  # 1 ~ 31, in MATLAB, it is 2~32, uncontrollable events

# %% hyperparameters defination
random.seed(5)
NUM_ACTION = 56
NUM_OBS = 7
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 130000
# initial state
INIT_OBS = [0, 0, 0, 0, 0, 0, 0]

# build network
tf.reset_default_graph()
RL = DeepQNetwork(NUM_ACTION, 
                  NUM_OBS,
                  learning_rate = 1e-3,
                  reward_decay = 0.98,
                  e_greedy = 0.95,
                  replace_target_iteration = 100,
                  memory_size = MEMORY_SIZE,
                  batch_size = BATCH_SIZE,
                  epsilon_increment = 5e-5,
                  epsilon_init = 0.10,
                  output_graph = False)
saver = tf.compat.v1.train.Saver(max_to_keep=None)
cwd = os.getcwd()

total_step = 0
reward_history = []
good_event_history = []
episode_step_history = [0]
max_epi_reward = -30

# %% train
for num_episode in range(NUM_EPISODE):
    # in each episode, reset initial state and total reward
    S = INIT_OBS
    S_norm = norm_state(S)
    episode_reward = 0
    episode_step = 0
    epi_good_event = 0
    epi_action_list = []
    if num_episode > 35000 and num_episode < 75000:
        RL.learning_rate -= 2e-8
    while True:         
        # initialize the Action
        A = RL.choose_action(S_norm)
        # take action and observe
        [S_, all_S_, R, isDone, IfAppear32, stop_ind, selected_action] = \
            StepFun(S, A, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR, E_c, E_u, B)
        S_norm_ = norm_state(S_)
        # if len(all_S_) > 1:
        #     print("goes here")
        all_S_norm_ = norm_state(all_S_)
        
        # store transition
        RL.store_exp(S_norm, A, R, all_S_norm_)
        # control the learning starting time and frequency
        if total_step > MEMORY_SIZE and (total_step % 10 == 0):
            RL.learn()
        # update states
        episode_reward += R
        episode_step += 1
        epi_good_event += IfAppear32
        S = S_
        S_norm = S_norm_
        if isDone or episode_step > 200:
            if stop_ind == 1:
                stop_reason = 'pattern is impossible'
            elif episode_step > 200:
                stop_reason = 'reach 200 steps'
            else:
                stop_reason = 'next state is empty'
            if max_epi_reward < episode_reward:
                max_epi_reward = episode_reward
            print('episode:', num_episode, '\n',
                  'episode reward:', episode_reward, '\n',
                  'episode step:', episode_step, '\n',
                  'good event:', epi_good_event, '\n',
                  'epsilon value:', RL.epsilon, '\n',
                  'action list:', epi_action_list, '\n',
                  'maximal running step:', np.max(episode_step_history), '\n',
                  'maximal episode reward:', max_epi_reward, '\n',
                  'total good event:', np.sum(good_event_history), '\n',
                  stop_reason, '\n',
                  '*******************************************'
                  )
            reward_history.append(episode_reward)
            good_event_history.append(epi_good_event)
            episode_step_history.append(episode_step)
            
            # save checkpoint model, if a good model is received
            # if episode_reward > 130:
            #     save_path = cwd + '\\checkpoint0628_ExpQ2acts\\' + str(num_episode) + '_reward' + str(episode_reward) + 'step' + str(episode_step) + '.ckpt'
            #     saver.save(RL.sess, save_path)
            
            break
        total_step += 1
        epi_action_list.append(selected_action)

# draw cost curve
RL.plot_cost()
RL.plot_reward(reward_history, 250)
RL.plot_epiEvent(good_event_history)

# %% for testing
# restore checkpoint path
# check_pt_path = 'C:\work\DES_RL\AGV_expDQN\checkpoint0628_ExpQ2acts\\96789_reward179.59999999999945step201.ckpt'
check_pt_path = 'C:\\work\\DES_RL\\AGV_expDQN\\results\\96789_reward179.59999999999945step201.ckpt'
# checkpoint0625_2acts\\48649_reward126.39999999999962step201.ckpt
# checkpoint0626_2acts\\74933_reward126.39999999999974step201.ckpt

tf.reset_default_graph()

action_num_list = []
train_num_list = []
train_diff_list = []
for num_episode in range(100):
    S = [0, 0, 0, 0, 0, 0, 0]
    isDone_test = 0
    action_list = []
    [action_list, pattern_list, state_list_his] = RL.check_action(300, S, check_pt_path, AGV_1, AGV_2, AGV_3, AGV_4, AGV_5, SUP_IPSR, SUP_ZWSR, E_c, E_u, B)
    print("the length is", len(action_list), "\n")
    print("the number of 31 is", action_list.count(31), "\n")
    print("the number of difference of the two parts is", np.abs(action_list.count(2) - action_list.count(6)), "\n")
    action_num_list.append(len(action_list))
    train_num_list.append(action_list.count(31))
    train_diff_list.append(np.abs(action_list.count(2) - action_list.count(6)))
    #print(np.array(action_list))



