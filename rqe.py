import numpy as np
from random import randint
import random, sys
import numpy as np
import mod_rqe as mod
import keras

# MACROS
# Gridworld Dimensions
grid_row = 5
grid_col = 5
observability = 1
hidden_nodes = 35
epsilon = 0.5  # Exploration Policy
alpha = 0.5  # Learning rate
gamma = 0.7 # Discount rate
total_steps = 100 #Total roaming steps without goal before termination
num_agents = 3
num_poi = 2
total_train_epoch = 100000
angle_res = 10
online_learning = False
agent_rand = True
poi_rand = True

#ABLATION VARS
use_rnn = False # Use recurrent instead of normal network
success_replay  = True
neat_growth = 2
use_prune = True #Prune duplicates
angled_repr = True


def test_random(gridworld,  illustrate = False):
    rand_suc = 0
    for i in range(1000):
        nn_state, steps, tot_reward = reset_board()
        hist = np.reshape(np.zeros(5 * num_agents), (num_agents, 5))  # Histogram of action taken
        for steps in range(total_steps):  # One training episode till goal is not reached
            for agent_id in range(num_agents):  # 1 turn per agent
                action = randint(0,4)
                hist[agent_id][action] += 1

                # Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)
                if 0: #illustrate:
                    mod.dispGrid(gridworld)
                    #raw_input('Press Enter')
            if gridworld.check_goal_complete():
                #mod.dispGrid(gridworld)
                break
        #print hist
        if steps < gridworld.optimal_steps * 2:
            rand_suc += 1
    return rand_suc/1000

def test_dqn(q_model, gridworld, illustrate = False, total_samples = 10):
    cumul_rew = 0; cumul_coverage = 0
    for sample in range(total_samples):
        nn_state, steps, tot_reward = reset_board()
        #hist = np.reshape(np.zeros(5 * num_agents), (num_agents, 5))  # Histogram of action taken
        for steps in range(total_steps):  # One training episode till goal is not reached
            for agent_id in range(num_agents):  # 1 turn per agent
                q_vals = get_qvalues(nn_state[agent_id], q_model)  # for first step, calculate q_vals here
                action = np.argmax(q_vals)
                if np.amax(q_vals) - np.amin(q_vals) == 0:  # Random if all choices are same
                    action = randint(0, len(q_model) - 1)
                #hist[agent_id][action] += 1

                # Get Reward and move
                reward, _ = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)
                if illustrate:
                    mod.dispGrid(gridworld)
                    print agent_id, action, reward
            if gridworld.check_goal_complete():
                #mod.dispGrid(gridworld)
                break
        cumul_rew += tot_reward/(steps+1); cumul_coverage += sum(gridworld.goal_complete) * 1.0/gridworld.num_poi
    return cumul_rew/total_samples, cumul_coverage/total_samples

def display_q_values(q_model):
    gridworld, agent, nn_state, steps, tot_reward = reset_board(use_rnn)  # Reset board
    for x in range(grid_row):
        for i in range(x):
            _ = mod.move_and_get_reward(gridworld, agent, action=2)
        for y in range(grid_col):
            #mod.dispGrid(gridworld, agent)
            print(get_qvalues(nn_state, q_model))
            _ = mod.move_and_get_reward(gridworld, agent, action=1)
            nn_state = mod.referesh_hist(gridworld, agent, nn_state, use_rnn)
        gridworld, agent, nn_state, steps, tot_reward = reset_board(use_rnn)  # Reset board

def test_nntrain(net, x, y):
    error = []
    for i in range(len(x)):
        input = np.reshape(x[i], (1, len(x[i])))
        error.append((net.predict(input) - y[i])[0][0])
    return error

def reset_board():
    gridworld.reset(agent_rand, poi_rand)
    first_input = []
    for i in range (num_agents):
        first_input.append(gridworld.get_first_state(i, use_rnn))
    return first_input, 0, 0

def get_qvalues(nn_state, q_model):
    values = np.zeros(len(q_model))
    for i in range(len(q_model)):
        values[i] = q_model[i].predict(nn_state)
    return values

def decay(epsilon, alpha):
    if epsilon > 0.1:
        epsilon -= 0.00005
    if alpha > 0.1:
        alpha -= 0.00005
    return epsilon, alpha

def reset_trajectories():
    trajectory_states = []
    trajectory_action = []
    trajectory_reward = []
    trajectory_max_q = []
    trajectory_qval = []
    trajectory_board_pos = []
    return trajectory_states, trajectory_action, trajectory_reward, trajectory_max_q, trajectory_qval, trajectory_board_pos

class statistics():
    def __init__(self):
        self.train_goal_met = 0
        self.reward_matrix = np.zeros(10) + 10000
        self.coverage_matrix = np.zeros(10) + 10000
        self.tr_reward_mat = []
        self.tr_coverage_mat = []
        #self.coverage_matrix = np.zeros(100)
        self.tr_reward = []
        self.tr_coverage = []
        self.tr_reward_mat = []

    def save_csv(self, reward, coverage, train_epoch):
        self.tr_reward.append(reward)
        self.tr_coverage.append(coverage)

        np.savetxt('reward.csv', np.array(self.tr_reward), fmt='%.3f')
        np.savetxt('coverage.csv', np.array(self.tr_coverage), fmt='%.3f')

        self.reward_matrix = np.roll(self.reward_matrix, -1)
        self.reward_matrix[-1] = reward
        self.coverage_matrix = np.roll(self.coverage_matrix, -1)
        self.coverage_matrix[-1] = coverage
        if self.reward_matrix[0] != 10000:
            self.tr_reward_mat.append(np.average(self.reward_matrix))
            np.savetxt('reward_matrix.csv', np.array(self.tr_reward_mat), fmt='%.3f')
            self.tr_coverage_mat.append(np.average(self.coverage_matrix))
            np.savetxt('coverage_matrix.csv', np.array(self.tr_coverage_mat), fmt='%.3f')



if __name__ == "__main__":
    tracker = statistics()
    gridworld = mod.Gridworld(grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand, angled_repr = angled_repr, angle_res = angle_res) #Create gridworld
    mod.dispGrid(gridworld)

    #initialize Q-estimator ensemble
    q_model = []
    if use_rnn:
        for i in range(5):
            q_model.append(mod.init_rnn(gridworld, hidden_nodes, angled_repr, angle_res)) #Recurrent Q-estimator
    else:
        for i in range(5):
            q_model.append(mod.init_nn(gridworld, hidden_nodes, angled_repr, angle_res)) #Normal Q-estimator
    print q_model[0].summary()
    success_traj = [[[],[]]] #Success trajectory initialization for success replay
    for i in range(4):
        success_traj.append([[],[]])

    for train_epoch in range(total_train_epoch): #Training Epochs Main Loop
        if train_epoch == 0: continue
        epsilon, alpha = decay(epsilon, alpha)
        nn_state, steps, tot_reward = reset_board() #Reset board
        trajectory_states, trajectory_action, trajectory_reward, trajectory_max_q, trajectory_qval, trajectory_board_pos = reset_trajectories()  #Reset trajectories
        if neat_growth > 0 and train_epoch % 1000 == 0: #complexity gradient
            grid_row += 1; grid_col += 1; neat_growth -= 1
            gridworld = mod.Gridworld(grid_row, grid_col, observability, num_agents, num_poi, agent_rand, poi_rand, angled_repr=angled_repr, angle_res=angle_res) #Grow Grid
            mod.dispGrid(gridworld)
            print 'GRIDWORLD INCREASED'
            epsilon = 0.5  # Exploration Policy
            alpha = 0.3  # Learning rate
            gamma = 0.3  # Discount rate


        for steps in range(total_steps): #One training episode till goal is not reached
            for agent_id in range(num_agents): #1 turn per agent
                table_pos = [gridworld.agent_pos[agent_id][0] - gridworld.observe,
                             gridworld.agent_pos[agent_id][1] - gridworld.observe]

                prev_nn_state = nn_state[:]  # Backup current state input
                if steps == 0:
                    q_vals = get_qvalues(nn_state[agent_id], q_model) #for first step, calculate q_vals here
                action = np.argmax(q_vals)
                if np.amax(q_vals) - np.amin(q_vals) == 0:  # Random if all choices are same
                    action = randint(0, len(q_model)-1)

                if random.random() < epsilon and steps % 7 != 0: #Random action epsilon greedy step + data sampling
                    action = randint(0,4)
                trajectory_qval.append(q_vals[action]) #Record q_val of the action actually taken

                #Get Reward and move
                reward, assist_agent = gridworld.move_and_get_reward(agent_id, action)
                tot_reward += reward

                # Update current state
                nn_state[agent_id] = gridworld.referesh_state(nn_state[agent_id], agent_id, use_rnn)

                #Save observations to trajectory and get q_values for next state
                trajectory_states.append(prev_nn_state[agent_id])
                trajectory_board_pos.append(table_pos)
                # Get qvalues and maxQ for next iteration
                q_vals = get_qvalues(nn_state[agent_id], q_model)
                max_q = np.amax(q_vals)
                trajectory_max_q.append(max_q)
                trajectory_action.append(action)
                trajectory_reward.append(reward)

                #Credit assignment to assisting agent
                #TODO CREDIT ASSIGNMENT!!!!!!
                if assist_agent != None: #ONE POI OBSERVED
                    index_assist = agent_id - assist_agent
                    if index_assist < 0:
                        index_assist += num_agents
                    #max_r = max(trajectory_reward[len(trajectory_reward) - 1 - num_agents:len(trajectory_reward) - 1])
                    try:
                        if trajectory_reward[len(trajectory_reward) - 1 - index_assist] < reward / 1.0:
                            trajectory_reward[len(trajectory_reward) - 1 - index_assist] = reward / 1.0
                    except:
                        1+1

                #ONLINE LEARNING
                if online_learning:
                    q_update = q_vals[action] + alpha * (reward - q_vals[action] + gamma * max_q)
                    x = np.array(prev_nn_state[agent_id])
                    y = np.reshape(np.array(q_update), (1,1))
                    #print x.shape, y.shape
                    q_model[action].fit(x,y, verbose=0, nb_epoch=1)


            if gridworld.check_goal_complete():
                #print 'GOAL MET'
                tracker.train_goal_met += 1
                break
        #END OF ONE ROUND OF SIMULATION
        #print 'Soft progress: ', sum(gridworld.goal_complete), 'out of ', num_poi

        if not online_learning:
            #Trajectory replay back_propagation to train
            all_q_updates = np.zeros(len(trajectory_states))  # All updates
            eff_alpha = alpha
            eff_gamma = gamma
            if (steps + 1) < total_steps: #Adjust effective learning rate and gamma for successful finds
                eff_alpha = alpha + 0.3; eff_gamma = gamma + 0.3
                if eff_alpha > 1.0: eff_alpha = 0.99
                if eff_gamma > 1.0: eff_gamma = 0.99

            for example in range(len(trajectory_states)): #Compute updated Q-Values
                index = len(trajectory_states) - example - 1  # Index from back
                real_max_q = trajectory_max_q[index]
                if num_agents <= example:  # Except last states
                    ig1 = all_q_updates[index + num_agents]
                    if all_q_updates[index + num_agents] > real_max_q:
                        real_max_q = all_q_updates[index + num_agents]  # Update max_Q as a result of new findings
                all_q_updates[index] = trajectory_qval[index] + eff_alpha * (trajectory_reward[index] - trajectory_qval[index] + eff_gamma * real_max_q)


            #Bin updates by action
            q_updates = [[], [], [], [], []]  # Q_value updates for each action
            update_states = [[], [], [], [], []] #Log states to update
            board_posit = [[], [], [], [], []]
            for i in range(len(all_q_updates)):
                q_updates[trajectory_action[i]].append(all_q_updates[i])
                update_states[trajectory_action[i]].append(trajectory_states[i][0])
                board_posit[trajectory_action[i]].append(trajectory_board_pos[i])

            #Train the network
            for i in range(len(q_model)):
                if len(q_updates[i]) < 1: #If update list is empty
                    continue
                if use_prune:
                    # #Prune update list by taking out duplicates
                    indice_del = []
                    for j in range(len(q_updates[i])):
                        index = len(q_updates[i]) - j -1
                        for k in range(index):
                            #if sum(abs(update_states[i][index] - update_states[i][k])) == 0:
                            if board_posit[i][index][0] == board_posit[i][k][0] and board_posit[i][index][1] == board_posit[i][k][1]:
                                indice_del.append(k)
                    indice_del = list(set(indice_del))
                    for ind in sorted(indice_del, reverse=True): #Delete dulicates
                        del q_updates[i][ind]
                        del update_states[i][ind]
                y = np.array(q_updates[i])
                x = np.array(update_states[i])
                if gridworld.check_goal_complete():
                    if len(success_traj[i][0]) == 0:
                        success_traj[i][0] = x; success_traj[i][1] = y
                    else:
                        success_traj[i][1] = np.concatenate((success_traj[i][1], y), axis=0)
                        success_traj[i][0] = np.concatenate((success_traj[i][0], x), axis=0)
                #earlyStopping = keras.callbacks.EarlyStopping(monitor='train_loss', patience=5, verbose=0, mode='min')
                #er1 = test_nntrain(q_model[i], x, y)
                q_model[i].fit(x, y, verbose=0, nb_epoch=1)#, callbacks=[earlyStopping])
                    #er2 = test_nntrain(q_model[i], x, y)
                    #print np.sum(abs(np.array(er1) - np.array(er2)))

        #Trajectory Replay for successful runs
        if train_epoch % 500 == 0 and success_replay  == True:
            for i in range(5):
                if len(success_traj[i][0]) > 1000: #Limit size of trajectory storage
                    cut = len(success_traj[i][0]) - 1000
                    success_traj[i][0] = success_traj[i][0][cut:]
                    success_traj[i][1] = success_traj[i][1][cut:]

                if len(success_traj[i][0]) != 0:
                    y = success_traj[i][1]
                    x = success_traj[i][0]
                    q_model[i].fit(x, y, verbose=0, nb_epoch=1)

        #Save RQE net
        if train_epoch % 5000 == 0:  # Save Qmodel
            for i in range(len(q_model)):
                q_model[i].save_weights('Models/model_weights_' + str(i) + '.h5', overwrite=True)

        #UI and Statisctics
        if train_epoch % 100 == 0:
            tot_reward, coverage = test_dqn(q_model, gridworld)
            tracker.save_csv(tot_reward, coverage, train_epoch)




            # tracker.reward_matrix = np.roll(tracker.reward_matrix, -1); tracker.success_matrix = np.roll(tracker.success_matrix, -1)
            # #if steps <= gridworld.optimal_steps*1.5: success_matrix[-1] = 1
            # if steps < total_steps: success_matrix[-1] = 1
            # else: success_matrix[-1] = 0
            # reward_matrix[-1] = tot_reward
            #
            # tr_reward = np.append(tr_reward, np.average(reward_matrix))
            # np.savetxt('reward.csv', tr_reward, fmt='%.3f')
            # tr_success = np.append(tr_success, np.sum(success_matrix))
            # np.savetxt('hist.csv', tr_success, fmt='%.3f')
            # if steps <= gridworld.optimal_steps / 2: category = 'Quasi-Optimal'
            # elif steps < gridworld.optimal_steps and steps < total_steps: category = 'Sub-Optimal'
            # else: category = 'Failure'

            print 'Epochs:', tracker.train_goal_met, 'met of', train_epoch, 'Epsilon', epsilon, ' Alpha:', alpha, 'Gridsize', grid_col, 'Coverage:', coverage, '  Reward: ', tot_reward









