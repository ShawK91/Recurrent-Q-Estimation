import numpy as np
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
import math, sys
import json
import numpy as np
import random


def save_model(model):
    json_string = model.to_json()
    open('model_architecture.json', 'w').write(json_string)
    model.save_weights('shaw_2/model_weights.h5',overwrite=True)

class Gridworld:
    def __init__(self, dim_row = 10, dim_column = 10, observe = 2, num_agents = 2, num_poi = 2, agent_rand = False, poi_rand = False, angle_res = 10, angled_repr = False, obs_dist = 2):
        self.observe = observe
        self.dim_row = dim_row
        self.dim_col = dim_column
        self.num_agents = num_agents
        self.angle_res = angle_res #Angle resolution
        self.angled_repr = angled_repr #Angled state representation
        if num_agents > 1:
            self.coupling = 2 #coupling requirement
        else:
            self.coupling = 1
        self.obs_dist = obs_dist
        self.num_poi = num_poi
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()
        self.poi_pos = []
        self.goal_complete = []
        for i in range(self.num_poi):
            self.init_poi(poi_rand) #Initialize POIs
            self.goal_complete.append(False)
        self.agent_pos = []
        for i in range(self.num_agents):
            self.init_agent(agent_rand)

        self.optimal_steps = self.get_optimal_steps()


    def init_wall(self):
        for i in range(self.observe):
            for x in range(self.state.shape[0]):
                self.state[x][i] = 3
                self.state[x][self.state.shape[1] - 1-i] = 3
            for y in range(self.state.shape[1]):
                self.state[i][y] = 3
                self.state[self.state.shape[0] - 1-i][y] = 3

    def check_spawn(self, x, y):
        for i in range(len(self.poi_pos)):
            if x == self.poi_pos[i][0] and y == self.poi_pos[i][1]:  # Check to see for other POI
                return False
        try:
            for i in range(len(self.agent_pos)):
                if x == self.agent_pos[i][0] and y == self.agent_pos[i][1]:  # Check to see for other agent
                    return False
        except:
            1 + 1
        return True

    def init_poi(self, rand_start):
        start = self.observe
        end = self.state.shape[0] - self.observe-1
        rad = int(self.dim_row/math.sqrt(3)/2)
        center = int((start + end)/2)
        if rand_start:
            while True:
                rand = random.random()
                if rand < 0.25:
                    x = randint(start, center - rad - 1)
                    y = randint(start, end)
                elif rand < 0.5:
                    x = randint(center + rad + 1, end)
                    y = randint(start, end)
                elif rand < 0.75:
                    x = randint(center - rad, center + rad)
                    y = randint(start, center - rad - 1)
                else:
                    x = randint(center - rad, center + rad)
                    y = randint(center + rad + 1, end)
                if self.check_spawn(x, y):
                    break

        else:
            #x = self.state.shape[0] - self.observe-1 ;y = self.state.shape[1] - self.observe-1 #Goals at ends
            x = self.state.shape[0]/2 + 1 ;y = self.state.shape[1]/2 #Goals in middle
            if len(self.poi_pos) > 0:
                for i in range(len(self.poi_pos)):
                    if x == self.poi_pos[i][0] and y == self.poi_pos[i][1]:  # Check to see for other agent
                        y = self.state.shape[0] - self.observe - 2
                        x = self.observe + 1
        self.state[x][y] = 2
        self.poi_pos.append([x, y])
        return [x,y]


    def init_agent(self, rand_start):
        start = self.observe
        end = self.state.shape[0] - self.observe-1
        rad = int(self.dim_row/math.sqrt(3)/2)
        center = int((start + end)/2)
        if rand_start:
            while True:
                x = randint(center - rad, center + rad)
                y = randint(center - rad, center + rad)
                if self.check_spawn(x, y):
                    break
        else: #Not random
            while True:
                if len(self.agent_pos) == 0:
                    x = start; y = start
                if len(self.agent_pos) == 1:
                    x = start + (end - start) / 2; y = start
                if len(self.agent_pos) == 2:
                    x = end; y = start
                if len(self.agent_pos) > 2:
                    x = start + len(self.agent_pos); y = start
                for i in range(len(self.agent_pos)):
                    if x == self.agent_pos[i][0] and y == self.agent_pos[i][1]:  # Check to see for other POIs
                        continue
                break
        self.state[x][y] = 1 #Agent Code
        self.agent_pos.append([x,y])
        return [x,y]


    def bck_init_poi(self, rand_start):
        start = self.observe
        end = self.state.shape[0] - self.observe-1

        if rand_start:
            while True:
                x = randint(start, end)
                y = randint(start + (end-start)/2, end)
                if self.check_spawn(x, y):
                    break

        else:
            #x = self.state.shape[0] - self.observe-1 ;y = self.state.shape[1] - self.observe-1 #Goals at ends
            x = self.state.shape[0]/2 + 1 ;y = self.state.shape[1]/2 #Goals in middle
            if len(self.poi_pos) > 0:
                for i in range(len(self.poi_pos)):
                    if x == self.poi_pos[i][0] and y == self.poi_pos[i][1]:  # Check to see for other agent
                        y = self.state.shape[0] - self.observe - 2
                        x = self.observe + 1
        self.state[x][y] = 2
        self.poi_pos.append([x, y])
        return [x,y]

    def bck_init_agent(self, rand_start):
        start = self.observe
        end = self.state.shape[0] - self.observe-1
        if rand_start:
            while True:
                x = randint(start, end)
                y = randint(start, (end - start) / 2)
                if self.check_spawn(x, y):
                    break
        else: #Not random
            while True:
                if len(self.agent_pos) == 0:
                    x = start; y = start
                if len(self.agent_pos) == 1:
                    x = start + (end - start) / 2; y = start
                if len(self.agent_pos) == 2:
                    x = end; y = start
                if len(self.agent_pos) > 2:
                    x = start + len(self.agent_pos); y = start
                for i in range(len(self.agent_pos)):
                    if x == self.agent_pos[i][0] and y == self.agent_pos[i][1]:  # Check to see for other POIs
                        continue
                break
        self.state[x][y] = 1 #Agent Code
        self.agent_pos.append([x,y])
        return [x,y]

    def reset(self, agent_rand, poi_rand):
        self.state = np.zeros((self.dim_row + self.observe*2, self.dim_col + self.observe*2)) #EMPTY SPACE = 0, AGENT = 1, #POI = 2, WALL = 3
        self.init_wall()
        self.poi_pos = []
        for i in range(self.num_poi):
            self.init_poi(poi_rand)  # Initialize POIs
            self.goal_complete[i] = False
        self.agent_pos = []
        for i in range(self.num_agents):
            self.init_agent(agent_rand)
        self.optimal_steps = self.get_optimal_steps()

    def move_and_get_reward(self, agent_id, action):
        next_pos = np.copy(self.agent_pos[agent_id])
        if action == 1:
            next_pos[1] += 1  # Right
        elif action == 2:
            next_pos[0] += 1  # Down
        elif action == 3:
            next_pos[1] -= 1  # Left
        elif action == 4:
            next_pos[0] -= 1  # Up

        # Computer reward and check illegal moves
        reward = 0 #If nothing else
        x = next_pos[0]
        y = next_pos[1]
        if self.state[x][y] == 3:  # Wall
            next_pos[0] = self.agent_pos[agent_id][0]
            reward = -0.0001
            next_pos[1] = self.agent_pos[agent_id][1]
        if self.state[x][y] == 1 and action != 0:  # Other Agent
            reward = -0.05
            next_pos[0] = self.agent_pos[agent_id][0]
            next_pos[1] = self.agent_pos[agent_id][1]

        if self.state[x][y] == 0 or (self.state[x][y] == 1 and action == 0):  # Free Space
            reward = -0.0001

        for poi_id in range(self.num_poi): # POI COUPLED
            goal_agents = 0
            assist_agent = None
            for ag in range(self.num_agents):
                if ag == agent_id:
                    #if self.poi_pos[poi_id][0] == x and self.poi_pos[poi_id][1] == y and self.goal_complete[poi_id] == False:
                    if abs(self.poi_pos[poi_id][0]-x) <= self.obs_dist and abs(self.poi_pos[poi_id][1] - y) <= self.obs_dist and self.goal_complete[poi_id] == False:
                        goal_agents += 1
                else:
                    #if self.poi_pos[poi_id][0] == self.agent_pos[ag][0] and self.poi_pos[poi_id][1] == self.agent_pos[ag][1]  and self.goal_complete[poi_id] == False:
                    if abs(self.poi_pos[poi_id][0] - self.agent_pos[ag][0]) <= self.obs_dist and abs(self.poi_pos[poi_id][1] - self.agent_pos[ag][1]) <= self.obs_dist and self.goal_complete[poi_id] == False:
                        assist_agent = ag
                        goal_agents += 1
            if goal_agents >= self.coupling:
                self.goal_complete[poi_id] = True
                reward += 1

        # Update gridworld and agent position
        if self.state[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] != 2:
            self.state[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] = 0
        if self.state[next_pos[0]][next_pos[1]]!= 2:
            self.state[next_pos[0]][next_pos[1]] = 1
        self.agent_pos[agent_id][0] = next_pos[0]
        self.agent_pos[agent_id][1] = next_pos[1]

        if goal_agents >= self.coupling:
            return reward, assist_agent
        else:
            return reward, None

    def get_state(self, agent_id):  # Returns a flattened array around the agent position
        if self.angled_repr: #If state representation uses angle
            st = self.angled_state(agent_id)
            return st

        x_beg = self.agent_pos[agent_id][0] - self.observe
        y_beg = self.agent_pos[agent_id][1] - self.observe
        x_end = self.agent_pos[agent_id][0] + self.observe + 1
        y_end = self.agent_pos[agent_id][1] + self.observe + 1
        st = np.copy(self.state)
        st = st[x_beg:x_end, :]
        st = st[:, y_beg:y_end]
        st = np.reshape(st, (1, pow(self.observe * 2 + 1, 2))) # Flatten array
        #return st
        k = np.reshape(np.zeros(len(st[0]) * 4), (len(st[0]), 4)) #4-bit encoding
        for i in range(len(st[0])):
            k[i][int(st[0][i])] = 1
        k = np.reshape(k, (1, len(st[0]) * 4))  # Flatten array
        return k

    def get_first_state(self, agent_id, use_rnn):  # Get first state, action input to the q_net
        if not use_rnn: #Normal NN
            st = self.get_state(agent_id)
            return st

        rnn_state = []
        st = self.get_state(agent_id)
        for time in range(3):
            rnn_state.append(st)
        rnn_state = np.array(rnn_state)
        rnn_state = np.reshape(rnn_state, (1, rnn_state.shape[0], rnn_state.shape[2]))
        return rnn_state

    def referesh_state(self, current_state, agent_id, use_rnn):
        st = self.get_state(agent_id)
        if use_rnn:
            new_state = np.roll(current_state, -1, axis=1)
            new_state[0][2] = st
            return new_state
        else:
            return st

    def check_goal_complete(self):
        goal_complete = True #Check if all agents found POI's
        for poi_id in range(self.num_poi):
            goal_complete *= self.goal_complete[poi_id]
        return  goal_complete

    def get_optimal_steps(self):
        #TODO MAKE MATCHING GENERALIZABLE
        return self.dim_col + self.dim_row
        steps = []
        for i in range(self.num_poi):
            ig = []
            for agent_id in range(self.num_agents):
                ig.append(abs(self.poi_pos[i][0] - self.agent_pos[agent_id][0]) + abs(self.poi_pos[i][1] - self.agent_pos[agent_id][1]))
            steps.append(ig)
        opt_steps = 10000000000
        for i in range(self.num_poi):
            for j in range(self.num_poi):
                step = steps[j]

                # if abs(self.poi_pos[i][0] - self.agent_pos[agent_id][0]) + abs(self.poi_pos[i][1] - self.agent_pos[agent_id][1])  > opt_steps:
                #     opt_steps = abs(self.poi_pos[0] - self.agent_pos[agent_id][0]) + abs(self.poi_pos[1] - self.agent_pos[agent_id][1])
        return opt_steps

    def get_angle_dist(self, x1, y1, x2, y2):  # Computes angles and distance between two agents relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  # dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle)
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist

    def angled_state(self, agent_id):
        state = np.zeros(((360/self.angle_res), 4))
        for id in range(self.num_poi):
            if self.goal_complete[id] == False: #For all POI's that are still active
                x1 = self.poi_pos[id][0] - self.agent_pos[agent_id][0]; x2 = 1
                y1 = self.poi_pos[id][1] - self.agent_pos[agent_id][1]; y2 = 0
                angle, dist = self.get_angle_dist(x1,y1,x2,y2)
                bracket = int(angle / self.angle_res)
                state[bracket][0] += 1 #Add POIs
                if state[bracket][1] > dist or state[bracket][1] == 0:  # Update min distance from POI
                    state[bracket][1] = dist

        for id in range(self.num_agents):
            if id != agent_id: #FOR ALL AGENTS MINUS MYSELF
                x1 = self.agent_pos[id][0] - self.agent_pos[agent_id][0]; x2 = 1
                y1 = self.agent_pos[id][1] - self.agent_pos[agent_id][1]; y2 = 0
                angle, dist = self.get_angle_dist(x1,y1,x2,y2)
                bracket = int(angle / self.angle_res)
                state[bracket][2] += 1 #Add agent
                if state[bracket][3] > dist or state[bracket][3] == 0: #Update min distance from other agent
                    state[bracket][3] = dist




        state = np.reshape(state, (1, 360/self.angle_res * 4)) #Flatten array
        return state







class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self



def init_rnn(gridworld, hidden_nodes, angled_repr, angle_res, hist_len = 3, design = 1):
    ### BUILD THE MODEL ###
    model = Sequential()
    if angled_repr:
        sa_sp = (360/angle_res) * 4
    else:
        sa_sp = (pow(gridworld.observe * 2 + 1,2)*4) #BIT ENCODING?????
    if design == 1:
        model.add(LSTM(hidden_nodes, init= 'zero', return_sequences=False, input_shape=(hist_len, sa_sp)))
    elif design == 2:
        model.add(SimpleRNN(hidden_nodes, init='zero', input_shape=(hist_len, sa_sp), inner_init='orthogonal'))
    elif design == 3:
        model.add(GRU(hidden_nodes, init='zero',  input_shape=(hist_len, sa_sp),inner_init='orthogonal'))
    #model.add(Dropout(0.1))
    #model.add(LeakyReLU(alpha=.2))
    #model.add(Activation('sigmoid'))
    #model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dense(1, init= 'zero'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def init_nn(gridworld, hidden_nodes, angled_repr, angle_res):
    model = Sequential()
    if angled_repr:
        sa_sp = (360/angle_res) * 4
    else:
        sa_sp = (pow(gridworld.observe * 2 + 1,2)) * 4
    model.add(Dense(hidden_nodes, input_dim=sa_sp, init='zero'))
    #model.add(LeakyReLU(alpha=.2))
    #model.add(Dropout(0.1))
    model.add(Activation('sigmoid'))
    model.add(Dense(1, init= 'zero'))
    model.compile(loss='mse', optimizer='adam')
    return model


def q_values(hist_input, q_model):
    # action_ind = hist_input.shape[2] - 1
    # for i in range(5):
    #     k[i][2][action_ind] = i
    return q_model.predict(hist_input)

def test_hist(q_model):
    gridworld = Gridworld()
    agent2 = Agent(gridworld, True)
    hist_input = get_first_hist(gridworld, agent2)
    return q_model.predict(hist_input)

def dispGrid(gridworld, state = None, full=True, agent_id = None):

    if state == None: #Given agentq
        if full:
            st = np.copy(gridworld.state)
        else:
            x_beg = gridworld.agent_pos[agent_id][0] - gridworld.observe
            y_beg = gridworld.agent_pos[agent_id][1] - gridworld.observe
            x_end = gridworld.agent_pos[agent_id][0] + gridworld.observe + 1
            y_end = gridworld.agent_pos[agent_id][1] + gridworld.observe + 1
            st = np.copy(gridworld.state)
            st = st[x_beg:x_end,:]
            st = st[:,y_beg:y_end]
    else:
        st = []
        print len(state)
        row_leng = int(math.sqrt(len(state)))
        for i in range(row_leng):
            ig = []
            for j in range(row_leng):
                ig.append(state[i*row_leng + j])
            st.append(ig)

    grid = [["-" for i in range(len(st))] for i in range(len(st))]
    grid[0][0] = "o"
    for i in range(len(st)):
        for j in range(len(st)):
            if st[i][j] == 2:
                grid[i][j] = '$'
            if st[i][j] == 1:
                grid[i][j] = '*'
            if st[i][j] == 3:
                grid[i][j] = '#'
    for row in grid:
        for e in row:
            print e,
        print

def test_q_table(q_table, gridworld, agent1):
    steps = 0
    tot_reward = 0
    while True:  # Till goal is not reached
        steps += 1
        table_pos = [agent1.position[0] - gridworld.observe,
                     agent1.position[1] - gridworld.observe]  # Transform to work q-table
        action = np.argmax(q_table[table_pos[0]][[table_pos[1]]])
        # Get Reward and move
        reward = move_and_get_reward(gridworld, agent1, action)
        tot_reward += reward

        if gridworld.poi_pos[0] == agent1.position[0] and gridworld.poi_pos[1] == agent1.position[1]:  # IF POI is met
            break
        if steps > 100:
            break
    return tot_reward, steps




