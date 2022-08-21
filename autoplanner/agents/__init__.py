import numpy as np
from math import ceil, floor
from collections import deque
import random

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam

class Agent:
    def __init__(self, env, memory_size=50000, min_memory_size=1000, batch_size=64, target_update_period=1, discount=0.99):
        self.env = env
        self.action_space_size = 8
        self.batch_size = batch_size
        self.target_update_period = target_update_period
        self.discount = discount

        self.model = self.createModel()

        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

        self.replay_memory = deque(maxlen=memory_size)
        self.min_memory_size = min_memory_size

        
    def createModel(self):
        model = Sequential()
        model.add(Conv2D(258, (3,3), input_shape=(self.env.map.lines, self.env.map.columns, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(.2))

        model.add(Conv2D(258, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(self.action_space_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=.001), metrics=["accuracy"])

        return model

    def updateReplayMemory(self, transition):
        self.replay_memory.append(transition)

    def getQs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/11, verbose=False)[0]

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.min_memory_size:
            return
        
        batch = random.sample(self.replay_memory, self.batch_size)

        current_states = np.array([transition[0] for transition in batch])/11
        current_qs_list = self.model.predict(current_states, verbose=False)

        #new_current_states = np.array([transition[3] for transition in batch])/11
        #future_qs_list = self.target_model.predict(new_current_states, verbose=False)
        future_qs_list = np.array([transition[5] for transition in batch])

        X, y = [], []
        for index, (current_state, action, reward, new_current_state, done, _) in enumerate(batch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/11, np.array(y), batch_size=self.batch_size, verbose=False, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.target_update_period:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def build(self, dev_type, map):
        if self.cell.type == 0:
            return
        self.cell.setDeveloped(dev_type=dev_type, map=map)

    def destroy(self):
        if self.cell.type == 0:
            return
        self.cell.setUndeveloped()

    def prospectReward(self, action):
        if (action in [0, 1, 2, 3]): # movement actions do not change the reward
            return self.env.reward()    
        prev_type = self.cell.type
        self.playAction(action)
        reward = self.env.reward()

        if (prev_type == -1):
            self.destroy()
        else:
            self.build(prev_type, self.env.map)

        return reward

    # Changes the agent current cell
    def moveTo(self, cell):
        self.cell = cell
        self.x, self.y = cell.idx[0], cell.idx[1]
        return

    # Execute selected action
    def playAction(self, action):
        world = self.env

        if(action==0): # Right
            next_cell = world.map.cells[self.x, (self.y+1)%world.map.columns]
            self.moveTo(next_cell)
        elif(action==1): # Left
            next_cell = world.map.cells[self.x, self.y-1]
            self.moveTo(next_cell)
        elif(action==2): # Down
            next_cell = world.map.cells[self.x-1, self.y]
            self.moveTo(next_cell)
        elif(action==3): # Up
            next_cell = world.map.cells[(self.x+1)%world.map.lines, self.y]
            self.moveTo(next_cell)
        elif(action==4): # Build residential
            self.build(dev_type=1, map=world.map)
        elif(action==5): # Build commercial
            self.build(dev_type=2, map=world.map)
        elif(action==6): # Build industrial
            self.build(dev_type=3, map=world.map)
        elif(action==7): # Build recreational
            self.build(dev_type=4, map=world.map)

    # Execute explore-build behavior
    def interact(self, epsilon):
        # Choosing next action 
        #for i in range(8):
        #    print(f"For action {i} reward is {self.prospectReward(i)}")

        if np.random.random() > epsilon:
            # Optimal action by Q table
            action = np.argmax(self.getQs(self.env.getState()))
        else:
            # Random action
            action = np.random.randint(0, self.action_space_size)

        self.playAction(action)
        return action