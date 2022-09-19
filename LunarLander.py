from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
import random
import gym
from gym.wrappers.monitoring import video_recorder
import numpy as np


class DQLAgent:
    def __init__(self, env):
        # hyperparameter / parameter
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)

        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(58, input_dim=self.state_size, activation="relu"))
        model.add(Dense(58, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        #model.load_weights("models/LunarLander-v1.h5")

        return model

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if(random.uniform(0,1) <= self.epsilon):
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        "vectorized replay method"
        if (len(agent.memory) < batch_size):
            return

        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        if (len(not_done_indices[0]) > 0):
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 0]), verbose=0)
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]), verbose=0)

            y[not_done_indices] += np.multiply(self.gamma, predict_sprime_target[
                not_done_indices, np.argmax(predict_sprime[not_done_indices, :][0], axis=1)][0])

        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]), verbose=0)
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)

    """       
    def replay(self,batch_size):
        #training
        if(len(self.memory) < batch_size):
            return
        minibatch = random.sample(self.memory, batch_size)
        for state,action,reward,next_state,done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state,verbose = 0)[0])
            train_target = self.model.predict(state, verbose = 0)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose = 0)
            """

    def adaptiveEGreedy(self):
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay

    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())


if __name__=="__main__":

        env = gym.make("LunarLander-v2")
        vid = video_recorder.VideoRecorder(env=env,path="video/LunarLander_vid.mp4")
        agent = DQLAgent(env)
        batch_size = 16
        episodes = 500
        old_total_reward = -1000000
        for i in range(episodes):
            #initialize environment
            state = env.reset()
            state = np.reshape(state,[1,8])

            time = 0
            total_reward = 0

            while True:
                vid.capture_frame()
                #env.render()
                #act
                action = agent.act(state) # select an action

                #step
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state,[1,8])

                #remember / storage
                agent.remember(state, action, reward, next_state, done)

                #update state
                state = next_state

                #replay
                agent.replay(batch_size)

                #adjust epsilon


                total_reward += reward
                time += 1

                if done:
                    agent.targetModelUpdate()
                    print("Episode: {}, time: {}, Reward: {}".format(i, time, total_reward))

                    if(total_reward > old_total_reward):
                        old_total_reward = total_reward
                        agent.model.save_weights("models/LunarLander-v1.h5", overwrite=True)
                        print("Saved Model")
                    break

            agent.adaptiveEGreedy()



