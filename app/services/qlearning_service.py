import numpy as np
import random
import os
from dotenv import load_dotenv

load_dotenv()

class QLearning:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

        self.q_table = np.zeros((states, actions))

        # ambil dari .env
        self.alpha = float(os.getenv("ALPHA", 0.1))
        self.gamma = float(os.getenv("GAMMA", 0.9))
        self.epsilon = float(os.getenv("EPSILON", 0.1))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.actions - 1)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (target - predict)