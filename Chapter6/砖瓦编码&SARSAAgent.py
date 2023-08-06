import gym 
import numpy as np
from matplotlib import pyplot as plt

env=gym.make('MountainCar-v0')

class TileCoder:
    def __init__(self, layers, features):
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:
            return hash(codeword)
        else:
            self.codebook[codeword]=count
            return count

    def __call__(self,floats=(),ints=()):
        dim = len(floats)
        scaled_floats = tuple(f*self.layers*self.layers for f in floats)
        features=[]
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) / self.layers) for i,f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features

class SARSAAgent:
    def __init__(self,env,features=1893,layers=8,gamma=1.,learning_rate=0.03,epsilon=0.001):
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.epsilon=epsilon

        self.action_n=env.action_space.n
        self.obs_low=env.observation_space.low
        self.obs_scal=env.observation_space.high-env.observation_space.low

        self.encoder=TileCoder(layers,features)
        self.w=np.zeros(features)
    
    def encode(self,observation,action):
        states=tuple((observation-self.obs_low)/self.obs_scal)
        actions=(action,)
        return self.encoder(states,actions)

    def get_q(self,observation,action):
        features=self.encode(observation,action)
        return self.w[features].sum()

    def decide(self,observation):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs=[self.get_q(observation,action) for action in range(self.action_n)]
            return np.argmax(qs)
    
    def learn(self,observation,action,reward,done,next_observation,next_action):
        u=reward+self.gamma*self.get_q(next_observation,next_action)*(1.-done)
        td_error = u - self.get_q(observation,action)
        features=self.encode(observation,action)
        self.w[features] += self.learning_rate*td_error

def play_SRASA(env,agent,learn=False,render=False):
    total_reward=0
    observation=env.reset()
    action=agent.decide(observation)
    while 1:
        if render:
            env.render()
        next_observation,reward,done,info=env.step(action)
        total_reward += reward
        next_action=agent.decide(next_observation)
        if learn:
            agent.learn(observation,action,reward,done,next_observation,next_action)
        if done:
            break
        observation,action=next_observation,next_action
    return total_reward

#env=gym.make('MountainCar-v0')
agent=SARSAAgent(env)
episodes=5000
episode_rewards=[]
for episode in range(episodes):
    episode_reward=play_SRASA(env,agent,learn=True)
    episode_rewards.append(episode_reward)
env.close()
plt.plot(episode_rewards)
plt.show()