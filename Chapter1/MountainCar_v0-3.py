import numpy as np
import gym

class BespokerAgent():
    def __init__(self,env):
        pass
    def decide(self,observation):
        position,volicity=observation
        lb=min( -0.09* (position+ 0.25)** 2+ 0.03,
                        0.3* (position + 0.9) ** 4 - 0.008)
        ub=-0.07*(position+0.38)**2+0.06
        if lb<volicity<ub:
            action=2
        else:
            action=0
        return action

    def learn(self,*args):
        pass

def Montecarlor(env,agent,render=False,train=False):
    ep_reward=0
    observation=env.reset()
    
    while 1:
        if render:
            env.render()
        action=agent.decide(observation)
        netx_observation,reward,done,_=env.step(action)
        ep_reward += reward

        if train:
            agent(observation,action,reward,done)

        if done:
            break
        observation=netx_observation

    return ep_reward


env=gym.make('MountainCar-v0')
agent=BespokerAgent(env)

env.seed(0)

reward=[Montecarlor(env,agent,False,False) for _ in range(100)]
print(np.mean(reward))
'''
reward=Montecarlor(env,agent,True,False)
print(reward)
'''