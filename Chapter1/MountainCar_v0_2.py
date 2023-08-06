import gym
import numpy as np

class BespokerAgent():
    def __init__(self,env):
        pass

    def decide(self,observation):
        position,volicity=observation
        lb=min( -0.09* (position+ 0.25)** 2+ 0.03,
                        0.3* (position + 0.9) ** 4 - 0.008)
        ub = -0.07*(position+0.38)**2+0.06
        if lb<volicity<ub:
            action=2
        else:
            action=0
        return action
        
    def learn(self, *args):
        pass

def Montecarlo(env,agent,render=True,train=False):
    ep_reward=0
    observation=env.reset()
    while 1:
        if render:
            env.render()

        action=agent.decide(observation)
        next_observation,reward,done,_=env.step(action)
        ep_reward +=reward
        
        if train:
            agent.learn(observation,action,reward,done)

        if done:
            break

        observation=next_observation

    return ep_reward




env=gym.make('MountainCar-v0')
agent=BespokerAgent(env)
env.seed(0)
#运行1次
rew=Montecarlo(env, agent, render=True,train=False)
print(rew)

# 运行100次
#rew=[Montecarlo(env,agent,False,False) for _ in range(100)]
#print('100步的平均回合奖励为：{}'.format(np.mean(rew)))
env.close()