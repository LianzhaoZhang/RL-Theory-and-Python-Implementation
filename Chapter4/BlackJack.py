import gym
import numpy as np
from matplotlib import pyplot as plt


env=gym.make('Blackjack-v1')

def ob2state(observation):
    return int(observation[0]) , int(observation[1]) , int(observation[2])

def evaluate_action_Mont_carlo(env,policy,episode_num=10000):
    q=np.zeros_like(policy)
    c=np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions=[]#储存状态动作轨迹
        observation=env.reset()
        while 1:#获得状态动作轨迹
            state=ob2state(observation)
            action=np.random.choice(env.action_space.n,p=policy[state])
            state_actions.append((state,action))
            observation,reward,done,info=env.step(action)
            if done :
                break
        g=reward
        for state , action in state_actions:
            c[state][action] +=1
            q[state][action] += (g-q[state][action])/c[state][action]
    return q

def plot(data):
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    titles=['withour ace','with ace']
    have_aces=[0,1]
    extent=[12,22,1,11]
    for title,have_ace,axis in zip(titles,have_aces,axes):
        dat=data[extent[0]:extent[1],extent[2]:extent[3],have_ace].T
        axis.imshow(dat,extent=extent,origin='lower')
        axis.set_xlabel('palyer_sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()

def MontCarlo_with_soft(env,episode_num=100000,epsilon=0.1):
    policy=np.ones((22,11,2,2))*0.5
    q=np.zeros_like(policy)
    c=np.zeros_like(policy)
    for _ in range(episode_num):
        state_actions=[]
        observation=env.reset()
        while 1:
            state=ob2state(observation)
            action=np.random.choice(env.action_space.n,p=policy[state])
            state_actions.append((state,action))
            observation,reward,done,info=env.step(action)
            if done :
                break
        g=reward
        for state,action in state_actions:
            c[state][action] += 1
            q[state][action] += (g-c[state][action])/c[state][action]
            a=q[state].argmax()
            policy[state]=epsilon/2.
            policy[state][a] += (1.-epsilon)
    return policy,q
                    
policy,q=MontCarlo_with_soft(env)
v=q.max(axis=-1)
plot(v)                    

'''
policy=np.zeros(shape=(22,11,2,2))
policy[20:,:,:,0]=1  #>20时不要牌
policy[:20,:,:,1]=1  #<20时要牌
q=evaluate_action_Mont_carlo(env,policy)
v=(q*policy).sum(axis=-1)
plot(v)
'''