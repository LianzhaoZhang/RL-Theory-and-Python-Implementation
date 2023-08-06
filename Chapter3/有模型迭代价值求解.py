import gym
import numpy as np
from sympy import re, true



def v2q(env,v,s=None,gamma=1):
    if s is not None:
        q=np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob,next_state,reward,done in env.P[s][a]:
                q[a] += prob*(reward+gamma*v[next_state]*(1.-done))
    else:
        q=np.zeros((env.observation_space.n,env.action_space.n))
        for s in range(env.observation_space.n):
            q[s]=v2q(env,v,s,gamma)
    return q

def play_policy(env,policy,render=True):
    totla_reward=0
    observation=env.reset()
    while 1:
        if render:
            env.render()
        action=np.random.choice(env.action_space.n,p=policy[observation])
        observation,reward,done,info=env.step(action)
        totla_reward += reward
        if done:
            break
    return totla_reward

def iterate_value(env,gamma=1,tolerant=1e-6):
    v=np.zeros(env.observation_space.n)
    while true:
        delta=0
        for s in range(env.observation_space.n):
            vmax=max(v2q(env,v,s,gamma))
            delta=max(delta,abs(v[s]-vmax))
            v[s]=vmax
        if delta<tolerant:
            break
    policy=np.zeros((env.observation_space.n,env.action_space.n))
    for s in range(env.observation_space.n):
        a=np.argmax(v2q(env,v,s,gamma))
        policy[s][a]=1.
    return policy , v

env=gym.make('FrozenLake-v1')
policy_vi,v_vi=iterate_value(env)
print('状态价值函数={}'.format(v_vi.reshape(4,4)))
print('最优策略={}'.format(np.argmax(policy_vi,axis=1).reshape(4,4)))
ep_rew=[play_policy(env,policy_vi,render=False) for _ in range(100) ]
print('价值迭代平均奖励{}'.format(np.mean(ep_rew)))