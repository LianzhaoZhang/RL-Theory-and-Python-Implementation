#代码清单3-1～3-8，有模型策略迭代求解

from email import policy
from random import random
import gym 
import numpy as np

def play_policy(env,policy,render=True):   #代码清单3-1
    total_reward=0
    observation=env.reset()
    while 1:
        if render:
            env.render()

        action=np.random.choice(env.action_space.n,p=policy[observation])
        next_observation,reward,done,info=env.step(action)
        total_reward += reward
        if done :
            break

        observation=next_observation
    return total_reward

def v2q(env,v,s=None,gamma=1):#根据状态价值函数计算动作价值函数
    if s is not None:
        q=np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob,next_state,reward,done in env.P[s][a]:
                q[a] += prob*(reward+gamma*v[next_state]*(1.-done))
    else:
        q=np.zeros(shape=(env.observation_space.n,env.action_space.n))
        for s in range(env.observation_space.n):
            q[s]=v2q(env,v,s,gamma)
    return q

def evaluate_policy(env,policy,gamma=1,tolerant=1e-6):#迭代计算给定策略policy的状态价值
    v=np.zeros(env.observation_space.n)
    while True:
        delta=0
        for s in range(env.observation_space.n):
            vs=sum(policy[s]*v2q(env,v,s,gamma))
            delta=max(delta,abs(v[s]-vs))
            v[s]=vs
        if delta<tolerant:
            break
    return v

def improve_policy(env,v,policy,gamma=1): #策略改进
    optimal=True
    for s in range(env.observation_space.n):
        q=v2q(env,v,s,gamma)
        a=np.argmax(q)
        if policy[s][a] !=1.:
            optimal=False
            policy[s]=0.
            policy[s][a]=1.
    return optimal


def itrate_policy(env,gamma=1.,tolerant=1e-6):
    policy=np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
    while 1:
        v=evaluate_policy(env,policy,gamma,tolerant)
        if improve_policy(env,v,policy):
            break
    return policy,v



env=gym.make('FrozenLake-v1')
policy_pi,v_pi=itrate_policy(env)
print('状态价值函数={}'.format(v_pi.reshape(4,4)))
print('最优策略={}'.format(np.argmax(policy_pi,axis=1).reshape(4,4)))

#env=env.unwrapped
#print('动作空间={}'.format(env.action_space))
#print('状态空间={}'.format(env.observation_space))

#random_policy=np.ones((env.observation_space.n,env.action_space.n))/env.action_space.n
#v_random=evaluate_policy(env,random_policy)
#q_random=v2q(env,v_random)
#print('随即状态价值为{} \n 随机动作价值为{}'.format(v_random,q_random))

#poli=random_policy.copy()
#optimal=improve_policy(env,v_random,poli)
#if optimal:
#    print('无更新，最优策略为：')
#else:
#    print('有更新，最优策略为：')
#print(poli)



#re=[play_policy(env,random_policy) for _ in range(100)]
#print('随即策略平均奖励:{}'.format(np.mean(re)))

#env.close()