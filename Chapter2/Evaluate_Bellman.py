# 代码清单 2-6 2-7 2-8 用Bellman方程求解状态价值和动作价值
import numpy as np
import gym
import scipy.optimize as opt


def evaluate_bellman(env,policy,gamma=1):
    a,b=np.eye(env.nS),np.zeros(env.nS)
    for state in range(env.nS-1):
        for action in range (env.nA):
            pi=policy[state][action]
            for p , next_state,reward,done in env.P[state][action]:
                a[state,next_state] -= (pi*gamma*p)
                b[state] += (pi*reward*p)

    v=np.linalg.solve(a,b)
    q=np.zeros(shape=(env.nS,env.nA))
    for state in range(env.nS-1):
        for action in range (env.nA):
            for p, next_state,reward,done in env.P[state][action]:
                q[state][action] += (p*(reward+gamma*v[next_state]))
    return v,q

def optimal_bellman(env,gamma=1):
    p=np.zeros(shape=(env.nS,env.nA,env.nS))
    r=np.zeros(shape=(env.nS,env.nA))
    for state in range (env.nS-1):
        for action in range (env.nA):
            for prob,next_state,reward,done in env.P[state][action]:
                p[state,action,next_state] += prob
                r[state,action] += (reward*prob)
    c=np.ones(shape=env.nS)
    a_ub=gamma*p.reshape(-1,env.nS)-np.repeat(np.eye(env.nS),env.nA,axis=0)
    b_ub=-r.reshape(-1)
    a_eq=np.zeros(shape=(0,env.nS))
    b_eq=np.zeros(0)
    bounds=[(None,None),]*env.nS
    res=opt.linprog(c,a_ub,b_ub,bounds=bounds,method='interior-point')
    v=res.x
    q=r+gamma*np.dot(p,v)
    return v,q

'''
actions=np.ones(env.shape,dtype=int)
actions[-1,:]=0
actions[:,-1]=2
policy = np.random.uniform(size=(env.nS,env.nA))
policy = policy/np.sum(policy,axis=1)[:,np.newaxis]
optimal_policy=np.eye(4)[actions.reshape(-1)]

state_values,action_values=evaluate_bellman(env,policy)
optimal_state_values,optimal_action_values=evaluate_bellman(env,optimal_policy)

print('最优状态价值函数={}'.format(optimal_state_values))
print('最优动作价值函数={}'.format(optimal_action_values))
print('状态价值函数为{}'.format(state_values))
print('动作价值函数为{}'.format(action_values))
'''
env=gym.make('CliffWalking-v0')
optimal_state_values,optimal_action_values=optimal_bellman(env)
print('最优状态价值函数={}'.format(optimal_state_values))
print('最优动作价值函数={}'.format(optimal_action_values))

#用最优动作价值函数确定最优策略
optimal_actions=optimal_action_values.argmax(axis=1)
print('最优策略={}'.format(optimal_actions))