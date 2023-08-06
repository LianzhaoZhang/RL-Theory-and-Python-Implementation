# 代码清单2-3 & 2-4 & 2-5

import gym
import numpy as np

def play_once (env,policy):
    total_reward=0
    state=env.reset()
    cal=0
   # env.render()
    while 1:
        loc=np.unravel_index(state,env.shape)
        print('状态={}，位置={}'.format(state,loc),end=' ')

        action=policy[loc]
        state,reward,done,_=env.step(action)
        print('动作={}，奖励={}'.format(action,reward))
        total_reward += reward
        cal += 1

        if done :
            break

    return total_reward,cal

env=gym.make('CliffWalking-v0',render_mode='human')
print('观测空间={} \n 状态数量={} \n 动作空间={}  \n 动作数量={} \n 地图大小={}'.format(env.observation_space,env.nS,env.action_space,env.nA,env.shape))
actions = np.ones(env.shape,dtype=int)
actions[-1,:]=0
actions[:,-1]=2
optimal_policy=np.eye(4)[actions.reshape(-1)]

total,cal=play_once(env,actions)
print(total,cal)
env.close()