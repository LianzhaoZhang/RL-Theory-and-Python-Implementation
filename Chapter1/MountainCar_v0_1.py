import gym

env=gym.make('MountainCar-v0')
print('观测空间={}'.format(env.observation_space))
print('动作空间={}'.format(env.action_space))
print('观测范围={}~{}'.format(env.observation_space.low,env.observation_space.high))
print('动作数={}'.format(env.action_space.n))

class BeSpokerAgent():
    def __init__(self,env):
        pass

    def decide(self, observation):
        position,  volicity  =  observation
        lb = min( -0.09* (position+ 0.25)** 2+ 0.03,
                        0.3* (position + 0.9) ** 4 - 0.008)
        ub = -0.07*(position+0.38)**2+0.06
        if lb<volicity<ub:
            action=2
        else:
            action=0
        return action

    def learn(delf,*args):
        pass


def play_Montecarlo(env,agent,render=False,Train=False):
    episode_reward=0.
    observation=env.reset()
    while 1:
        if render:
            env.render()

        action=agent.decide(observation)
        next_observation,reward,done,_=env.step(action)
        episode_reward += reward
        if Train:
            agent.learn(observation,action,reward,done)
        if done:
            pass
        observation=next_observation
    return episode_reward

agent=BeSpokerAgent(env)
env.seed(0)
reward=play_Montecarlo(env,agent,True,False)
print('1个回合的奖励={}'.format(reward))
env.close()