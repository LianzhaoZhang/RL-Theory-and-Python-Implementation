def play_Montecarlo(env,agent,render=False,Train=False):
    episode_reward=0.
    observation=env.reset(seed=0)
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