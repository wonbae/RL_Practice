# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Observation {}".format(observation))
#             print("Reward is {}".format(reward))
#             print("Info is {}".format(info))
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# import gym
# env = gym.make('CartPole-v0')
# print(env.action_space)
# #> Discrete(2)
# print(env.observation_space)
# #> Box(4,)

# from gym import spaces
# space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
# x = space.sample()
# assert space.contains(x)
# assert space.n == 8

from gym import envs
print(envs.registry.all())