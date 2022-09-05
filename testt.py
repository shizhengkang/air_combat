from airwar_3D import AirWarEnv3D

env = AirWarEnv3D()
observation = env.reset()
for i in range(1000):
    env.render()
    if i < 100:
        action = [[[0.1, 0.2, 0, 0]], [[0.1, 0, 0.2, 0]]]
    else:
        action = [[[0, 0.2, 0, 1]], [[-0.1, 0, 0.2, 0]]]
    observation, reward, done, result = env.step(action)
    if done:
        break
env.close()
