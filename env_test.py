
from droplet_gym.envs import *
import time
env = Dropletenv(5,5,3)
# env=wrappers.Monitor(env,'./experiment-1')
# print(env.state)
agent_num = env.agent_number
for i in range(10):
    env.reset()
    # action=[1]*env.agent_number
    for j in range(100):
        action = []
        for k in range(agent_num):
            action.append(env.action_space.sample())
        # print(action)
        reward_n, done_n, info_n = env.step(action)
        print('-----------------')
        print(j)
        # print(reward_n)
        # env.render()
        print(done_n)
        print(reward_n)
        # time.sleep(1)
        # print(env.state)
        # print(env.global_state.shape)
        # done_n=np.array(done_n)
        # flag=np.all(done_n)
        # if flag:
        #     print("has break")
        #     break
    # print(obs_n)
env.close()
## 草拟吗