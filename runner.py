import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.episode_steps=[]

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        #n_step 每一次完整实验的总steps.\
        zqq_count = 0
        while time_steps < self.args.n_steps:
            if np.mod(zqq_count, 50) == 0:
                print('Run {}, time_steps {}'.format(num, time_steps))
            zqq_count += 1
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            #n_eisode 'the number of episodes before once training'
            for episode_idx in range(self.args.n_episodes):
                episode, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        episode_reward,episode_steps = self.evaluate()
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)

    def evaluate(self):
        episode_rewards = 0
        episode_steps=0
        for epoch in range(self.args.evaluate_epoch):
        # for epoch in range(2):
            #2021.6.7 添加每个epoch的总steps
            _, episode_reward, total_step = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            episode_steps += total_step
        return episode_rewards / self.args.evaluate_epoch, episode_steps/self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_steps)), self.episode_steps)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_steps')
        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        np.save(self.save_path + '/episode_steps_{}'.format(num), self.episode_steps)
        plt.close()









