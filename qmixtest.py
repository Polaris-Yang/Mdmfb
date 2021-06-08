from runner import Runner
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from droplet_gym.envs import *



def evaluate(self):
    episode_rewards = 0
    average_steps=0
    for epoch in range(self.args.evaluate_epoch):
        _, episode_reward, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
        episode_rewards += episode_reward
    return episode_rewards / self.args.evaluate_epoch

if __name__ == '__main__':
    args = get_common_args()
    args.alg = 'qmix'
    args.cuda = True
    args = get_mixer_args(args)
    env = Dropletenv(10, 10, 2)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    runner = Runner(env, args)
    win_rate, _ = runner.evaluate()
    env.close()