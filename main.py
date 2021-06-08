from runner import Runner
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from droplet_gym.envs import *

if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        args.alg = 'qmix'
        args.cuda = True
        args = get_mixer_args(args)
        # if args.alg.find('coma') > -1:
        #     args = get_coma_args(args)
        # elif args.alg.find('central_v') > -1:
        #     args = get_centralv_args(args)
        # elif args.alg.find('reinforce') > -1:
        #     args = get_reinforce_args(args)
        # else:
        #     args = get_mixer_args(args)
        # if args.alg.find('commnet') > -1:
        #     args = get_commnet_args(args)
        # if args.alg.find('g2anet') > -1:
        #     args = get_g2anet_args(args)
        #     env = StarCraft2Env(map_name=args.map,
        #                         step_mul=args.step_mul,
        #                         difficulty=args.difficulty,
        #                         game_version=args.game_version,
        #                         replay_dir=args.replay_dir)
        env = Dropletenv(5, 5, 2)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            average_rewad_epoch, episode_steps = runner.evaluate()
            print('The averege total_rewards of {} is  {}'.format(args.alg, average_rewad_epoch))
            print('The each epoch total_steps is:')
            print(episode_steps)
            break
        env.close()
