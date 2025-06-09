import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import DDQN

from plot_results import plot_results
from reports import gen_detailed_report, export_results

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, gamma=1.):
	eval_env = gym.make(env_name)
	eval_env.reset(seed=seed + 100)

	avg_reward = 0.
	disc_reward = 0.

	for _ in range(eval_episodes):
		state, done, truncated = eval_env.reset(), False, False
		state = np.array(state[0], dtype=np.float32)
		episode_step = 0

		while not done and not truncated:
			action = policy.select_action(np.array(state))
			state, reward, done, truncated, _ = eval_env.step(action)

			avg_reward += reward
			disc_reward += reward * pow(gamma, episode_step)
			episode_step += 1

	avg_reward /= eval_episodes
	disc_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} ({disc_reward:.3f})")
	print("---------------------------------------")
	return avg_reward, disc_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="LunarLanderContinuous-v3")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--plot_results", action="store_true")      # Generate a simple plot of the latest raw training results
	parser.add_argument("--plot_ave", action="store_true")          # Generate a simple plot of the latest average training results
	parser.add_argument("--plot_discount", action="store_true")     # Flag for plot function to show discounted returns instead of total returns
	parser.add_argument("--dev", action="store_true")               # Flag to enable development mode features
	parser.add_argument("--gen_report", action="store_true")        # Generate a detailed report of the results
	parser.add_argument("--results_path", default="./results")      # Path to the input results for the detailed report
	parser.add_argument("--export_results", action="store_true")    # Export all results in CSV format
	args = parser.parse_args()

	if args.gen_report:
		gen_detailed_report(
			policies=["TD3", "TD3-DEV"],
			envs=["BipedalWalker-v3",
		 		"LunarLanderContinuous-v3",
				"Humanoid-v5",
				"HalfCheetah-v5", 
				"Walker2d-v5",
				"Hopper-v5",
				"Ant-v5",
				"Reacher-v5",
				"InvertedPendulum-v5",
				"InvertedDoublePendulum-v5"],
			seeds=range(10),
			path=args.results_path
		)
		exit(0)

	if args.dev:
		args.policy += "-DEV"

	if args.policy.startswith("DDQN"):
		args.batch_size = 64

	if args.export_results:
		export_results(args.results_path, args.discount, args.eval_freq)
		exit(0)

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.plot_results:
		while True:
			plot_results(f"{args.env}_{args.seed}", eval_freq=args.eval_freq, show_ave=False, discounted=args.plot_discount)
		exit(0)

	if args.plot_ave:
		while True:
			plot_results(f"{args.env}_{args.seed}", eval_freq=args.eval_freq, show_train=False, discounted=args.plot_discount)
		exit(0)

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	if type(env.action_space) == gym.spaces.Discrete:
		discrete_action = True
		action_dim = env.action_space.n
		max_action = env.action_space.n + env.action_space.start - 1
		#print(f"Discrete action space with {action_dim} actions, max action: {max_action}")
		#exit(0)
	else:
		discrete_action = False
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy.startswith("TD3"):
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["dev_mode"] = args.dev
		policy = TD3.TD3(**kwargs)
	elif args.policy.startswith("DDQN"):
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["dev_mode"] = args.dev
		policy = DDQN.DDQN(**kwargs)
	elif args.policy.startswith("OurDDPG"):
		kwargs["dev_mode"] = args.dev
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim if not discrete_action else 1, gamma=args.discount)
	
	# Evaluate untrained policy
	ave_reward, disc_reward = eval_policy(policy, args.env, args.seed, gamma=args.discount)
	evaluations = [ave_reward]
	disc_evaluations = [disc_reward]

	state, done, truncated = env.reset(), False, False
	state = np.array(state[0], dtype=np.float32)
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			if discrete_action:
				action = policy.select_action(np.array(state), noisy=True)
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, truncated, _ = env.step(action) 

		# Store data in replay buffer
		replay_buffer.push(state, action, next_state, reward, done, truncated)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done or truncated: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}\t{'Done' if done else 'Truncated'}")
			# Reset environment
			state, done, truncated = env.reset(), False, False
			state = np.array(state[0], dtype=np.float32)
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			ave_reward, disc_reward = eval_policy(policy, args.env, args.seed, gamma=args.discount)
			evaluations.append(ave_reward)
			disc_evaluations.append(disc_reward)
			np.save(f"./results/{file_name}", evaluations)
			np.save(f"./results/{file_name}_disc", disc_evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
