import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiscreteCritic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(DiscreteCritic, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.l3(a)


class DDQN(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		dev_mode=False
	):
	
		self.critic = DiscreteCritic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.epsilon_start = 1.0 #epsilon_start
		self.epsilon_end = 0.01 #epsilon_end
		self.decay_rate = 50_000
		self.epsilon = self.epsilon_start

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.dev_mode = dev_mode

		self.total_it = 0


	def select_action(self, state, noisy=False):
		if noisy and np.random.random() < self.epsilon:
			action = np.random.randint(0, self.max_action + 1)
		else:
			with torch.no_grad():
				state = torch.FloatTensor(state.reshape(1, -1)).to(device)
				actions = self.critic(state)
				action = actions.argmax(dim=1).cpu().data.numpy().flatten().item()

		return action


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done, G, G_est = replay_buffer.sample(batch_size)

		with torch.no_grad():
			action_values = self.critic(next_state)
			next_actions = action_values.argmax(dim=1, keepdim=True)
			target_Q = self.critic_target(next_state).gather(1, next_actions)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q = self.critic(state).gather(1, action.to(torch.long))

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update the frozen target models
		if self.total_it % 1000 == 0:
			self.critic_target.load_state_dict(self.critic.state_dict())

		# Update epsilon for exploration
		self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1 * self.total_it / self.decay_rate)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		