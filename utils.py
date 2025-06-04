import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), gamma=None):
		self.max_size = max_size
		self.gamma = gamma
		self.ptr = 0
		self.size = 0
		self.pending = []

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.g_return = np.zeros((max_size, 1))
		self.g_est = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def push(self, state, action, next_state, reward, done, truncated):
		step = (state, action, next_state, reward, done)

		self.pending.append(step)

		if done or truncated:
			G = 0.
			T = 0.
			UT = 0.

			n = 0
			for sample in self.pending:
				s, a, ns, r, d = sample
				dr = r * pow(self.gamma, n)
				UT += r
				T += dr
				n += 1

			while len(self.pending) > 0:
				s, a, ns, r, d = self.pending.pop()

				if truncated:
					ave_reward = UT / n
					G = ave_reward
					
				G = r + self.gamma * G

				self.add(s, a, ns, r, float(d), G, g_est=truncated)


	def add(self, state, action, next_state, reward, done, g_return, g_est):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.g_return[self.ptr] = g_return
		self.g_est[self.ptr] = float(g_est)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.g_return[ind]).to(self.device),
			torch.FloatTensor(self.g_est[ind]).to(self.device)
		)