""" DQN Based Agent for Banana collector env."""

import random
from typing import Tuple
from collections import namedtuple, deque

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model. 2x(FC hidden layer + Relu activation) + FC"""

    def __init__(self, state_size: int, action_size: int, seed: int, fc1_units: int = 64,
                 fc2_units: int = 64):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param fc1_units: Number of nodes in first hidden layer
        :param fc2_units: Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))


class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int = 0, batch_size: int = 64,
                 step_to_update: int = 5, buffer_size: int = int(1e5), gamma: float = .99,
                 lr: float = 5e-4, tau: float = 1e-3, episodes_window_size: int = 100):
        """Initialize an Agent object.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param seed: initializes pseudo random gen. random.seed(seed)
        :param batch_size: training batch_size
        :param step_to_update: after which the local network is trained/and the target is updated
        :param buffer_size:  replay buffer size
        :param gamma: reward discount factor
        :param lr: learning rate
        :param tau: Q soft update tau
        :param episodes_window_size: deque storing the [episodes_window_size] last episodes score
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.batch_size = batch_size  # initialize learning batch size
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step (for updating every self.steps_to_update steps)
        self.t_step = 0
        self.steps_to_update = step_to_update

        # Agent history stats (collected during the env. exploration/solving)
        self.scores = []  # list containing scores from each episode
        self.scores_window = deque(maxlen=episodes_window_size)

    @property
    def buffer_size(self):
        """ gets the current reply buffer size"""
        return len(self.memory)

    def explore(self, env, n_episodes=1800, max_t=1000, brain_name=None,
                eps_start=1.0, eps_end=0.01, eps_decay=0.99):
        """ explore/solve the env"""
        brain_name_ = brain_name
        if brain_name_ is None:
            brain_name_ = env.brain_names[0]

        eps = eps_start  # initialize eps

        for i_episode in range(1, n_episodes + 1):
            score = 0
            env_info = env.reset(train_mode=True)[brain_name_]  # reset the environment
            state = env_info.vector_observations[0]

            for _ in range(max_t):
                action = self.get_eps_greedy_action(state, eps)  # pick action (epsilon-greedy way)

                env_info = env.step(action)[brain_name_]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score

                self.step(state, action, reward, next_state,
                          done)  # update agent with the collected experience
                state = next_state  # roll over the state to next time step
                if done:
                    break

            self.scores_window.append(score)  # save most recent score
            self.scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                               np.mean(self.scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                                   np.mean(self.scores_window)))
            if np.mean(self.scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    i_episode, np.mean(
                        self.scores_window)))
                torch.save(self.qnetwork_local.state_dict(), 'checkpoint.pth')
                break

    def step(self, state, action, reward, next_state, done):
        """ Update the replay buffer and eventually train (each self.steps_to_update)"""
        self.memory.add(state, action, reward, next_state,
                        done)  # update replay buffer with the exp.
        # Learn every self.steps_to_update steps.
        self.t_step = (self.t_step + 1) % self.steps_to_update
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                # enough experience were sampled to build the batch
                self.learn(self.memory.sample())

    def get_eps_greedy_action(self, state: np.ndarray, eps: float = 0.) -> int:
        """ Returns an action for given state as per current policy (using an epsilon greedy).

        :param state: current state
        :param eps: epsilon, for epsilon-greedy action selection (default greedy)
        :return : action index as int
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())

        return random.choice(np.arange(self.action_size))

    def learn(self, experiences: Tuple[torch.Tensor]):
        """Update value parameters using given batch of experience tuples.

        :param experiences: tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft-update target network
        DQNAgent.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: weights will be copied from
        :param  target_model: weights will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object.

        : param action_size: dimension of each action
        : param buffer_size: maximum size of buffer
        : param batch_size: size of each training batch
        : param seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state",
                                                  "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
