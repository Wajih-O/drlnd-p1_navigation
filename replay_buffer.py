from typing import Tuple
from heapq import heappush, heappop, heappushpop, nsmallest
import random
import abc
from collections import namedtuple, deque
import torch


class Experience(namedtuple("Experience",
                            field_names=['error', 'state', 'action', 'reward', 'next_state',
                                         'done'])):
    def __lt__(self, other):
        return self.error > other.error  # reversed to sort highest error first


class BaseReplayBuffer:
    """ a Base class for ReplayBuffer"""

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object.
        : param buffer_size: maximum size of buffer
        : param batch_size: size of each training batch
        : param seed: random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        random.seed(seed)

    @abc.abstractmethod
    def sample(self) -> Tuple[torch.tensor]:
        """" Define the sampling variant for the ReplayBuffer"""

    @abc.abstractmethod
    def __len__(self) -> int:
        """ Return the current size of internal memory."""

    @abc.abstractmethod
    def replace(self, experience: Experience):
        """ Replacing (push back) the experience after re-evaluation)"""


class BaseReplayBufferFactory:
    """" Replay Buffer factory"""

    @abc.abstractmethod
    def build(self, buffer_size: int, batch_size: int, seed: int = 0) -> BaseReplayBuffer:
        """ Build """
        # TODO refactoring:  get rid of batch size from attributes


class UniformReplayBuffer(BaseReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object.
        : param buffer_size: maximum size of buffer
        : param batch_size: size of each training batch
        : param seed: random seed
        """
        super(UniformReplayBuffer, self).__init__(buffer_size, batch_size, seed)
        self.memory = deque(maxlen=self.buffer_size)

    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""
        self.memory.append(Experience(error, state, action, reward, next_state, done))

    def replace(self, experience: Experience):
        """ Not supported for this Replay-buffer (the behavior is do-nothing)"""

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return [experience for experience in random.sample(self.memory, k=self.batch_size) if
                experience is not None]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int, alpha: float = 1.0,
                 decay: float = .999999, min_alpha=.4):
        """Initialize a ReplayBuffer object.
        : param buffer_size: maximum size of buffer
        : param batch_size: size of each training batch
        : param seed: random seed
        :param alpha:
        :param decay: alpha decay

        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, batch_size, seed)
        self.memory = []
        self.alpha = alpha
        self.decay = decay
        self.min_alpha = min_alpha

    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to the Replay buffer."""
        if len(self.memory) < self.buffer_size:
            heappush(self.memory, Experience(error, state, action, reward, next_state, done, ))
        else:
            heappushpop(self.memory, Experience(error, state, action, reward, next_state, done))

    def replace(self, experience: Experience):
        """ Replacing (push back) the experience after re-evaluation)"""
        heappush(self.memory, experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        highest_error_experience = nsmallest(self.batch_size, self.memory, key=lambda item: item.error)
        uniform_random_sample = random.sample(self.memory, k=self.batch_size)
        experiences = [
            uniform_random_sample[index] if random.random() < self.alpha else
            heappop(highest_error_experience) for index in range(self.batch_size)]
        # update alpha
        self.alpha = max(self.alpha * self.decay, self.min_alpha)
        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class UniformReplayBufferFactory(BaseReplayBufferFactory):
    @abc.abstractmethod
    def build(self, buffer_size: int, batch_size: int, seed: int = 0) -> BaseReplayBuffer:
        return UniformReplayBuffer(buffer_size, batch_size, seed)


class PrioritizedReplayBufferFactory(BaseReplayBufferFactory):
    @abc.abstractmethod
    def build(self, buffer_size: int, batch_size: int, seed: int = 0) -> BaseReplayBuffer:
        return PrioritizedReplayBuffer(buffer_size, batch_size, seed)
