from typing import Tuple, List
from heapq import heappush, heappop
import random
import abc
from collections import namedtuple, deque
import torch


class Experience(namedtuple("Experience",
                            field_names=["state", "action", "reward", "next_state", "done",
                                         "error"])):

    def prioritizing(self, other):
        return self.error > other.error  # prioritizing higher error while learning

    def __lt__(self, other):
        return self.prioritizing(other)


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
    def replace(self, experiences: List[Experience]):
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
        self.memory.append(Experience(state, action, reward, next_state, done, error))

    def replace(self, experiences: List[Experience]):
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

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object.
        : param buffer_size: maximum size of buffer
        : param batch_size: size of each training batch
        : param seed: random seed
        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, batch_size, seed)
        self.memory = []

    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to the Replay buffer."""
        heappush(self.memory, Experience(state, action, reward, next_state, done, error))

    def replace(self, experiences: List[Experience]):
        """ Replacing (push back) the experience after re-evaluation)"""
        for experience in experiences:
            heappush(self.memory, experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = self.memory[-self.batch_size:-1]
        for _ in range(self.batch_size):
            experience = heappop(self.memory)
            if experience is not None:  # TODO: check the need of this test
                experiences.append(experience)
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