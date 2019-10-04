import torch
import torch.nn.functional as F
import torch.nn as nn


class QFCNet(nn.Module):
    """Actor (Policy) Model. 2x(FC hidden layer + Relu activation) + FC"""

    def __init__(self, state_size: int, action_size: int, seed: int = 0, fc1_units: int = 64,
                 fc2_units: int = 64):
        """Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param seed: Random seed
        :param fc1_units: Number of nodes in first hidden layer
        :param fc2_units: Number of nodes in second hidden layer
        """
        super(QFCNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))

    def action_size(self):
        return self.fc3.out_features


class QConvNet(nn.Module):
    """ A convolutional network """

    def __init__(self, action_size, seed: int = 0):
        super(QConvNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Conv. layers
        self.conv_1 = nn.Conv2d(1, 32, 9, stride=4, padding=4)
        self.conv_2 = nn.Conv2d(32, 64, 5, stride=3, padding=1)

        # Fully connected layers
        self.fc_layers_params = [(64 * 7 * 7, 256), (256, action_size)]
        self.fc1 = nn.Linear(*self.fc_layers_params[0])
        self.fc2 = nn.Linear(*self.fc_layers_params[1])

    def forward(self, state):
        """ forward ..."""
        # Applying Conv. section
        output = F.relu(self.conv_2(F.relu(self.conv_1(state))))
        # flatten the output and relay to the FC part
        output = output.view(-1, self.fc_layers_params[0][0])
        return self.fc2(F.relu(self.fc1(output)))

    def action_size(self):
        return self.fc2.out_features


class QNetFactory:
    """ Abstract factory"""


class QFCNetFactory(QNetFactory):
    """ Fully connected arch. Neural Net factory."""

    def __init__(self, state_size, action_size, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

    def build(self, device):
        """ build an FC based network """
        return QFCNet(self.state_size, self.action_size, self.seed).to(device)


class QConvNetFactory(QNetFactory):
    """ Fully connected arch. Neural Net factory."""

    def __init__(self, action_size, seed: int = 0):
        self.action_size = action_size
        self.seed = seed

    def build(self, device):
        """ build an FC based network """
        return QConvNet(self.action_size, self.seed).to(device)
