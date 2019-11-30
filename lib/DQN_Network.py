import torch
import numpy as np
from torch import nn
from torch import optim

class DQN(nn.Module):

    def __init__(self, input_shape, n_actions, learning_rate):
        super(DQN, self).__init__()

        channels = input_shape[0]

        # First conv layer
        conv_out_channels = 32  # <-- Filters in your convolutional layer
        kernel_size = 8  # <-- Kernel size
        conv_stride = 4  # <-- Stride
        conv_pad = 0  # <-- Padding

        # Second conv layer
        conv_out_channels_2 = 64  # <-- Filters in your convolutional layer
        kernel_size_2 = 4  # <-- Kernel size
        conv_stride_2 = 2  # <-- Stride
        conv_pad_2 = 0  # <-- Padding

        # Third conv layer
        conv_out_channels_3 = 64  # <-- Filters in your convolutional layer
        kernel_size_3 = 3  # <-- Kernel size
        conv_stride_3 = 1  # <-- Stride
        conv_pad_3 = 0  # <-- Padding

        self.conv = nn.Sequential(
            nn.Conv2d(channels, conv_out_channels, kernel_size, conv_stride, conv_pad),
            nn.ReLU(),
            nn.Conv2d(conv_out_channels, conv_out_channels_2, kernel_size_2, conv_stride_2, conv_pad_2),
            nn.ReLU(),
            nn.Conv2d(conv_out_channels_3, conv_out_channels_3, kernel_size_3, conv_stride_3, conv_pad_3),
            nn.ReLU()
        )


        out_features = self.conv_out_features(input_shape)

        self.out = nn.Sequential(
            nn.Linear(out_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        return nn.Softmax(dim=1)(self.out(x))

    def conv_out_features(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def calculate_loss(self, batch, net, target_net, GAMMA, only_DQN, device="cpu"):
        """
        Calculate MSE between actual state action values,
        and expected state action values from DQN
        """
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done = torch.tensor(dones).to(device)

        if only_DQN:
            state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
            next_state_values = net(next_states_v).max(1)[0]
        else:
            state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
            next_state_values = target_net(next_states_v).max(1)[0]

        next_state_values[done] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v

        return nn.MSELoss()(state_action_values, expected_state_action_values)