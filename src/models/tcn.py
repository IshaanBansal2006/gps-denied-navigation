from typing import List

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def chomp(self, x: torch.Tensor, chomp_size: int) -> torch.Tensor:
        if chomp_size == 0:
            return x
        return x[:, :, :-chomp_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp(out, self.conv1.padding[0])
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp(out, self.conv2.padding[0])
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + residual)


class TCNRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        channel_sizes: List[int] = [32, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 3,
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(channel_sizes):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(channel_sizes[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, time, features)
        # Conv1d expects: (batch, channels, time)
        x = x.transpose(1, 2)

        features = self.network(x)

        # Take final time step representation
        last_step = features[:, :, -1]

        out = self.head(last_step)
        return out