"""A small two-source spectral masker.

The network reads a log magnitude spectrogram and predicts two masks that sum
to one. Each mask multiplies the mixture STFT. The inverse STFT is the
estimated stem, so the two waveforms are model output rather than a copy or a
fixed filter of the input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1),
            nn.GroupNorm(4, channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1),
            nn.GroupNorm(4, channels_out),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MaskSeparator(nn.Module):
    def __init__(self, n_fft: int = 512, hop_length: int = 128) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.encoder_1 = ConvBlock(1, 16)
        self.encoder_2 = ConvBlock(16, 32)
        self.pool = nn.MaxPool2d(2)
        self.middle = ConvBlock(32, 32)
        self.decoder_2 = ConvBlock(64, 16)
        self.decoder_1 = ConvBlock(32, 16)
        self.mask_head = nn.Conv2d(16, 2, kernel_size=1)

    def spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

    def waveform(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length,
        )

    def forward(self, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = self.spectrogram(mixture)
        magnitude = spectrum.abs().unsqueeze(1)
        network_input = torch.log1p(magnitude)
        frequency_pad = (4 - network_input.shape[-2] % 4) % 4
        time_pad = (4 - network_input.shape[-1] % 4) % 4
        padded = F.pad(network_input, (0, time_pad, 0, frequency_pad))

        skip_1 = self.encoder_1(padded)
        skip_2 = self.encoder_2(self.pool(skip_1))
        hidden = self.middle(self.pool(skip_2))
        hidden = F.interpolate(hidden, size=skip_2.shape[-2:], mode="nearest")
        hidden = self.decoder_2(torch.cat([hidden, skip_2], dim=1))
        hidden = F.interpolate(hidden, size=skip_1.shape[-2:], mode="nearest")
        hidden = self.decoder_1(torch.cat([hidden, skip_1], dim=1))
        masks = self.mask_head(hidden).softmax(dim=1)
        masks = masks[..., : magnitude.shape[-2], : magnitude.shape[-1]]

        krar = self.waveform(spectrum * masks[:, 0], mixture.shape[-1])
        other = self.waveform(spectrum * masks[:, 1], mixture.shape[-1])
        return krar, other
