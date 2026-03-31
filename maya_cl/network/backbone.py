# backbone.py -- Maya-Manas network architecture (Paper 7)
# Extends Maya-Viveka/Chitta backbone with O-LIF oscillatory threshold modulation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional

from maya_cl.utils.config import (
    CONV1_CHANNELS, CONV2_CHANNELS, CONV3_CHANNELS,
    FC1_SIZE, NUM_CLASSES, V_THRESHOLD, V_RESET, TAU_MEMBRANE,
    PROTOTYPE_DIM, A_MANAS, T_STEPS,
)
from maya_cl.plasticity.manas import ManasGate


class LIFLayer(nn.Module):
    """Standard LIF layer -- used for output head, unchanged from P6."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc  = nn.Linear(in_features, out_features, bias=False)
        self.lif = neuron.LIFNode(
            tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
            v_reset=V_RESET, detach_reset=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))


class MayaManasLIFLayer(nn.Module):
    """
    O-LIF layer -- fc1 replacement for Paper 7.

    Applies Manas oscillatory threshold schedule across T_STEPS.
    Each timestep t uses a different threshold:
        V_threshold(t) = V_base + A_manas * cos(pi * t / (T_STEPS - 1))

    Expects input [T, B, in_features] -- iterates manually per timestep.
    Records which neurons fired during peak-aligned timesteps for Manas-GANE.
    """

    def __init__(self, in_features: int, out_features: int,
                 a_manas: float = A_MANAS):
        super().__init__()
        self.fc  = nn.Linear(in_features, out_features, bias=False)
        self.lif = neuron.LIFNode(
            tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
            v_reset=V_RESET, detach_reset=True
        )
        self.gate         = ManasGate(t_steps=T_STEPS, v_base=V_THRESHOLD, a_manas=a_manas)
        self.out_features = out_features
        self.in_features  = in_features
        self.peak_active: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, in_features]
        returns: [T, B, out_features]
        """
        T = x.shape[0]
        outputs = []
        peak_fire_accum = torch.zeros(
            self.out_features, device=x.device, dtype=torch.float32)

        functional.reset_net(self.lif)

        for t in range(T):
            self.lif.v_threshold = self.gate.get_threshold(t)
            x_t   = x[t]               # [B, in_features]
            pre_t = self.fc(x_t)       # [B, out_features]
            out_t = self.lif(pre_t)    # [B, out_features]
            outputs.append(out_t)

            if self.gate.is_peak_aligned(t):
                peak_fire_accum += out_t.detach().mean(dim=0)

        self.lif.v_threshold = V_THRESHOLD
        self.peak_active = peak_fire_accum > 0.0
        return torch.stack(outputs, dim=0)   # [T, B, out_features]


class OrthogonalPrototypeHead(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        raw = torch.randn(num_classes, dim)
        if num_classes <= dim:
            Q, _ = torch.linalg.qr(raw.T)
            prototypes = Q.T[:num_classes]
        else:
            prototypes = F.normalize(raw, dim=1)
        self.register_buffer('prototypes', prototypes)
        self.num_classes = num_classes
        self.dim         = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        return (x_norm @ p_norm.T) * 10.0


class MayaManasNet(nn.Module):
    """
    Maya-Manas network -- P7 backbone.
    Conv layers in multi-step mode. fc1 is O-LIF with manual timestep loop.
    """

    def __init__(self, use_orthogonal_head: bool = False,
                 a_manas: float = A_MANAS):
        super().__init__()

        self.conv1 = nn.Sequential(
            layer.Conv2d(3, CONV1_CHANNELS, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(CONV1_CHANNELS),
            neuron.LIFNode(tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
                           v_reset=V_RESET, detach_reset=True)
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(CONV2_CHANNELS),
            neuron.LIFNode(tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
                           v_reset=V_RESET, detach_reset=True),
            layer.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            layer.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(CONV3_CHANNELS),
            neuron.LIFNode(tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
                           v_reset=V_RESET, detach_reset=True),
            layer.MaxPool2d(2, 2)
        )

        # conv_out_dim: CONV3 * 8 * 8 = 128 * 8 * 8 = 8192
        conv_out_dim = CONV3_CHANNELS * 8 * 8
        self.fc1 = MayaManasLIFLayer(conv_out_dim, FC1_SIZE, a_manas=a_manas)

        if use_orthogonal_head:
            self.fc_out = OrthogonalPrototypeHead(NUM_CLASSES, PROTOTYPE_DIM)
        else:
            self.fc_out = nn.Linear(FC1_SIZE, NUM_CLASSES, bias=False)

        functional.set_step_mode(self.conv1, step_mode='m')
        functional.set_step_mode(self.conv2, step_mode='m')
        functional.set_step_mode(self.conv3, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, B, C, H, W]"""
        x = self.conv1(x)               # [T, B, 64, 32, 32]
        x = self.conv2(x)               # [T, B, 64, 16, 16]
        x = self.conv3(x)               # [T, B, 128, 8, 8]
        T, B = x.shape[0], x.shape[1]
        x = x.reshape(T, B, -1)         # [T, B, 8192] -- safe TB-preserving flatten
        x = self.fc1(x)                 # [T, B, 2048] -- O-LIF per-timestep
        x = x.mean(dim=0)              # [B, 2048]
        return self.fc_out(x)

    def reset(self):
        functional.reset_net(self.conv1)
        functional.reset_net(self.conv2)
        functional.reset_net(self.conv3)
        functional.reset_net(self.fc1.lif)

    def get_fc1_peak_active(self) -> torch.Tensor:
        return self.fc1.peak_active
